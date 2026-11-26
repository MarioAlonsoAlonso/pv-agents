"""The stages from the coordinator: the monitor's store and lock, the triage of what is waiting,
the tools for every stage, the confirmation of big jobs and the conversation. A toy Bluesky,
a fake triage and a scripted coordinator: no network, no 27B, no data/."""

import json
import re
from datetime import UTC, datetime

import pytest

import crawler.monitor as monitor
from agent.react_agent import Brain
from coordinator import etapas
from coordinator.agent import Conversacion, coordinar, describe
from coordinator.cases import CaseStore
from crawler.sources import FrozenSource, Post

BRANDS = {"sertralina": ["Besitran"]}


def _p(pid, text, date, thread=None, parent=None, replies=0):
    return Post(pid, text, f"{date}T10:00:00Z", thread or pid, parent, replies, "es")


POSTS = [
    _p("a1", "Llevo un mes con sertralina y tengo muchísimo sueño", "2025-03-01", replies=1),
    _p("a2", "A mí también me pasó", "2025-03-02", "a1", "a1"),
    _p("b1", "El Besitran me sienta genial, ningún problema", "2025-04-01"),
    _p("c1", "jaja necesito una sertralina para el lunes", "2025-05-01"),
]


class _LiveLike(FrozenSource):
    def uri(self, pid):
        return f"at://{pid}"

    def remember(self, pid, uri):
        pass


class _FakeTriage:
    """Answers like the triage: an adverse event if the post talks of sleep."""

    name, last_prompt_tokens = "fake-triage", 100

    def generate(self, messages, max_new_tokens=512, enable_thinking=False, **kw):
        prompt = messages[-1]["content"]
        out = []
        for block in prompt.split("### id: ")[1:]:
            i = block.split()[0]
            post = block.split("Post:", 1)[1]
            if "sueño" in post:
                out.append(
                    {
                        "id": i,
                        "es_efecto_adverso": "si",
                        "reacciones": "muchísimo sueño",
                        "paciente": "autor",
                        "edad": "desconocido",
                        "sexo": "desconocido",
                        "grave": "no_consta",
                        "desenlace": "desconocido",
                    }
                )
            else:
                motivo = "broma_opinion_noticia" if "jaja" in post else "sin_reaccion"
                out.append({"id": i, "es_efecto_adverso": "no", "motivo_no": motivo})
        return json.dumps(out, ensure_ascii=False), 20

    def count_tokens(self, text):
        return len(text.split())


class _ScriptedLLM:
    name, last_prompt_tokens = "scripted", 50

    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, messages, max_new_tokens=512, enable_thinking=False, **kw):
        return self.replies.pop(0), 10

    def count_tokens(self, text):
        return len(text.split())


def _call(name, **arguments):
    return (
        f"Thought: ok\n<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"
    )


@pytest.fixture
def live(tmp_path, monkeypatch):
    """The monitor's files in tmp_path, a toy Bluesky and the fake triage."""
    monkeypatch.setattr(monitor, "STATE_DIR", tmp_path / "monitor")
    monkeypatch.setattr(monitor, "load_brands", lambda: BRANDS)
    monkeypatch.setattr("crawler.cima.load_brands", lambda: BRANDS)
    monkeypatch.setattr(etapas, "SOURCE_FACTORY", lambda: _LiveLike(POSTS))
    monkeypatch.setattr(etapas, "TRIAGE_LLM_FACTORY", lambda: _FakeTriage())
    return tmp_path


def _store(tmp_path):
    """The live store only (no evaluation sample), as the coordinator builds it."""
    f = monitor.files()
    return (
        CaseStore.from_files(
            [(f["menciones"], f["triaje"], "vigilante")], meddra_file=tmp_path / "meddra.json"
        )
        if f["menciones"].exists()
        else CaseStore(sources=[(f["menciones"], f["triaje"], "vigilante")])
    )


# ---------------------------------------------------------------------------- monitor


def test_crawl_saves_once_and_triage_takes_what_is_waiting(live):
    now = datetime(2025, 6, 1, tzinfo=UTC)
    first = monitor.crawl_and_save(_LiveLike(POSTS), ["sertralina"], since="2025-01-01", now=now)
    assert first["nuevas"] == 3 and first["por_origen"]["busqueda_fija"] == 3  # a1, b1, c1
    again = monitor.crawl_and_save(_LiveLike(POSTS), ["sertralina"], since="2025-01-01", now=now)
    assert again["nuevas"] == 0  # the same posts are not saved twice
    assert len(monitor.pending_mentions(["sertralina"])) == 3
    assert monitor.pending_mentions(["sertralina"], already={m.id for m in []}) != []
    done = monitor.triage_pending(
        ["sertralina"], llm_factory=lambda: _FakeTriage(), progress=lambda s: None
    )
    assert done["triadas"] == 3 and done["efectos_adversos"] == 1
    assert monitor.pending_mentions(["sertralina"]) == []  # nothing is triaged twice
    assert not monitor.files()["lock"].exists()  # released


def test_only_one_writer_at_a_time(live):
    with monitor.exclusive("rastreo de prueba"):
        assert monitor.running()["que"] == "rastreo de prueba"
        with pytest.raises(monitor.Busy, match="ya hay algo en marcha"):
            monitor.crawl_and_save(_LiveLike(POSTS), ["sertralina"], since="2025-01-01")
    assert monitor.running() is None


# ---------------------------------------------------------------------------- store and tools


def test_the_store_keeps_every_stage_and_the_tools_read_them(live, tmp_path):
    monitor.crawl_and_save(_LiveLike(POSTS), ["sertralina"], since="2025-01-01")
    f = monitor.files()
    store = CaseStore.from_files(
        [(f["menciones"], f["triaje"], "vigilante")], meddra_file=tmp_path / "m.json"
    )
    assert len(store.mentions) == 3 and store.cases == {}  # found, not triaged yet
    from coordinator.tools import make_registry

    reg = make_registry(store)
    state = reg.call("estado_del_sistema", {"farmaco": "sertralina"}).result
    s = state["farmacos"]["sertralina"]
    assert s["menciones"] == 3 and s["esperando_triaje"] == 3
    assert any("triar(farmaco='sertralina')" in x for x in s["que_falta"])
    never = reg.call("estado_del_sistema", {"farmaco": "lorazepam"}).result
    assert "nunca se ha rastreado" in never["farmacos"]["lorazepam"]["que_falta"][0]
    monitor.triage_pending(["sertralina"], llm_factory=lambda: _FakeTriage(), progress=print)
    store.reload()
    assert len(store.cases) == 1 and len(store.answers) == 3
    found = reg.call("listar_menciones", {"farmaco": "sertralina"}).result
    assert found["coinciden"] == 3 and found["por_triaje"] == {
        "efecto_adverso": 1,
        "descartada": 2,
    }
    jokes = reg.call("listar_descartes", {"motivo": "broma_opinion_noticia"}).result
    assert jokes["descartadas"] == 2 and jokes["coinciden"] == 1
    assert jokes["por_motivo"] == {"sin_reaccion": 1, "broma_opinion_noticia": 1}
    only = reg.call("listar_menciones", {"triaje": "efecto_adverso", "buscar": "sueño"}).result
    assert only["coinciden"] == 1 and only["menciones"][0]["fuente"] == "vigilante"
    assert "error" in reg.call("rastrear", {"farmaco": "omeprazol", "desde": "2025-01-01"}).result
    assert "error" in reg.call("rastrear", {"farmaco": "tramadol", "desde": "2019-01-01"}).result


def test_a_mention_in_two_sources_counts_once(tmp_path):
    from crawler.store import RawMention

    m = RawMention(
        id="M-0000000001",
        post_id="p1",
        hilo_id="p1",
        farmaco="sertralina",
        tipo_mencion="nombre",
        texto="sueño",
        fecha="2025-01-01T00:00:00Z",
        encontrado_por="busqueda_fija",
    )
    store = CaseStore.build(
        [([m], {m.id: {"es_efecto_adverso": "no"}}), ([m], {})], names=["muestra", "vigilante"]
    )
    assert len(store.mentions) == 1 and store.origin[m.id] == "muestra"
    assert store.triage_state(m.id) == "descartada"


# ---------------------------------------------------------------------------- confirmation


def test_estimates_and_plan():
    assert etapas.estimate_crawl(30)["menciones_aprox"] == 45  # 1.5 a day
    assert "M tokens" in etapas.estimate_triage(500)["texto"]
    assert etapas.watched("Besitran") == ("sertralina", None)
    assert etapas.watched("omeprazol")[1] is not None
    assert etapas.parse_day("2026-13-01")[1] is not None


def test_a_big_search_waits_for_the_technician_then_runs(live, tmp_path):
    store = _store(tmp_path)
    llm = _ScriptedLLM(
        [
            _call("rastrear", farmaco="sertralina", desde="2025-01-01", con_agente=False),
            _call("final_answer", answer="Es un trabajo grande, ¿lo lanzo?"),
            # after the confirmation: triage (3 mentions: small, runs at once), then answer
            _call("triar", farmaco="sertralina"),
            _call("final_answer", answer="Rastreado y triado: 1 caso."),
        ]
    )
    conv = Conversacion(store, brain="fake", brains={"fake": Brain(llm=llm)}, coder=str)
    first = conv.preguntar("Busca todo lo de la sertralina desde enero de 2025")
    assert [a.name for a in conv.pendientes] == ["rastrear"]
    assert "confirmación" in first.steps[0].observation
    # without the agent the crawl is free; the estimate still says what triaging will cost
    assert "sin el agente (gratis)" in describe(conv.pendientes[0], store)
    assert not monitor.files()["menciones"].exists()  # nothing ran before the "yes"
    job = conv.confirmar(0)  # a crawl runs in the background: the chat is free meanwhile
    job.hilo.join(timeout=30)
    assert job.terminado and conv.terminados() == [job] and "menciones_nuevas" in job.resultado
    second = conv.informar(job)  # the coordinator is told, and goes on with the triage
    assert second.session_id == first.session_id  # the same conversation
    assert len(store.mentions) == 3 and len(store.cases) == 1  # crawled, then triaged
    assert "Rastreado" in second.final_answer and conv.pendientes == []
    assert conv.terminados() == []  # told once


def test_in_an_exam_even_a_small_search_is_only_proposed(live, tmp_path):
    store = _store(tmp_path)
    llm = _ScriptedLLM(
        [
            _call("triar", farmaco="sertralina"),
            _call("final_answer", answer="Pendiente."),
        ]
    )
    monitor.crawl_and_save(_LiveLike(POSTS), ["sertralina"], since="2025-01-01")
    store.reload()
    response = coordinar(
        "Tría la sertralina",
        store,
        brain="fake",
        brains={"fake": Brain(llm=llm)},
        coder=str,
        hold_stages=True,
    )
    assert [a.name for a in response.pending_actions] == ["triar"] and store.answers == {}


def test_the_chat_reads_yes_and_no():
    from coordinator.chat import command

    assert command("sí") == ("si", 0) and command("Sí 2") == ("si", 1)
    assert command("no") == ("no", 0) and command("salir")[0] == "salir"
    assert command("¿sí hay casos graves de sertralina?")[0] == "pregunta"
    assert re.match(r"pregunta", command("no sé qué pedir, ¿qué hay pendiente hoy?")[0])


# ---------------------------------------------------------------------------- benchmark v5


def test_the_v5_bench_and_its_checks(live, tmp_path):
    import inspect

    import agent.evaluate
    import coordinator.evaluate
    from coordinator.bench.v5 import build
    from coordinator.evaluate import check_one
    from coordinator.tools import STAGE_TOOLS, make_registry

    tasks = build()
    assert len(tasks) == 24 and {t["conjunto"] for t in tasks} == {"dev", "test"}
    known = set(
        re.findall(
            r'kind == "(\w+)"',
            inspect.getsource(coordinator.evaluate.check_one)
            + inspect.getsource(agent.evaluate._check_one),
        )
    )
    used = {c["type"] for t in tasks for c in t["comprobar"]}
    used |= {
        d["type"] for t in tasks for c in t["comprobar"] if c["type"] == "alguno" for d in c["de"]
    }
    assert used <= known, used - known

    monitor.crawl_and_save(_LiveLike(POSTS), ["sertralina"], since="2025-01-01")
    monitor.triage_pending(["sertralina"], llm_factory=lambda: _FakeTriage(), progress=print)
    store = _store(tmp_path)
    assert check_one({"type": "n_menciones", "filtro": {}}, "son 3 menciones", {}, store)
    assert check_one({"type": "n_descartes", "motivo": "sin_reaccion"}, "solo 1", {}, store)
    assert not check_one({"type": "n_descartes", "motivo": None}, "5 descartes", {}, store)
    assert check_one({"type": "sin_pendientes"}, "", {"pending_actions": []}, store)
    assert not set(STAGE_TOOLS) & set(make_registry(store, "sin_etapas").tools)
    assert set(STAGE_TOOLS) <= set(make_registry(store).tools)


def test_the_chat_can_start_with_no_data_at_all(live, tmp_path):
    f = monitor.files()
    store = CaseStore.from_files(
        [
            (tmp_path / "no.jsonl", tmp_path / "no2.jsonl", "muestra"),
            (f["menciones"], f["triaje"], "vigilante"),
        ],
        meddra_file=None,
        allow_empty=True,
    )
    assert store.mentions == {} and store.cases == {}
    monitor.crawl_and_save(_LiveLike(POSTS), ["sertralina"], since="2025-01-01")
    store.reload()  # the first crawl shows up
    assert len(store.mentions) == 3
    with pytest.raises(FileNotFoundError):
        CaseStore.from_files([(tmp_path / "no.jsonl", tmp_path / "no2.jsonl")], meddra_file=None)


def test_a_closed_period_and_reading_a_whole_mention(live, tmp_path):
    from datetime import date

    from coordinator.tools import make_registry

    now = datetime(2025, 6, 1, tzinfo=UTC)
    got = monitor.crawl_and_save(
        _LiveLike(POSTS), ["sertralina"], since="2025-01-01", until="2025-04-01", now=now
    )
    assert got["nuevas"] == 1  # only a1 (March); b1 (1 April) and c1 (May) are outside
    state = monitor.load_state(monitor.files()["estado"])
    assert "sertralina" not in state["ultima_ejecucion"]  # a past window: the cursor stays
    # "only September" is 30 days: small, it runs without asking
    day, end, error = etapas.window("2026-09-01", "2026-09-30")
    assert (day, end, error) == (date(2026, 9, 1), date(2026, 10, 1), None)
    assert etapas.window("2026-09-30", "2026-09-01")[2] is not None
    store = _store(tmp_path)
    reg = make_registry(store)
    mid = next(iter(store.mentions))
    whole = reg.call("ver_mencion", {"mencion_id": mid}).result
    assert whole["texto"].startswith("Llevo un mes") and whole["triaje"] == "sin_triar"
    assert not reg.call("ver_mencion", {"mencion_id": "M-nope"}).ok


def test_the_icsr_still_runs_at_once_and_jobs_can_be_foreground(live, tmp_path):
    store = _store(tmp_path)
    llm = _ScriptedLLM(
        [
            _call("rastrear", farmaco="sertralina", desde="2025-01-01", con_agente=False),
            _call("final_answer", answer="¿Lo lanzo?"),
            _call("final_answer", answer="Hecho."),
        ]
    )
    conv = Conversacion(store, brain="fake", brains={"fake": Brain(llm=llm)}, coder=str)
    conv.preguntar("Busca la sertralina desde 2025")
    done = conv.confirmar(0, segundo_plano=False)  # in the foreground: the answer comes back
    assert done.final_answer == "Hecho." and conv.trabajos == [] and len(store.mentions) == 3


def test_the_web_page_sends_confirms_and_reports_background_jobs(live, tmp_path):
    from coordinator.web import Panel, build_app

    store = _store(tmp_path)
    llm = _ScriptedLLM(
        [
            _call("rastrear", farmaco="sertralina", desde="2025-01-01", con_agente=False),
            _call("final_answer", answer="Es grande, ¿lo lanzo?"),
            _call("final_answer", answer="Ya está: 3 menciones."),
        ]
    )
    panel = Panel(Conversacion(store, brain="fake", brains={"fake": Brain(llm=llm)}, coder=str))
    panel.send("Busca la sertralina desde 2025")
    assert panel.history[-1]["content"] == "Es grande, ¿lo lanzo?"
    assert "Rastrear sertralina" in panel.pending_text() and "rastrear" in panel.trace
    panel.confirm(panel.pending_choices()[0])
    assert "segundo plano" in panel.history[-1]["content"]
    panel.conv.trabajos[0].hilo.join(timeout=30)
    assert panel.check_jobs() and panel.history[-1]["content"] == "Ya está: 3 menciones."
    assert "3 menciones" in panel.store_text() and not panel.check_jobs()
    assert build_app(panel) is not None  # Gradio builds the page


def test_a_notification_goes_through_the_triage_and_becomes_a_case(live, tmp_path, monkeypatch):
    from common import rutas
    from coordinator.tools import make_registry

    monkeypatch.setattr(rutas, "NOTIFICACIONES_MENCIONES", tmp_path / "n" / "menciones.jsonl")
    monkeypatch.setattr(rutas, "NOTIFICACIONES_TRIAJE", tmp_path / "n" / "triaje.jsonl")
    store = _store(tmp_path)
    reg = make_registry(store)
    text = "Me escribe una paciente: con Besitran tiene muchísimo sueño. Su correo es a@b.es"
    got = reg.call(
        "registrar_notificacion", {"texto": text, "farmaco": "Besitran", "origen": "correo"}
    ).result
    assert got["farmaco"] == "sertralina" and got["triaje"]["es_efecto_adverso"] == "si"
    case = store.cases[got["id"]]  # the same kind of case as a post
    assert case.fuente == "correo" and case.summary()["origen"] == "notificación (correo)"
    seen = reg.call("ver_mencion", {"mencion_id": got["id"]}).result
    assert "[correo]" in seen["texto"] and "registrada por el técnico" in seen["nota"]
    assert (
        "error"
        in reg.call(
            "registrar_notificacion", {"texto": "algo largo de verdad", "farmaco": "xelofrina"}
        ).result
    )


def test_in_an_exam_a_notification_is_triaged_but_not_saved(live, tmp_path, monkeypatch):
    """Review (2026-11-26): ``hold_stages`` held rastrear and triar, but registrar_notificacion
    wrote to data/ and made a case that the next tasks of the same exam would see."""
    from common import rutas

    saved = tmp_path / "n" / "menciones.jsonl"
    monkeypatch.setattr(rutas, "NOTIFICACIONES_MENCIONES", saved)
    monkeypatch.setattr(rutas, "NOTIFICACIONES_TRIAJE", tmp_path / "n" / "triaje.jsonl")
    store = _store(tmp_path)
    before = dict(store.cases)
    text = "Me escribe una paciente: con Besitran tiene muchísimo sueño desde hace días."
    llm = _ScriptedLLM(
        [
            _call("registrar_notificacion", texto=text, farmaco="Besitran", origen="correo"),
            _call("final_answer", answer="El triaje dice que es un efecto adverso."),
        ]
    )
    out = coordinar(
        "Registra esta notificación: " + text,
        store,
        brain="fake",
        brains={"fake": Brain(llm=llm)},
        coder=str,
        hold_stages=True,
    )
    observation = out.steps[0].observation
    assert '"es_efecto_adverso": "si"' in observation and "modo examen" in observation
    assert store.cases == before and not saved.exists()


def test_the_frozen_sample_ignores_the_monitors_state(live, tmp_path):
    """Review (2026-11-26): the benchmarks read only the frozen sample, so a crawl made on that
    machine (the monitor's state) must not appear in estado_del_sistema."""
    from coordinator.tools import make_registry

    monitor.crawl_and_save(_LiveLike(POSTS), ["sertralina"], since="2025-01-01")
    f = monitor.files()
    state = {
        "ultima_ejecucion": {"sertralina": "2026-11-26T10:00:00Z"},
        "hilos": {},
        "terminos": {},
    }
    f["estado"].write_text(json.dumps(state), "utf-8")
    seen_live = CaseStore.from_files([(f["menciones"], f["triaje"], "vigilante")])
    frozen = CaseStore.from_files([(f["menciones"], f["triaje"], "muestra")])  # as a sample
    for store, last in ((seen_live, "2026-11-26"), (frozen, "nunca")):
        got = make_registry(store).call("estado_del_sistema", {"farmaco": "sertralina"}).result
        assert got["farmacos"]["sertralina"]["ultimo_rastreo_del_vigilante"] == last
