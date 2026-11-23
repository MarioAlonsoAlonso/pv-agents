"""Coordinator: case store, tools, the loop with a scripted brain and the benchmark checks,
without network, GPU or data/."""

import json

from agent.react_agent import Brain
from coordinator.agent import CoordinatorAgent, coordinar
from coordinator.cases import CaseStore
from coordinator.evaluate import (
    check_one,
    fill,
    placeholders,
    score_task,
    unsupported_citations,
    unsupported_numbers,
)
from coordinator.tools import make_registry, parse_terms
from crawler.store import ConversationPost, RawMention

YES = {
    "es_efecto_adverso": "si",
    "reacciones": "me da muchísimo sueño",
    "paciente": "autor",
    "edad": "desconocido",
    "sexo": "desconocido",
    "grave": "no_consta",
    "criterio_gravedad": "",
    "desenlace": "desconocido",
}


def _m(i, text, date="2026-01-01T00:00:00Z", origin="busqueda_fija", chain=()):
    return RawMention(
        id=f"M-{i:010x}",
        post_id=f"p{i}",
        hilo_id="h",
        farmaco="sertralina",
        tipo_mencion="nombre" if origin == "busqueda_fija" else "indirecta",
        texto=text,
        conversacion=[ConversationPost(post_id=f"c{k}", texto=t) for k, t in enumerate(chain)],
        fecha=date,
        encontrado_por=origin,
    )


def _store(tmp_path=None):
    ms = [
        _m(1, "me da sueño", "2026-01-01T00:00:00Z"),
        _m(2, "me ingresaron", "2025-06-01T00:00:00Z"),
        _m(3, "a mí también", "2026-02-01T00:00:00Z", "agente", ["con sertralina tengo náuseas"]),
        _m(4, "me va genial"),
        _m(5, "mareada", "2025-01-01T00:00:00Z"),
    ]
    answers = {
        ms[0].id: YES,
        ms[1].id: {**YES, "grave": "si", "criterio_gravedad": "hospitalizacion"},
        ms[2].id: {**YES, "reacciones": "[náuseas]"},
        ms[3].id: {"es_efecto_adverso": "no", "motivo_no": "sin_reaccion"},
        ms[4].id: {**YES, "sexo": "mujer", "sexo_fuente": "deducido"},
    }
    meddra = tmp_path / "meddra.json" if tmp_path else None
    return ms, CaseStore.build([(ms, answers)], meddra)


def test_store_keeps_adverse_events_by_priority(tmp_path):
    ms, store = _store(tmp_path)
    assert store.reviewed == {"sertralina": 5} and len(store.cases) == 4
    order = [c.id for c in store.of_drug("sertralina")]
    # serious first; then the most complete (sex known); then the newest
    assert order == [ms[1].id, ms[4].id, ms[2].id, ms[0].id]
    assert [c.id for c in store.of_drug("sertralina", serious_only=True)] == [ms[1].id]
    assert "náuseas" in store.get(ms[2].id).report_text()
    assert store.of_drug("lorazepam") == []


def test_tools_list_read_code_and_group(tmp_path):
    ms, store = _store(tmp_path)
    calls = []

    def coder(text):
        calls.append(text)
        return "Somnolence; Sueño raro"

    reg = make_registry(store, coder=coder)
    listed = reg.call("listar_casos", {"farmaco": "Sertralina", "limite": 2}).result
    assert listed["efectos_adversos"] == 4 and listed["graves"] == 1
    assert listed["menciones_triadas"] == 5 and len(listed["casos"]) == 2
    seen = reg.call("ver_caso", {"caso_id": f"[{ms[2].id}]"}).result
    assert "náuseas" in seen["conversacion_anterior"]
    coded = reg.call("codificar_meddra", {"caso_id": ms[0].id}).result
    assert coded["terminos_meddra"] == ["Somnolence", "Sueño raro"]
    assert coded["fuera_del_subconjunto"] == ["Sueño raro"] and "licencia" in coded["nota"]
    reg.call("codificar_meddra", {"caso_id": ms[0].id})
    assert len(calls) == 1  # coded once, then cached (and saved)
    again = reg.call("codificar_meddra", {"caso_id": ms[0].id}).result
    assert again["fuera_del_subconjunto"] == ["Sueño raro"]  # same answer from the cache
    saved = json.loads((tmp_path / "meddra.json").read_text("utf-8"))[ms[0].id]
    assert saved == ["Somnolence", "Sueño raro"]
    grouped = reg.call("agrupar_por_meddra", {"farmaco": "sertralina"}).result
    assert grouped["codificados"] == 1
    assert {g["termino"] for g in grouped["por_termino"]} == {"Somnolence", "Sueño raro"}
    assert not reg.call("ver_caso", {"caso_id": "M-zzz"}).ok


def test_variants_remove_components():
    _, store = _store()
    full = set(make_registry(store).tools)
    assert {"codificar_meddra", "search_knowledge_base", "buscar_literatura"} <= full
    assert "buscar_mas_casos" not in full  # parked: patients' reports are in the case store
    assert "naranjo_causality" not in full
    assert "search_knowledge_base" not in make_registry(store, "sin_rag").tools
    assert "codificar_meddra" not in make_registry(store, "sin_meddra").tools
    assert "buscar_literatura" not in make_registry(store, "sin_literatura").tools
    assert "buscar_literatura" not in make_registry(store, "sin_evidencia").tools  # old name


def test_parse_terms():
    assert parse_terms("Somnolence; Nausea.\nInsomnia") == (
        ["Somnolence", "Nausea", "Insomnia"],
        [],
    )


class _ScriptedLLM:
    name, last_prompt_tokens = "scripted", 50

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, messages, max_new_tokens=512, enable_thinking=False, **kw):
        self.prompts.append(messages)
        return self.replies.pop(0), 10

    def count_tokens(self, text):
        return len(text.split())


def _call(name, **arguments):
    return (
        f"Thought: ok\n<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"
    )


def test_loop_codes_and_holds_the_icsr_for_a_person(tmp_path):
    ms, store = _store(tmp_path)
    case = ms[4].id
    llm = _ScriptedLLM(
        [
            _call("listar_casos", farmaco="sertralina"),
            _call("codificar_meddra", caso_id=case),
            _call(
                "create_icsr_report",
                reporter_type="consumer",
                patient_age=None,
                patient_sex="female",
                suspect_drug="sertraline",
                reactions=["Dizziness"],
                serious=False,
                outcome="unknown",
                narrative="Mareo con sertralina contado en redes sociales.",
            ),
            _call("final_answer", answer=f"Borrador del ICSR de {case} pendiente de confirmar."),
        ]
    )
    response = coordinar(
        "Prepara el ICSR",
        store,
        brain="fake",
        coder=lambda text: "Dizziness",
        brains={"fake": Brain(llm=llm)},
    )
    assert response.stop_reason == "final_answer"
    assert [a.name for a in response.pending_actions] == ["create_icsr_report"]
    assert "Dizziness" in response.steps[1].observation
    system = llm.prompts[0][0]["content"]
    assert "coordinador" in system and "codificar_meddra" in system


def test_slow_tools_have_long_timeouts():
    _, store = _store()
    agent = CoordinatorAgent(registry=make_registry(store))
    assert agent.timeouts["codificar_meddra"] >= 300 and agent.timeouts["buscar_literatura"] >= 600


def test_bench_placeholders_and_checks(tmp_path):
    ms, store = _store(tmp_path)
    slots = placeholders(store)
    assert slots["caso_prioritario"] == ms[1].id and slots["caso_con_sexo"] == ms[4].id
    assert slots["caso_indirecto"] == ms[2].id
    task = fill(
        {"tarea": "Resume {caso_indirecto}", "comprobar": [{"value": "{caso_indirecto}"}]}, slots
    )
    assert task["tarea"] == f"Resume {ms[2].id}" and task["comprobar"][0]["value"] == ms[2].id
    store.save_meddra(ms[1].id, ["Somnolence"])
    store.save_meddra(ms[4].id, ["Somnolence", "Dizziness"])
    answer = (
        f"Hay 4 efectos adversos de 5 menciones, 1 grave. Revisa {ms[1].id} y {ms[4].id}. "
        "Somnolence."
    )
    assert check_one({"type": "n_efectos"}, answer, {}, store)
    assert check_one({"type": "n_triadas"}, answer, {}, store)
    assert check_one({"type": "n_graves"}, answer, {}, store)
    assert check_one({"type": "ids_prioritarios", "n": 3, "min": 2}, answer, {}, store)
    assert check_one({"type": "meddra_del_caso", "caso": ms[1].id}, answer, {}, store)
    assert check_one({"type": "pt_mas_frecuente", "n": 5}, answer, {}, store)
    assert check_one({"type": "cita_fragmento"}, "según [1d381329ee-0023]", {}, store)
    assert not check_one({"type": "cita_fragmento"}, "según la ficha", {}, store)
    assert check_one(
        {"type": "pending_action", "value": "create_icsr_report"},
        "",
        {"pending_actions": [{"name": "create_icsr_report"}]},
        store,
    )


def test_unsupported_numbers_catch_invented_figures():
    response = {"steps": [{"observation": '{"prr": 2.617, "a": 1680}'}]}
    answer = "PRR 2,62 con 1.680 notificaciones (caso M-0000000001); hay 4 300 en total en 2025"
    assert unsupported_numbers(answer, "tarea", response) == [4300.0]
    row = score_task(
        {
            "id": "x",
            "categoria": "c",
            "tarea": "t",
            "herramientas": ["listar_casos"],
            "comprobar": {"type": "regex", "value": "PRR"},
        },
        {
            "final_answer": answer,
            "steps": [{"action": {"name": "listar_casos"}, "observation": "1680 2.617"}],
            "n_steps": 1,
            "tokens_used": 5,
            "stop_reason": "final_answer",
        },
        _store()[1],
    )
    assert (
        row["correcta"]
        and row["herramientas_requeridas"]
        and row["numeros_sin_respaldo"] == [4300.0]
    )


def test_unsupported_citations():
    response = {"steps": [{"observation": "[ee3f3fd9f0-0073] (Ficha técnica …) insomnio"}]}
    answer = "Muy frecuente [ee3f3fd9f0-0073]; además [ee3f3fd9f0-0016]"
    assert unsupported_citations(answer, response) == ["ee3f3fd9f0-0016"]


def test_a_coding_that_does_not_finish_is_an_error_not_terms(tmp_path, monkeypatch):
    import agent.react_agent as ra
    from coordinator.tools import CodingFailed, rlm_coder

    class _Cut:
        calls = 0

        def answer(self, text):
            _Cut.calls += 1

            class R:
                raw = "<think>Okay, let's tackle this query. The user is asking"

            return R()

    monkeypatch.setattr(ra, "_RLM", _Cut())
    try:
        rlm_coder("me da sueño")
        raise AssertionError("expected CodingFailed")
    except CodingFailed:
        pass
    assert _Cut.calls == 2  # one retry
    ms, store = _store(tmp_path)
    record = make_registry(store).call("codificar_meddra", {"caso_id": ms[0].id})
    assert not record.ok and "did not finish" in record.error
    assert ms[0].id not in store.meddra  # nothing cached


def test_corrupt_codings_in_the_cache_are_dropped(tmp_path):
    ms = [_m(1, "me da sueño")]
    cache = tmp_path / "meddra.json"
    cache.write_text(
        json.dumps({ms[0].id: ["<think>", "Okay, let's tackle this query about the user"]}),
        "utf-8",
    )
    store = CaseStore.build([(ms, {ms[0].id: YES})], cache)
    assert store.meddra == {}


def test_open_request_checks(tmp_path):
    ms, store = _store(tmp_path)
    serious = ms[1].id
    response = {
        "steps": [
            {"action": {"name": "listar_casos"}},
            {"action": {"name": "search_knowledge_base"}},
        ]
    }
    assert check_one({"type": "ids_graves"}, f"Urgente: {serious}", response, store)
    assert not check_one({"type": "ids_graves"}, "nada urgente", response, store)
    either = {
        "type": "alguno",
        "de": [
            {"type": "pending_action", "value": "create_icsr_report"},
            {"type": "regex", "value": "identificab"},
        ],
    }
    assert check_one(either, "el paciente no es identificable", response, store)
    assert not check_one(either, "listo", response, store)
    tools = {"type": "herramientas_distintas", "de": ["search_knowledge_base", "codificar_meddra"]}
    assert check_one({**tools, "min": 1}, "", response, store)
    assert not check_one({**tools, "min": 2}, "", response, store)


def test_an_icsr_that_could_never_be_registered_is_not_queued(tmp_path):
    ms, store = _store(tmp_path)
    agent = CoordinatorAgent(registry=make_registry(store))
    args = {
        "reporter_type": "consumer",
        "patient_age": None,
        "patient_sex": "unknown",
        "suspect_drug": "sertraline",
        "reactions": ["Insomnia"],
        "serious": True,
        "outcome": "unknown",
        "narrative": "Caso grave sin edad ni sexo.",
    }
    pending = []
    obs = agent.observe("create_icsr_report", args, False, pending)
    assert obs.startswith("error") and "not identifiable" in obs and pending == []
    obs = agent.observe("create_icsr_report", {**args, "patient_sex": "male"}, False, pending)
    assert "human confirmation" in obs and len(pending) == 1
    obs = agent.observe("create_icsr_report", {**args, "reactions": ["Mania"]}, False, [])
    assert "not Preferred Terms" in obs


def test_without_the_reasoning_model_the_cached_codes_do_not_leak(tmp_path):
    ms, store = _store(tmp_path)
    store.save_meddra(ms[0].id, ["Somnolence"])
    seen = make_registry(store, "sin_meddra").call("ver_caso", {"caso_id": ms[0].id}).result
    assert "meddra" not in seen
    assert make_registry(store).call("ver_caso", {"caso_id": ms[0].id}).result["meddra"] == [
        "Somnolence"
    ]


def test_case_filter_and_v2_checks(tmp_path):
    ms, store = _store(tmp_path)
    assert len(store.filter("sertralina", sexo="mujer")) == 1
    assert len(store.filter("sertralina", grave=True)) == 1
    assert len(store.filter("sertralina", origen="agente")) == 1
    assert len(store.filter("sertralina", desde="2026-01-01")) == 2
    assert check_one({"type": "n_casos", "filtro": {"sexo": "mujer"}}, "Hay 1 mujer", {}, store)
    assert check_one(
        {"type": "n_casos", "filtro": {"paciente": "familiar_o_conocido"}},
        "No hay ninguno",
        {},
        store,
    )
    notif = {"type": "meddra_notif", "esperados": ["Cough"], "ausentes": ["Diarrhoea"]}
    assert check_one(notif, "Cough\nDiarrhoea no se codifica: se niega", {}, store)
    assert not check_one(notif, "Cough; Diarrhoea", {}, store)
    assert not check_one(notif, "Tos", {}, store)
    assert not check_one(
        {"type": "regex_ausente", "value": "registrado"}, "ya está registrado", {}, store
    )


def test_routing_of_a_request():
    from coordinator.evaluate import routing, routing_summary

    task = {"fuentes": ["rag", "faers"]}
    r = routing(task, ["search_knowledge_base", "listar_casos", "final_answer"])
    assert r["fuentes_faltan"] == ["faers"] and r["fuentes_de_mas"] == ["casos"]
    rows = [{**r, "id": "x"}, {**routing({"fuentes": []}, ["final_answer"]), "id": "y"}]
    summary = routing_summary(rows)["rutas"]
    assert summary["todas_las_fuentes_necesarias"] == 0.5
    assert summary["por_fuente"]["casos"]["usada_sin_necesitarla"] == 1


def test_the_v2_bench_is_well_formed():
    from coordinator.bench.v2 import build

    tasks = build()
    assert len(tasks) >= 100 and len({t["id"] for t in tasks}) == len(tasks)
    assert {t["conjunto"] for t in tasks} == {"dev", "test"}
    known = {"casos", "meddra", "rag", "faers", "literatura"}
    assert all(set(t["fuentes"]) <= known for t in tasks)
    death = [t for t in tasks if t.get("familia") == "death"]
    assert all("Death" in t["comprobar"][0]["ausentes"] for t in death)


def test_cases_can_be_searched_by_reaction(tmp_path):
    ms, store = _store(tmp_path)
    # cases 0, 1 and 4 say "me da muchísimo sueño"; case 2 has "[náuseas]"
    assert {c.id for c in store.filter("sertralina", buscar="SUEÑO")} == {
        ms[0].id,
        ms[1].id,
        ms[4].id,
    }
    assert [c.id for c in store.filter("sertralina", buscar="nauseas")] == [ms[2].id]
    assert len(store.filter("sertralina", buscar=["sueño", "náuseas"])) == 4
    check = {"type": "ids_filtro", "filtro": {"buscar": "náuseas"}, "min": 2}
    assert check_one(check, f"El caso {ms[2].id}", {}, store)  # min is capped at what exists
    assert check_one(
        {"type": "ids_filtro", "filtro": {"buscar": "brux"}}, "No hay ninguno", {}, store
    )


def test_listar_casos_searches_filters_and_counts(tmp_path):
    ms, store = _store(tmp_path)
    reg = make_registry(store)
    out = reg.call("listar_casos", {"farmaco": "sertralina", "buscar": "nauseas, vómito"}).result
    assert out["coinciden"] == 1 and out["casos"][0]["id"] == ms[2].id
    out = reg.call("listar_casos", {"farmaco": "sertralina", "sexo": "mujer"}).result
    assert out["coinciden"] == 1 and out["efectos_adversos"] == 4
    out = reg.call(
        "listar_casos", {"farmaco": "sertralina", "origen": "respuesta_en_conversacion"}
    ).result
    assert [c["id"] for c in out["casos"]] == [ms[2].id]
    out = reg.call("listar_casos", {"farmaco": "sertralina", "limite": 2}).result
    assert out["coinciden"] == 4 and out["mostrados"] == 2 and "2 casos más" in out["nota"]
    out = reg.call("listar_casos", {"farmaco": "sertralina", "buscar": "calor"}).result
    assert out["coinciden"] == 0 and "sinónimos" in out["nota"]
    out = reg.call("listar_casos", {"farmaco": "sertralina", "desde": "2026-01-01"}).result
    assert out["coinciden"] == 2


def test_listar_casos_without_drug_and_by_patient(tmp_path):
    ms, store = _store(tmp_path)
    reg = make_registry(store)
    out = reg.call("listar_casos", {}).result
    assert out["farmaco"] == "todos" and out["farmacos_con_casos"] == {"sertralina": 4}
    out = reg.call("listar_casos", {"farmaco": "omeprazol"}).result
    assert out["coinciden"] == 0 and out["farmacos_con_casos"] == {"sertralina": 4}
    out = reg.call("listar_casos", {"paciente": "familiar_o_conocido"}).result
    assert out["coinciden"] == 0


def test_codificar_notificacion_uses_the_reasoning_model(tmp_path):
    _, store = _store(tmp_path)
    seen = []
    reg = make_registry(store, coder=lambda text: seen.append(text) or "Hypoglycaemia")
    out = reg.call("codificar_notificacion", {"texto": " temblor e hipoglucemia "}).result
    assert out["terminos_meddra"] == ["Hypoglycaemia"] and seen == ["temblor e hipoglucemia"]
    assert "codificar_notificacion" not in make_registry(store, "sin_meddra").tools


def test_percentages_of_observed_counts_are_supported():
    response = {"steps": [{"observation": '{"menciones": 725, "casos": 159}'}]}
    assert unsupported_numbers("159 de 725, un 21,9 %", "t", response) == []
    assert unsupported_numbers("un 37,5 % del total", "t", response)  # not a ratio of the two


def test_meddra_check_reads_what_is_coded():
    _, store = _store()
    check = {"type": "meddra_notif", "esperados": ["Cough"], "ausentes": ["Diarrhoea"]}
    assert check_one(
        check, "- **Cough** (tos seca)\n- Diarrhoea: se niega, no se codifica", {}, store
    )
    assert not check_one(check, "- **Dry cough** (tos seca)", {}, store)  # not the PT
    assert not check_one(check, "- Cough\n- Diarrhoea", {}, store)  # forbidden term coded
    both = {"type": "meddra_notif", "esperados": ["Anaphylactic reaction", "Headache"]}
    answer = (
        "- **Anaphylactic reaction**\n"
        "- El dolor de cabeza no se ha codificado: podría añadirse Headache"
    )
    assert not check_one(both, answer, {}, store)
    assert check_one(both, "1. Anaphylactic reaction\n2. Headache", {}, store)
    lack = {"type": "meddra_notif", "esperados": ["Drug ineffective"]}
    assert check_one(lack, "- **Drug ineffective** (no le hizo efecto)", {}, store)
