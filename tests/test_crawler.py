"""The crawler without network or GPU: a toy frozen sample, scripted brains, fake APIs."""

import json

import httpx
import pytest

from agent.react_agent import Brain
from common.llm import RemoteChatModel
from crawler import annotate
from crawler.agent import CrawlerAgent
from crawler.cima import brand_of
from crawler.evaluate import score
from crawler.monitor import run_once
from crawler.pubmed import parse_articles
from crawler.sources import FrozenSource, LiveSource, Post, to_iso
from crawler.store import MentionStore
from crawler.systems import run_system
from crawler.text import anonymize, fold, matches, post_id
from crawler.tools import CrawlerContext, make_registry

BRANDS = {"sertralina": ["Besitran", "Aremis"]}


def _p(pid, text, date, thread=None, parent=None, replies=0):
    return Post(pid, text, f"{date}T10:00:00Z", thread or pid, parent, replies, "es")


def toy_posts():
    return [
        _p("a1", "Llevo un mes con sertralina y me paso el día dormida", "2025-03-01", replies=2),
        _p("a2", "A mí también me dejó KO las primeras semanas", "2025-03-02", "a1", "a1"),
        _p("a3", "Qué bonito día hace hoy", "2025-03-03", "a1", "a1"),
        _p("b1", "El Besitran me sienta fatal", "2025-04-01"),
        _p("c1", "Me han recetado antidepresivos y tengo miedo", "2025-05-01", replies=1),
        _p("c2", "Las pastillas del psiquiatra me dan sueño", "2025-05-02", "c1", "c1"),
        _p("d1", "Hablando de sertralina otra vez", "2024-12-01"),
        _p("e1", "Tómate un ibuprofeno y a dormir", "2025-06-01"),
    ]


GOLD = {
    "a1": {"farmaco": "sertralina", "tipo_mencion": "nombre"},
    "a2": {"farmaco": "sertralina", "tipo_mencion": "indirecta"},
    "a3": {"farmaco": None, "tipo_mencion": None},
    "b1": {"farmaco": "sertralina", "tipo_mencion": "marca"},
    "c1": {"farmaco": None, "tipo_mencion": None},
    "c2": {"farmaco": None, "tipo_mencion": None},
}


@pytest.fixture
def source():
    return FrozenSource(toy_posts())


# ---------------------------------------------------------------------------- text


def test_text_helpers():
    assert fold("Sertralína ÁCIDO") == "sertralina acido"
    assert matches("La SERTRALÍNA me da sueño", "sertralina sueño")
    assert not matches("sertralinas", "sertralina")  # whole words, like Bluesky
    clean = anonymize("Mira @ana.bsky.social https://x.com/a y escribe a ana@mail.com")
    assert clean == "Mira @usuario [enlace] y escribe a [correo]"
    assert post_id("at://did:plc:abc/app.bsky.feed.post/1").startswith("p")
    assert "did" not in post_id("at://did:plc:abc/app.bsky.feed.post/1")


def test_brand_of_skips_generics_and_combinations():
    assert brand_of("BESITRAN 50 mg comprimidos", "sertralina") == "Besitran"
    assert brand_of("SERTRALINA CINFA 50 mg", "sertralina") is None
    assert (
        brand_of("DAPAGLIFLOZINA/METFORMINA X", "metformina", "dapagliflozina + metformina") is None
    )
    assert brand_of("CAPTOR 50 mg", "tramadol") is None  # an ordinary word


# ---------------------------------------------------------------------------- sources


def test_frozen_search_period_order_and_threads(source):
    hits = source.search("sertralina", since="2025-01-01")
    assert [p.id for p in hits] == ["a1"]  # d1 is before the period
    assert [p.id for p in source.search("sertralina")] == ["a1", "d1"]  # newest first
    assert [p.id for p in source.thread("a3")] == ["a1", "a2", "a3"]
    assert to_iso("2025-01-01") == "2025-01-01T00:00:00Z"


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_live_source_normalises_api_objects_and_hides_uris():
    view = {
        "uri": "at://did:plc:xyz/app.bsky.feed.post/2",
        "replyCount": 0,
        "record": {
            "text": "Hola @pepe.bsky.social, la sertralina me va bien",
            "createdAt": "2025-02-01T10:00:00.000Z",
            "langs": ["es"],
            "reply": {
                "root": {"uri": "at://did:plc:aaa/app.bsky.feed.post/1"},
                "parent": {"uri": "at://did:plc:aaa/app.bsky.feed.post/1"},
            },
        },
    }
    root = {
        **view,
        "uri": "at://did:plc:aaa/app.bsky.feed.post/1",
        "record": {"text": "inicio", "createdAt": "2025-01-31T10:00:00Z"},
    }
    feed = _NS(
        search_posts=lambda params: {"posts": [view], "cursor": None},
        get_post_thread=lambda params: {"thread": {"post": root, "replies": [{"post": view}]}},
    )
    live = LiveSource(client=_NS(app=_NS(bsky=_NS(feed=feed))), min_interval=0)
    [post] = live.search_all("sertralina")
    assert post.text == "Hola @usuario, la sertralina me va bien"
    assert post.parent_id == post.thread_id == post_id("at://did:plc:aaa/app.bsky.feed.post/1")
    assert "did:" not in json.dumps(post.to_json())
    thread = live.thread(post.id)
    assert [p.text for p in thread] == ["inicio", post.text]


# ---------------------------------------------------------------------------- tools


def _ctx(source, **kw):
    return CrawlerContext(
        source=source,
        store=MentionStore(),
        brands=BRANDS,
        drugs=("sertralina",),
        since="2025-01-01",
        **kw,
    )


def test_save_only_posts_the_model_has_seen(source):
    ctx = _ctx(source)
    reg = make_registry(ctx, "vigilancia")
    args = {"post_id": "a2", "farmaco": "sertralina", "tipo_mencion": "indirecta", "motivo": "x"}
    assert not reg.call("guardar_mencion", args).ok  # never shown
    assert reg.call("leer_hilo", {"post_id": "a1"}).ok
    first = reg.call("guardar_mencion", args)
    assert first.ok and "saved" in first.result
    assert "already" in reg.call("guardar_mencion", args).result
    mention = next(iter(ctx.store.mentions.values()))
    assert mention.texto == "A mí también me dejó KO las primeras semanas"  # copied, not written
    assert mention.contexto.responde_a is None and mention.contexto.inicio_hilo.startswith("Llevo")
    by_brand = reg.call("guardar_mencion", {**args, "farmaco": "Besitran"})
    assert by_brand.ok  # a brand is resolved to its active ingredient (already saved)


def test_budgets_are_enforced_by_the_tools(source):
    ctx = _ctx(source, max_searches=1, max_threads=1)
    reg = make_registry(ctx, "vigilancia")
    assert reg.call("buscar_bluesky", {"consulta": "sertralina"}).ok
    assert "budget" in reg.call("buscar_bluesky", {"consulta": "besitran"}).error
    assert reg.call("leer_hilo", {"post_id": "a1"}).ok
    assert "budget" in reg.call("leer_hilo", {"post_id": "c1"}).error
    assert reg.call("leer_hilo", {"post_id": "a2"}).ok  # same thread again: free


def test_hilos_mode_has_no_search(source):
    names = set(make_registry(_ctx(source), "hilos").tools)
    assert "buscar_bluesky" not in names and {"leer_hilo", "guardar_mencion"} <= names
    assert "buscar_pubmed" in make_registry(_ctx(source), "dirigido").tools


# ---------------------------------------------------------------------------- systems + score


def test_fixed_search_and_threads_baselines(source):
    fija = run_system("fija", source, "sertralina", BRANDS, "2025-01-01")
    assert fija.store.post_ids() == {"a1", "b1"}
    hilos = run_system("fija_hilos", source, "sertralina", BRANDS, "2025-01-01")
    assert hilos.store.post_ids() == {"a1", "a2", "a3", "b1"}
    s_fija, s_hilos = score([fija], GOLD), score([hilos], GOLD)
    assert s_fija["recall"] == pytest.approx(2 / 3) and s_fija["precision"] == 1.0
    assert s_fija["recall_por_tipo"]["indirecta"]["recall"] == 0.0
    assert s_hilos["recall"] == 1.0 and s_hilos["precision"] == pytest.approx(3 / 4)


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


def test_agent_loop_reads_a_thread_and_saves_the_indirect_reply(source):
    ctx = _ctx(source)
    llm = _ScriptedLLM(
        [
            _call("buscar_bluesky", consulta="sertralina"),
            _call("leer_hilo", post_id="a1"),
            _call(
                "guardar_mencion",
                post_id="a2",
                farmaco="sertralina",
                tipo_mencion="indirecta",
                motivo="responde a un post sobre sertralina",
            ),
            _call(
                "guardar_mencion",
                post_id="zzz",
                farmaco="sertralina",
                tipo_mencion="nombre",
                motivo="invented id",
            ),
            _call("final_answer", answer="1 mención"),
        ]
    )
    agent = CrawlerAgent(
        registry=make_registry(ctx, "vigilancia"),
        mode="vigilancia",
        brains={"fake": Brain(llm=llm)},
    )
    response = agent.run("Busca sertralina", max_steps=10, brain="fake")
    assert response.stop_reason == "final_answer"
    assert ctx.store.post_ids() == {"a2"}  # the invented id was rejected
    assert "unknown post id" in response.steps[3].observation
    assert response.steps[1].observation.startswith("thread a1")  # plain text, not JSON


# ---------------------------------------------------------------------------- annotation


def test_kappa_and_consolidation(tmp_path, source):
    assert annotate.kappa(["si", "no", "si", "no"], ["si", "no", "si", "no"]) == 1.0
    assert annotate.kappa(["si", "si", "no", "no"], ["si", "no", "si", "no"]) == 0.0
    posts = [source.get(i) for i in ("a1", "a2", "a3")]
    labels = {
        "a1": {"farmaco": "sertralina", "tipo_mencion": "nombre", "es_experiencia": "sí"},
        "a2": {"farmaco": "Sertralina", "tipo_mencion": "indirecta", "es_experiencia": "si"},
        "a3": {"farmaco": "ninguno"},
    }
    path = tmp_path / "lote.csv"
    annotate.write_csv(posts, source, path, labels)
    rows = annotate.read_csv(path)
    assert rows[1]["texto_responde_a"].startswith("Llevo un mes")
    gold, errors = annotate.consolidate(rows)
    assert not errors and gold[1] == {**gold[1], "farmaco": "sertralina", "es_experiencia": True}
    assert gold[2]["farmaco"] is None
    other = [dict(r, tipo_mencion="nombre") if r["post_id"] == "a2" else r for r in rows]
    result = annotate.agreement(rows, other)
    assert result["posts_comunes"] == 3 and len(result["desacuerdos"]) == 1
    assert annotate.validate({"farmaco": "sertralina", "tipo_mencion": "x"}) is not None


# ---------------------------------------------------------------------------- monitor


class _LiveLike(FrozenSource):
    """A frozen source with the extra methods of LiveSource used by the monitor."""

    def uri(self, pid):
        return f"at://{pid}"

    def remember(self, pid, uri):
        pass


def test_monitor_follows_threads_and_finds_late_replies(monkeypatch):
    monkeypatch.setattr("crawler.monitor.load_brands", lambda: BRANDS)
    posts = toy_posts()[:3]
    state = {"ultima_ejecucion": {}, "hilos": {}, "terminos": {}}
    now = "2025-03-04T00:00:00+00:00"
    from datetime import datetime

    store, summary = run_once(
        _LiveLike(posts),
        state,
        ["sertralina"],
        first_lookback_hours=24 * 10,
        now=datetime.fromisoformat(now),
    )
    assert store.post_ids() == {"a1"} and "a1" in state["hilos"]
    late = _p("a4", "Yo la dejé por el sueño", "2025-03-05", "a1", "a1")
    store2, summary2 = run_once(
        _LiveLike([*posts, late]),
        state,
        ["sertralina"],
        now=datetime.fromisoformat("2025-03-06T00:00:00+00:00"),
    )
    assert summary2["sertralina"]["hilos_con_novedades"] == 1
    assert "a4" in state["hilos"]["a1"]["vistos"]


# ---------------------------------------------------------------------------- remote LLM, PubMed


def test_remote_chat_model_payload_reasoning_and_retry(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "<tool_call>{}</tool_call>",
                            "reasoning_content": "pienso",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 7},
            },
        )

    model = RemoteChatModel(
        model="m", base_url="http://x/v1", transport=httpx.MockTransport(handler)
    )
    text, generated = model.generate(
        [{"role": "system", "content": "s"}, {"role": "tool", "content": "obs", "name": "t"}],
        enable_thinking=True,
    )
    assert len(calls) == 2  # one retry after the 503
    sent = calls[-1]
    assert sent["messages"][1] == {
        "role": "user",
        "content": "<tool_response>\nobs\n</tool_response>",
    }
    assert sent["temperature"] == 0.6 and sent["chat_template_kwargs"] == {"enable_thinking": True}
    assert text.startswith("<think>\npienso\n</think>") and generated == 7
    assert model.last_prompt_tokens == 120


def test_parse_pubmed_xml():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
    <Journal><Title>J Test</Title><JournalIssue><PubDate><Year>2024</Year></PubDate>
    </JournalIssue></Journal><ArticleTitle>SSRIs and falls</ArticleTitle><Abstract>
    <AbstractText>Older adults fell.</AbstractText></Abstract></Article></MedlineCitation>
    </PubmedArticle></PubmedArticleSet>"""
    [a] = parse_articles(xml)
    assert a == {
        "pmid": "123",
        "titulo": "SSRIs and falls",
        "revista": "J Test",
        "año": "2024",
        "resumen": "Older adults fell.",
    }


def test_errors_list_missed_and_extra_posts(source):
    from crawler.evaluate import errors

    result = run_system("fija_hilos", source, "sertralina", BRANDS, "2025-01-01")
    out = errors([result], GOLD, source)
    assert [e["post_id"] for e in out["sobrantes"]] == ["a3"]  # a3 is in the thread, not about it
    assert out["perdidos"] == []
    fija = errors([run_system("fija", source, "sertralina", BRANDS, "2025-01-01")], GOLD, source)
    assert fija["perdidos"][0]["post_id"] == "a2" and fija["perdidos"][0]["responde_a"]
