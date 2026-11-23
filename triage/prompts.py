"""The triage prompts, one per version. Each version is a template with ``{farmaco}`` and
``{menciones}`` (the mentions of one conversation, rendered by ``render_mentions``).

* ``v0``: the prompt the 27B used to pre-annotate the answer sheet (``evaluacion_triaje``),
  as is. It is the starting point, but beware: the answer sheet was built by reviewing the
  disagreements between Claude and the 27B *with this prompt*, so it is somewhat favoured.
* ``v1``: v0 plus the rules for the traps the annotation found (jokes with a real symptom,
  other drugs, symptoms of an earlier message, withdrawal and lack of effect) and invented
  examples. Nothing in it is copied from the real posts.

To add a version: write the template here, register it in ``PROMPTS`` and run it on the
development set (``python -m triage.run --version vN``). Write down in EXPERIMENTS.md what
changed and what happened.
"""

from __future__ import annotations

from crawler.store import RawMention
from evaluacion_triaje.lote import PRELABEL_PROMPT, conversation_text


def render_mentions(mentions: list[RawMention]) -> str:
    """The mentions of one conversation as the prompt shows them: each with its id, the chain
    of earlier messages it answers and the post itself."""
    blocks = []
    for m in mentions:
        chain = conversation_text(m) or "(es el primer post de la conversación)"
        blocks.append(f"### id: {m.id}\nConversación anterior:\n{chain}\nPost: {m.texto}")
    return "\n\n".join(blocks)


PROMPT_V1 = """Eres técnico de farmacovigilancia y haces el triaje de posts de Bluesky. El
rastreador te pasa posts que podrían hablar de {farmaco}, cada uno con los mensajes anteriores
de su conversación. Para cada post decides si describe un efecto adverso de {farmaco} y, si lo
describe, extraes los datos del caso. Se anota SOLO lo que dice el post (leído con su
conversación); no supongas nada.

## Paso 1. ¿Habla el post de {farmaco}?

El rastreador se equivoca a menudo, sobre todo con respuestas dentro de un hilo. Es
"no_es_mencion" (y es_efecto_adverso "no") si el post:
- es una respuesta de cortesía o apoyo sin contenido sobre el fármaco: "gracias", "ánimo",
  "un abrazo", emojis, "mucha fuerza";
- habla de OTRO fármaco (duloxetina, escitalopram, lorazepam…) o de "un antidepresivo" / "las
  pastillas" sin que se sepa que es {farmaco}: náuseas con la duloxetina NO son un efecto de
  {farmaco};
- usa el nombre para otra cosa (un personaje, una canción, un juego de palabras sin más).

Un post que NO nombra el fármaco sí habla de él si la conversación deja claro que se refiere a
{farmaco} ("a mí también me pasó", "yo la dejé por eso", respondiendo a alguien que habla de
{farmaco}).

## Paso 2. ¿Es un efecto adverso?

"si" cuando el autor cuenta que A ÉL o a ALGUIEN CONCRETO ("mi madre", "mi pareja") le pasó
algo malo, o algo distinto de lo esperado, con {farmaco}:
- síntomas o molestias atribuidos al fármaco, aunque dude de la causa ("no sé si es la
  sertralina o el estrés");
- síntomas al dejarlo o al bajar la dosis (retirada): "al dejarla me mareaba";
- falta de eficacia: "llevo dos meses y no me hace nada";
- AUNQUE lo cuente en tono de broma: "primer día con {farmaco} y solo quiero vomitar 😂" es un
  efecto adverso. El humor no anula un síntoma real.

"no" en todo lo demás, con su motivo_no:
- "pregunta_o_consejo": preguntas ("¿engorda?", "¿a alguien le dio insomnio?"), consejos
  ("habla con tu médico"), contar que se la recetaron sin más;
- "broma_opinion_noticia": bromas sin un síntoma propio, opiniones sobre los psicofármacos,
  noticias, estudios, memes;
- "sin_reaccion": la toma y no cuenta nada malo ("me va genial", "llevo dos años y bien"), o
  solo dice que empieza o que la deja;
- "no_es_mencion": ver paso 1.

Cuidado con la conversación: los síntomas de un mensaje anterior son de quien los escribió, NO
del autor de este post. Una respuesta como "¿y te dio sueño?" es una pregunta; "a mí también"
o "me pasó lo mismo" sí cuenta, y entonces la reacción se deduce de la conversación.

## Paso 3. Datos del caso (solo si es_efecto_adverso = "si")

- reacciones: cada reacción con las palabras del autor, separadas por " | ". Si solo se sabe
  por la conversación ("a mí también me pasó"), escribe la reacción deducida entre corchetes:
  "[náuseas]".
- negadas: lo que el autor dice que NO tuvo ("náuseas no he tenido" → "náuseas"); si nada, "".
- paciente: "autor" (le pasa a quien escribe), "familiar_o_conocido" o "desconocido".
- edad: el número SOLO si se dice ("tengo 34 años"); si no, "desconocido". No la deduzcas.
- sexo: "mujer", "hombre" o "desconocido". sexo_fuente: "dicho" ("soy mujer", "mi madre"),
  "deducido" (género gramatical del paciente: "estoy cansada" → mujer) o "" si desconocido.
- grave: "si" SOLO si el post dice muerte, riesgo vital, ingreso en el hospital, discapacidad u
  otro evento médicamente importante (intento o ideas de suicidio, convulsiones, manía); si no,
  "no_consta" (lo normal). Ir a urgencias SIN ingreso NO es hospitalización.
- criterio_gravedad: "muerte", "riesgo_vital", "hospitalizacion", "discapacidad",
  "otro_importante" o "" (vacío si grave = "no_consta").
- desenlace: "recuperado" ("se me quitó"), "recuperandose" ("ya voy mejor"), "no_recuperado"
  ("sigo igual") o "desconocido" (lo normal).

Si es_efecto_adverso = "no", deja vacíos todos los campos del paso 3.

## Ejemplos (inventados)

- "Gracias por contarlo 🫂" (respondiendo a alguien con náuseas por {farmaco}) →
  {{"es_efecto_adverso": "no", "motivo_no": "no_es_mencion"}}
- "Primera semana con {farmaco} y estoy agotada todo el día jajaja" →
  {{"es_efecto_adverso": "si", "reacciones": "estoy agotada todo el día", "paciente": "autor",
  "edad": "desconocido", "sexo": "mujer", "sexo_fuente": "deducido", "grave": "no_consta",
  "criterio_gravedad": "", "desenlace": "desconocido"}}
- "A mí también me pasó al principio, luego se fue" (respondiendo a "con {farmaco} tengo
  diarrea") → {{"es_efecto_adverso": "si", "reacciones": "[diarrea]", "paciente": "autor",
  "desenlace": "recuperado", …}}
- "Con la duloxetina me mareaba muchísimo" (en un hilo sobre {farmaco}) →
  {{"es_efecto_adverso": "no", "motivo_no": "no_es_mencion"}}
- "¿La {farmaco} quita el apetito?" → {{"es_efecto_adverso": "no", "motivo_no":
  "pregunta_o_consejo"}}
- "Llevo un año con {farmaco} y me ha cambiado la vida" → {{"es_efecto_adverso": "no",
  "motivo_no": "sin_reaccion"}}

## Posts

{menciones}

Responde SOLO con una lista JSON, un objeto por post, con su "id" y los campos de arriba:
[{{"id": "...", "es_efecto_adverso": "si|no", "motivo_no": "...", "reacciones": "...",
  "negadas": "", "paciente": "...", "edad": "...", "sexo": "...", "sexo_fuente": "...",
  "grave": "...", "criterio_gravedad": "", "desenlace": "..."}}]"""


PROMPTS = {"v0": PRELABEL_PROMPT, "v1": PROMPT_V1}


REPAIR = """Algunas respuestas no se pudieron usar:
{problemas}

Responde de nuevo SOLO para estos ids ({ids}), con una lista JSON en el mismo formato y los
valores exactos de las instrucciones."""


def build_prompt(version: str, mentions: list[RawMention]) -> str:
    if version not in PROMPTS:
        raise ValueError(f"unknown prompt version {version!r}; use one of {sorted(PROMPTS)}")
    return PROMPTS[version].format(farmaco=mentions[0].farmaco, menciones=render_mentions(mentions))
