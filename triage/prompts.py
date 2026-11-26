"""The triage prompts, one per version. Each version is a template with ``{farmaco}`` and
``{menciones}`` (the mentions of one conversation, rendered by ``render_mentions``).

* ``v0``: the prompt the 27B used to pre-annotate the answer sheet (``evaluacion_triaje``),
  as is. It is the starting point, but beware: the answer sheet was built by reviewing the
  disagreements between Claude and the 27B *with this prompt*, so it is somewhat favoured.
* ``v1``: v0 plus the rules for the traps the annotation found (jokes with a real symptom,
  other drugs, symptoms of an earlier message, withdrawal and lack of effect) and invented
  examples. Nothing in it is copied from the real posts.
* ``v2``: v1 was stricter than v0 (precision 75 → 86 %) but lost cases (detection 82 → 75 %).
  Its failures on the development set (2026-11-23): questions that also tell the author's own
  symptom ("¿es normal este mareo?") taken as questions; vague complaints
  attributed to the drug ("sus efectos secundarios", "me sienta fatal") taken as opinions;
  a post that blames the drug while also talking about another one taken as "otro fármaco";
  forgotten doses and stopping taken as "sin reacción". v2 keeps v1's rules and adds those
  cases, with the principle that in doubt the case goes to the human reviewer. It also stops
  deducing "hombre" when no word has gender. The examples are still invented.

To add a version: write the template here, register it in ``PROMPTS`` and run it on the
development set (``python -m triage.run --version vN``). Write down in EXPERIMENTS.md what
changed and what happened.
"""

from __future__ import annotations

from crawler.store import RawMention, conversation_text

# v0: the prompt of the pre-annotation of the answer sheet (it was defined in
# evaluacion_triaje/lote.py until 2026-11-26; the annotation tool imports it from here).
PRELABEL_PROMPT = """Eres anotador de farmacovigilancia. Para cada mención de abajo (un post de
Bluesky que habla de {farmaco}, con los mensajes anteriores de su conversación), decide lo
siguiente.
Anota SOLO lo que dice el post, leído con su conversación; no supongas nada.

- es_efecto_adverso: "si" si el autor cuenta que a él o a alguien concreto le pasó algo malo con
  el fármaco (síntomas, molestias, síntomas al dejarlo o bajar la dosis, o que no le hace efecto),
  aunque dude de la causa. "no" en otro caso.
- motivo_no (solo si "no"): "no_es_mencion" (no habla del fármaco: un "gracias", otro fármaco,
  otra cosa), "pregunta_o_consejo", "broma_opinion_noticia" o "sin_reaccion" (lo toma o le va
  bien, sin nada malo).
- reacciones (solo si "si"): cada reacción con las palabras del autor, separadas por " | ". Si
  solo se sabe por la conversación ("a mí también me pasó"), escríbela entre corchetes: [náuseas].
- negadas: lo que el autor dice que NO tuvo (vacío si nada).
- paciente: "autor", "familiar_o_conocido" o "desconocido".
- edad: el número si se dice; si no, "desconocido".
- sexo: "mujer", "hombre" o "desconocido"; sexo_fuente: "dicho", "deducido" (por el género
  gramatical: "estoy cansada") o "" si desconocido.
- grave: "si" solo si dice muerte, riesgo vital, hospitalización (ingreso), discapacidad u otro
  evento médicamente importante (p. ej. ideación suicida, manía, convulsiones); si no, "no_consta".
- criterio_gravedad: "muerte", "riesgo_vital", "hospitalizacion", "discapacidad",
  "otro_importante" o "".
- desenlace: "recuperado", "recuperandose", "no_recuperado" o "desconocido".

Menciones:
{menciones}

Responde SOLO con una lista JSON, un objeto por mención, con "id" y los campos de arriba:
[{{"id": "...", "es_efecto_adverso": "...", "motivo_no": "...", "reacciones": "...", "negadas": "",
  "paciente": "...", "edad": "...", "sexo": "...", "sexo_fuente": "...", "grave": "...",
  "criterio_gravedad": "", "desenlace": "..."}}]"""


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


PROMPT_V2 = """Eres técnico de farmacovigilancia y haces el triaje de posts de Bluesky. El
rastreador te pasa posts que podrían hablar de {farmaco}, cada uno con los mensajes anteriores
de su conversación. Para cada post decides si describe un efecto adverso de {farmaco} y, si lo
describe, extraes los datos del caso. Se anota lo que dice el post (leído con su conversación).

Lo que marques "si" lo revisará una persona; lo que marques "no" se pierde. Por eso, si el
autor cuenta algo malo que le pasó y lo relaciona con {farmaco}, aunque sea de pasada, de forma
vaga o en broma, marca "si". Marca "no" cuando de verdad no lo hay.

## Paso 1. ¿Habla el post de {farmaco}?

Es "no_es_mencion" (y es_efecto_adverso "no") si el post:
- es una respuesta de cortesía o apoyo sin nada propio sobre el fármaco: "gracias", "ánimo",
  "un abrazo", emojis;
- cuenta efectos de OTRO fármaco (escitalopram, paroxetina, duloxetina…) o de "un
  antidepresivo" sin que se sepa que es {farmaco}: náuseas con la duloxetina NO son un efecto de
  {farmaco}. Mira de qué fármaco habla la conversación: en un hilo sobre el escitalopram, "a mí
  me pasó lo mismo" es del escitalopram;
- usa el nombre para otra cosa (un personaje, una canción).

Pero si el post nombra otro fármaco y ADEMÁS atribuye algo a {farmaco} ("la otra pastilla me
va bien, aunque la libido me la quitó la {farmaco}"), sí habla de {farmaco}.

Un post que NO nombra el fármaco habla de él si la conversación deja claro que se refiere a
{farmaco}: "a mí también me pasó", "a mí me pasó por el insomnio", "me dio bajón" respondiendo a
alguien que habla de {farmaco}.

## Paso 2. ¿Es un efecto adverso?

"si" cuando el autor cuenta que A ÉL o a ALGUIEN CONCRETO le pasó algo malo, o distinto de lo
esperado, relacionado con {farmaco}:
- síntomas o molestias atribuidos al fármaco, aunque dude de la causa;
- quejas vagas pero propias: "entre la {farmaco} y sus efectos secundarios…", "la {farmaco} me
  sienta fatal", "me jode mucho", "fue un desastre": la reacción es lo que dice ("efectos
  secundarios", "me sienta fatal");
- síntomas al dejarla, al bajar la dosis o al OLVIDAR tomas ("3 días sin tomarla y estoy fatal",
  "toques eléctricos al bajarla"), y por tomar de más o mezclarla con alcohol;
- falta de eficacia: "llevo dos meses y no me hace nada";
- un motivo para dejarla que es un efecto: "la dejé porque no me dejaba dormir";
- AUNQUE sea en broma o en forma de diálogo, si los síntomas son del autor: "primer día con
  {farmaco} y solo quiero vomitar 😂";
- AUNQUE lo cuente dentro de una pregunta: "¿es normal estar mareada al dejar la {farmaco}? me
  siento rarísima" es un efecto adverso (mareo), no solo una pregunta.

"no" en todo lo demás, con su motivo_no:
- "pregunta_o_consejo": preguntas sin ningún síntoma propio ("¿la {farmaco} engorda?"),
  consejos ("habla con tu médico"), contar que se la recetaron sin más;
- "broma_opinion_noticia": bromas o ficción sin un síntoma propio (un personaje inventado al que
  le pasa algo), opiniones sobre los psicofármacos, noticias, memes;
- "sin_reaccion": la toma y no cuenta nada malo del fármaco ("me va genial", "volví a tomarla").
  Los síntomas de la ENFERMEDAD que trata (tristeza, ansiedad, ideas de suicidio, bajón) NO son
  un efecto adverso si el autor no se los achaca al fármaco: "sigo con ansiedad pero al menos
  duermo" es sin_reaccion. Un síntoma que el autor atribuye a otra causa también;
- "no_es_mencion": ver paso 1.

Cuidado con la conversación: los síntomas de un mensaje anterior son de quien los escribió, NO
del autor de este post, salvo que diga "a mí también", "me pasó lo mismo" o parecido; entonces
la reacción se deduce de la conversación.

## Paso 3. Datos del caso (solo si es_efecto_adverso = "si")

- reacciones: SIEMPRE al menos una, con las palabras del autor, separadas por " | ". Si solo se
  sabe por la conversación, la reacción deducida entre corchetes: "[náuseas]". Si el autor no
  dice cuál ("efectos secundarios"), escribe esas palabras.
- negadas: lo que el autor dice que NO tuvo; si nada, "".
- paciente: "autor" (le pasa a quien escribe), "familiar_o_conocido" o "desconocido".
- edad: el número SOLO si se dice ("tengo 34 años"); si no, "desconocido".
- sexo: "mujer" u "hombre" SOLO si se dice ("soy mujer", "mi madre") o si una palabra con género
  se refiere al paciente ("estoy cansada", "estoy mareado"); si no, "desconocido". No supongas
  "hombre" por defecto. sexo_fuente: "dicho", "deducido" o "" si desconocido.
- grave: "si" SOLO si el post dice muerte, riesgo vital, ingreso en el hospital, discapacidad u
  otro evento médicamente importante (intento de suicidio, convulsiones, manía) CAUSADO por el
  fármaco; si no, "no_consta" (lo normal). Urgencias SIN ingreso NO es hospitalización.
- criterio_gravedad: "muerte", "riesgo_vital", "hospitalizacion", "discapacidad",
  "otro_importante" o "" (vacío si grave = "no_consta").
- desenlace: "recuperado" ("se me quitó"), "recuperandose" ("ya voy mejor"), "no_recuperado"
  ("sigo igual") o "desconocido" (lo normal: no lo supongas).

Si es_efecto_adverso = "no", deja vacíos todos los campos del paso 3.

## Ejemplos (inventados)

- "Gracias por contarlo 🫂" (respondiendo a alguien con náuseas por {farmaco}) →
  {{"es_efecto_adverso": "no", "motivo_no": "no_es_mencion"}}
- "En mi caso el escitalopram me quitaba el sueño" (en un hilo sobre el escitalopram) →
  {{"es_efecto_adverso": "no", "motivo_no": "no_es_mencion"}}
- "Primera semana con {farmaco} y estoy agotada todo el día jajaja" →
  {{"es_efecto_adverso": "si", "reacciones": "estoy agotada todo el día", "paciente": "autor",
  "edad": "desconocido", "sexo": "mujer", "sexo_fuente": "deducido", "grave": "no_consta",
  "criterio_gravedad": "", "desenlace": "desconocido"}}
- "Vaya mes: entre la {farmaco} y sus efectos secundarios no levanto cabeza" →
  {{"es_efecto_adverso": "si", "reacciones": "efectos secundarios | no levanto cabeza", …}}
- "¿Alguien sabe si es normal el dolor de cabeza al bajar la {farmaco}? Llevo así tres días" →
  {{"es_efecto_adverso": "si", "reacciones": "dolor de cabeza al bajar la {farmaco}", …}}
- "A mí me pasó al principio, luego se fue" (respondiendo a "con {farmaco} tengo diarrea") →
  {{"es_efecto_adverso": "si", "reacciones": "[diarrea]", "paciente": "autor",
  "desenlace": "recuperado", …}}
- "¿La {farmaco} quita el apetito?" → {{"es_efecto_adverso": "no", "motivo_no":
  "pregunta_o_consejo"}}
- "Volví a la {farmaco}, a ver si esta vez se me pasa la ansiedad" → {{"es_efecto_adverso":
  "no", "motivo_no": "sin_reaccion"}}

## Posts

{menciones}

Responde SOLO con una lista JSON, un objeto por post, con su "id" y los campos de arriba:
[{{"id": "...", "es_efecto_adverso": "si|no", "motivo_no": "...", "reacciones": "...",
  "negadas": "", "paciente": "...", "edad": "...", "sexo": "...", "sexo_fuente": "...",
  "grave": "...", "criterio_gravedad": "", "desenlace": "..."}}]"""


PROMPTS = {"v0": PRELABEL_PROMPT, "v1": PROMPT_V1, "v2": PROMPT_V2}


REPAIR = """Algunas respuestas no se pudieron usar:
{problemas}

Responde de nuevo SOLO para estos ids ({ids}), con una lista JSON en el mismo formato y los
valores exactos de las instrucciones."""


def build_prompt(version: str, mentions: list[RawMention]) -> str:
    if version not in PROMPTS:
        raise ValueError(f"unknown prompt version {version!r}; use one of {sorted(PROMPTS)}")
    return PROMPTS[version].format(farmaco=mentions[0].farmaco, menciones=render_mentions(mentions))
