# Contrato rastreador → triaje

Qué recibe el triaje del rastreador, para poder desarrollarlo en paralelo. Versión del formato:
**1.0** (campo `version`).

## En una frase

El rastreador entrega **una línea JSON por cada post de Bluesky que habla de un fármaco vigilado**
(una `RawMention`), con el texto del post y lo mínimo de su conversación para entenderlo. **No
valora nada**: decidir si el post describe una reacción adversa, extraer paciente, gravedad, etc.
es trabajo del triaje.

## Dónde está

| Qué | Dónde |
|---|---|
| Menciones reales (vigilante periódico, sistema híbrido) | `data/bluesky/monitor/menciones.jsonl` en la máquina donde se ejecute `crawler.monitor` (fuera de git) |
| **Ejemplos inventados para desarrollar y probar** | [`contrato/ejemplo_menciones.jsonl`](contrato/ejemplo_menciones.jsonl) (7 casos, en el repositorio) |
| Esquema formal (JSON Schema), para no depender de Python | [`contrato/raw_mention.schema.json`](contrato/raw_mention.schema.json) |
| Definición en código (Pydantic, la fuente de verdad) | [`store.py`](store.py), clase `RawMention` |

El fichero real es **JSONL**: una mención por línea, y el vigilante **solo añade líneas al final**
en cada ejecución. Para procesar solo las nuevas, guardad los `id` ya vistos.

## Los campos

| Campo | Tipo | Siempre | Qué es |
|---|---|---|---|
| `version` | texto | sí | Versión del formato (`"1.0"`) |
| `id` | texto | sí | Id de la mención: `M-` + 10 caracteres. Mismo post y mismo fármaco → mismo id. **Usadlo para descartar duplicados** |
| `fuente` | texto | sí | De momento siempre `"bluesky"` |
| `post_id` | texto | sí | Id del post: `p` + 12 caracteres. Es un hash: **no identifica al autor ni se puede convertir en un enlace** |
| `hilo_id` | texto | sí | `post_id` del primer post de la conversación. **Menciones con el mismo `hilo_id` son de la misma conversación**: pueden ser el mismo caso contado por varias personas o la misma persona en varios mensajes |
| `farmaco` | texto | sí | Principio activo vigilado, en minúsculas: `sertralina`, `escitalopram`, `lorazepam`, `alprazolam`, `zolpidem`, `tramadol`, `metformina`, `levotiroxina`. Un post que habla de dos fármacos genera **dos menciones** |
| `tipo_mencion` | texto | sí | Cómo nombra el fármaco: `nombre`, `marca` (Besitran), `errata` ("sertra", "lorazepan"), `indirecta` (no lo nombra; se sabe por la conversación), `jerga` (lo describe: "las pastillas de la tiroides") |
| `texto` | texto | sí | **El post completo**, copiado tal cual de Bluesky (nunca lo escribe un modelo) y anonimizado: `@usuario`, `[enlace]`, `[correo]` |
| `contexto.inicio_hilo` | texto o `null` | no | Texto del primer post de la conversación. `null` si la mención **es** el primer post |
| `contexto.responde_a` | texto o `null` | no | Texto del post al que responde directamente. `null` si no es una respuesta o si responde al primer post (entonces está en `inicio_hilo`) |
| `idioma` | texto o `null` | no | Idioma declarado por el autor (`"es"`, a veces `"ca"`, `"pt"`…) |
| `fecha` | texto | sí | Cuándo se publicó el post, ISO 8601 en UTC (`2026-03-02T10:15:00Z`) |
| `encontrado_por` | texto | sí | `busqueda_fija`: el post contiene el nombre, una marca o una errata conocida (casi siempre acierta). `agente`: el agente leyó la conversación y decidió que habla del fármaco (menciones indirectas; en la evaluación **1 de cada 4 no hablaba realmente de él**) |
| `motivo` | texto | sí | Por qué se guardó. En las de `busqueda_fija` lo escribe el código (`contiene 'Besitran'`); **en las del `agente` está vacío**: para entender por qué una respuesta habla del fármaco, usad `contexto` (texto real), no una explicación del modelo |
| `guardado` | texto | sí | Cuándo lo guardó el rastreador, ISO 8601 en UTC |

**¿Y el "híbrido"?** Es el sistema que usa el vigilante, no un valor de `encontrado_por`: cada mención la encuentra una de sus dos partes, la búsqueda fija (el código) o el agente que lee las conversaciones, y por eso con el híbrido aparecen los dos valores. Distinguirlas importa porque su fiabilidad es muy distinta: en la evaluación acertaron 46 de 47 y 18 de 25.

## Un ejemplo: una pregunta y dos respuestas

```json
{"version": "1.0", "id": "M-56a92a5028", "fuente": "bluesky",
 "post_id": "p546aff1211d9", "hilo_id": "p546aff1211d9",
 "farmaco": "sertralina", "tipo_mencion": "nombre",
 "texto": "Llevo tres semanas con sertralina y me paso el día con náuseas, ¿a alguien más le pasa?",
 "contexto": {"inicio_hilo": null, "responde_a": null},
 "idioma": "es", "fecha": "2026-03-02T10:15:00Z",
 "encontrado_por": "busqueda_fija", "motivo": "contiene 'sertralina'",
 "guardado": "2026-11-20T09:00:00+00:00"}
```

```json
{"version": "1.0", "id": "M-2e2c777ec9", "post_id": "pdd2984e9aa82", "hilo_id": "p546aff1211d9",
 "farmaco": "sertralina", "tipo_mencion": "indirecta",
 "texto": "A mí también me pasó al principio, a las dos semanas se me quitó",
 "contexto": {"inicio_hilo": "Llevo tres semanas con sertralina y me paso el día con náuseas, ¿a alguien más le pasa?",
              "responde_a": null},
 "encontrado_por": "agente", "...": "…"}
```

La segunda **no nombra el fármaco**: para entenderla, el triaje tiene que leer `texto` junto con
`contexto` ("me pasó" = náuseas con sertralina). Una respuesta a esa respuesta tendría
`responde_a` relleno y el mismo `inicio_hilo`.

Los 7 ejemplos de [`contrato/ejemplo_menciones.jsonl`](contrato/ejemplo_menciones.jsonl) están
pensados para cubrir los casos que el triaje tiene que saber tratar:

| # | Caso | Qué debería hacer el triaje |
|---|---|---|
| 1 | Experiencia con síntoma, nombra el fármaco | Efecto adverso (náuseas) |
| 2 | Respuesta indirecta ("a mí también me pasó") | Efecto adverso, entendido gracias a `contexto` |
| 3 | Respuesta a una respuesta | Efecto adverso (mareo al bajar dosis); usar `responde_a` |
| 4 | Marca en lugar del principio activo (Besitran) | Efecto adverso (insomnio); `farmaco` ya viene normalizado |
| 5 | Broma que nombra el fármaco | **No** es un efecto adverso: descartar |
| 6 | Error del agente: un "gracias" en un hilo sobre el fármaco | **No** es una mención: descartar |
| 7 | Otro fármaco, mal escrito ("lorazepan") | Efecto adverso (somnolencia) |

## Qué puede dar por hecho el triaje

- **El texto es real.** Se copia del post; ningún modelo lo reescribe ni lo resume.
- **No hay duplicados** de la misma pareja post–fármaco: el `id` es estable entre ejecuciones.
- **Todo está anonimizado**: no hay nombres de usuario, enlaces ni correos. Para un ICSR, el
  notificante será siempre "consumidor" de identidad desconocida; no se le puede contactar.
- **Las menciones por `busqueda_fija` son casi siempre del fármaco** (46 de 47, 98 %, en la
  evaluación; el error fue el nombre de un personaje de novela parecido a una marca).

## Qué NO puede dar por hecho

- **Que sea un efecto adverso.** El rastreador guarda dudas, bromas, opiniones, noticias y
  experiencias sin síntomas. En la evaluación, de cada 10 menciones de sertralina, unas 8 contaban
  una experiencia propia o de alguien cercano; el resto eran dudas, bromas u opiniones.
- **Que hable del fármaco**, si `encontrado_por` es `agente`: en la evaluación, 7 de 25 (28 %)
  eran respuestas de cortesía o apoyo ("gracias", emojis) dentro de una conversación sobre el
  fármaco. El triaje debe poder
  decir "esto no es una mención".
- **Que cada mención sea un caso distinto**: varias del mismo `hilo_id` pueden ser la misma
  persona o el mismo caso.
- **Que esté en español**: hay posts en catalán, gallego o portugués (mirad `idioma`).
- **Imágenes.** De momento solo se entrega texto. Si el triaje multimodal las necesita, se
  pueden añadir como campo opcional (`imagenes`) sin romper el formato: pedidlo.

## Cómo leerlo

En Python, validando cada línea contra el contrato:

```python
from crawler.store import read_mentions

for m in read_mentions("crawler/contrato/ejemplo_menciones.jsonl"):
    print(m.id, m.farmaco, m.tipo_mencion, m.encontrado_por)
    print("  texto:", m.texto)
    if m.contexto.inicio_hilo:
        print("  hilo:", m.contexto.inicio_hilo)
```

Sin el repositorio, cualquier lector de JSON sirve; el esquema
[`raw_mention.schema.json`](contrato/raw_mention.schema.json) permite validar.

Para pedir menciones nuevas desde código (lo que hará el coordinador), el vigilante es una
función:

```python
from pathlib import Path

from crawler.monitor import load_state, run_once
from crawler.sources import LiveSource

state = load_state(Path("data/bluesky/monitor/estado.json"))
store, resumen = run_once(LiveSource(), state, drugs=["sertralina"], brain="27b")
menciones = list(store.mentions.values())  # list[RawMention]
```

## Cambios en el formato

- Los campos nuevos se añadirán siempre como **opcionales**, de modo que un fichero antiguo siga
  siendo válido. Quitar o renombrar un campo exigirá subir `version` a 2.0 y avisar antes.
- Si al triaje le falta algo (por ejemplo, el número de respuestas del post o las imágenes),
  pedidlo: se añade en `store.py` y se regeneran el esquema y los ejemplos con
  `uv run python crawler/contrato/generar_ejemplos.py`. Un test comprueba que no se desincronizan.
