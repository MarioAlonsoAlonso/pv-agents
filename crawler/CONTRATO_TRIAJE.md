# Contrato rastreador → triaje

Qué recibe el triaje del rastreador, para poder desarrollarlo en paralelo. Versión del formato:
**1.1** (campo `version`).

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
| `version` | texto | sí | Versión del formato (`"1.1"`) |
| `id` | texto | sí | Id de la mención: `M-` + 10 caracteres. Mismo post y mismo fármaco → mismo id. **Usadlo para descartar duplicados** |
| `fuente` | texto | sí | De momento siempre `"bluesky"` |
| `post_id` | texto | sí | Id del post: `p` + 12 caracteres. Es un hash: **no identifica al autor ni se puede convertir en un enlace** |
| `hilo_id` | texto | sí | `post_id` del primer post de la conversación. **Menciones con el mismo `hilo_id` son de la misma conversación**: pueden ser el mismo caso contado por varias personas o la misma persona en varios mensajes |
| `farmaco` | texto | sí | Principio activo vigilado, en minúsculas: `sertralina`, `escitalopram`, `lorazepam`, `alprazolam`, `zolpidem`, `tramadol`, `metformina`, `levotiroxina`. Un post que habla de dos fármacos genera **dos menciones** |
| `tipo_mencion` | texto | sí | Cómo nombra el fármaco: `nombre`, `marca` (Besitran), `errata` ("sertra", "lorazepan"), `indirecta` (no lo nombra; se sabe por la conversación), `jerga` (lo describe: "las pastillas de la tiroides") |
| `texto` | texto | sí | **El post completo**, copiado tal cual de Bluesky (nunca lo escribe un modelo) y anonimizado: `@usuario`, `[enlace]`, `[correo]` |
| `contexto.inicio_hilo` | texto o `null` | no | Texto del primer post de la conversación. `null` si la mención **es** el primer post |
| `contexto.responde_a` | texto o `null` | no | Texto del post al que responde directamente. `null` si no es una respuesta o si responde al primer post (entonces está en `inicio_hilo`) |
| `conversacion` | lista | sí (vacía si es el primer post) | **Todos los mensajes anteriores de la cadena**, desde el primero de la conversación hasta el que responde este post, en orden (el más antiguo primero). Cada uno con `post_id` y `texto`. El propio post no está: está en `texto` |
| `conversacion_completa` | sí/no | sí | `false` si falta algún mensaje de la cadena (borrado, bloqueado o fuera de la muestra): entonces `conversacion` empieza en el más antiguo disponible |
| `idioma` | texto o `null` | no | Idioma con el que **la aplicación del autor etiquetó el post** (`"es"`, a veces `"ca"`, `"pt"`…). No se detecta del texto: ver más abajo |
| `fecha` | texto | sí | Cuándo se publicó el post, ISO 8601 en UTC (`2026-03-02T10:15:00Z`) |
| `encontrado_por` | texto | sí | `busqueda_fija`: el post contiene el nombre, una marca o una errata conocida (casi siempre acierta). `agente`: el agente leyó la conversación y decidió que habla del fármaco (menciones indirectas; en la evaluación **1 de cada 4 no hablaba realmente de él**) |
| `motivo` | texto | sí | Por qué se guardó. En las de `busqueda_fija` lo escribe el código (`contiene 'Besitran'`); **en las del `agente` está vacío**: para entender por qué una respuesta habla del fármaco, usad `contexto` (texto real), no una explicación del modelo |
| `guardado` | texto | sí | Cuándo lo guardó el rastreador, ISO 8601 en UTC |

**¿Y el "híbrido"?** Es el sistema que usa el vigilante, no un valor de `encontrado_por`: cada mención la encuentra una de sus dos partes, la búsqueda fija (el código) o el agente que lee las conversaciones, y por eso con el híbrido aparecen los dos valores. Distinguirlas importa porque su fiabilidad es muy distinta: en la evaluación acertaron 46 de 47 y 18 de 25.

## Un ejemplo: una pregunta y dos respuestas

```json
{"version": "1.1", "id": "M-56a92a5028", "fuente": "bluesky",
 "post_id": "p546aff1211d9", "hilo_id": "p546aff1211d9",
 "farmaco": "sertralina", "tipo_mencion": "nombre",
 "texto": "Llevo tres semanas con sertralina y me paso el día con náuseas, ¿a alguien más le pasa?",
 "contexto": {"inicio_hilo": null, "responde_a": null},
 "conversacion": [], "conversacion_completa": true,
 "idioma": "es", "fecha": "2026-03-02T10:15:00Z",
 "encontrado_por": "busqueda_fija", "motivo": "contiene 'sertralina'",
 "guardado": "2026-11-20T09:00:00+00:00"}
```

```json
{"version": "1.1", "id": "M-2e2c777ec9", "post_id": "pdd2984e9aa82", "hilo_id": "p546aff1211d9",
 "farmaco": "sertralina", "tipo_mencion": "indirecta",
 "texto": "A mí también me pasó al principio, a las dos semanas se me quitó",
 "contexto": {"inicio_hilo": "Llevo tres semanas con sertralina y me paso el día con náuseas, ¿a alguien más le pasa?",
              "responde_a": null},
 "conversacion": [{"post_id": "p546aff1211d9",
                   "texto": "Llevo tres semanas con sertralina y me paso el día con náuseas, ¿a alguien más le pasa?"}],
 "conversacion_completa": true,
 "encontrado_por": "agente", "motivo": "", "...": "…"}
```

La segunda **no nombra el fármaco**: para entenderla, el triaje tiene que leer `texto` junto con
su conversación ("me pasó" = náuseas con sertralina).

**Cadenas largas.** `contexto` solo trae el primer mensaje y el inmediatamente anterior; en una
cadena *inicio → A → B → C*, la mención C no vería A. Por eso existe `conversacion` (desde la 1.1):
trae la cadena entera, `[inicio, A, B]`, en orden. **Para interpretar una mención, leed
`conversacion` + `texto`**; `contexto` se mantiene por compatibilidad. La cadena son los mensajes
a los que responde este post, no todas las respuestas del hilo: las ramas paralelas no aparecen.

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
- **Que esté en español**: la búsqueda pide posts en español, pero las respuestas de un hilo
  entran en cualquier idioma; en el lote evaluado había alguno en catalán, gallego y portugués.
- **Que `idioma` sea exacto**: Bluesky no detecta el idioma del texto; lo pone la aplicación del
  autor al publicar, por defecto el idioma en que la tiene configurada, y el autor puede cambiarlo
  en cada post. Alguien con la aplicación en español que escribe en catalán aparece como `"es"`.
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

- **1.1** (2026-11-20): `conversacion` y `conversacion_completa`. Un fichero 1.0 se sigue leyendo
  (la conversación queda vacía).

- Los campos nuevos se añadirán siempre como **opcionales**, de modo que un fichero antiguo siga
  siendo válido. Quitar o renombrar un campo exigirá subir `version` a 2.0 y avisar antes.
- Si al triaje le falta algo (por ejemplo, el número de respuestas del post o las imágenes),
  pedidlo: se añade en `store.py` y se regeneran el esquema y los ejemplos con
  `uv run python crawler/contrato/generar_ejemplos.py`. Un test comprueba que no se desincronizan.
