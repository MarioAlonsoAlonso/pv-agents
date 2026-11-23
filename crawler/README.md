# Rastreador · menciones de fármacos en Bluesky

**Qué hace:** encuentra en Bluesky los posts en español que hablan de un fármaco vigilado,
también los que **no lo nombran** ("a mí también me dejó KO", respondiendo a alguien que toma
sertralina), y se los pasa al triaje como `RawMention`. **No valora nada**: decidir si un post
describe una reacción adversa es trabajo del triaje.

**Su segundo papel:** sub-agente de investigación del analista (`buscar_evidencia`). Ante una
pregunta abierta ("¿hay más casos de caídas en ancianos con sertralina?") busca en Bluesky y
PubMed y devuelve un resumen con referencias.

**Sistema elegido: el híbrido** (búsqueda fija + agente que lee las conversaciones). En la
evaluación sobre sertralina encuentra el 97 % de las menciones (17 de 19 indirectas) frente al 70 %
de la búsqueda fija, con la mitad de coste que el agente solo. Es el que usa el vigilante
(`crawler.monitor`). Los otros tres sistemas siguen implementados para comparar.

| Documento | Para quién |
|---|---|
| [`RESULTADOS.md`](RESULTADOS.md) | Cifras, las cuatro versiones y sus fallos, límites (informe) |
| [`CONTRATO_TRIAJE.md`](CONTRATO_TRIAJE.md) | Qué recibe el triaje: formato, ejemplos, qué puede dar por hecho |
| [`ANOTACION.md`](ANOTACION.md) | Cómo se anotó la hoja de soluciones |

## Por qué hay un agente aquí (y por qué no en todo)

Una búsqueda fija por principio activo y marcas encuentra casi todos los posts que **nombran**
el fármaco, sin coste y sin errores. El agente solo se justifica donde esa búsqueda no llega:

| | Búsqueda fija | Agente |
|---|---|---|
| Posts que nombran el fármaco o su marca | ✅ | ✅ (más caro) |
| Respuestas de un hilo que hablan de él sin nombrarlo | ❌, o todo el hilo con ruido (`fija_hilos`) | ✅ elige cuáles |
| Formas nuevas de nombrarlo ("la sertra", Zoloft) | ❌ solo lo que está en la lista | ✅ las descubre y las propone |
| Preguntas abiertas del analista | ❌ | ✅ |

Por eso el diseño es **híbrido**: el código hace la búsqueda masiva y el agente lee los hilos
y aprende términos nuevos. La evaluación compara las cuatro opciones para comprobarlo con datos.

## Piezas

```
crawler/
  config.py     fármacos vigilados, términos amplios, carpeta de datos
  cima.py       marcas comerciales desde la API de CIMA → crawler/data/marcas.json (versionado)
  text.py       anonimización (@usuarios, enlaces, correos), ids con hash, coincidencia sin tildes
  sources.py    LiveSource (Bluesky en vivo) y FrozenSource (muestra congelada): misma interfaz
  pubmed.py     PubMed (E-utilities), para el modo dirigido
  citas.py      comprueba que cada PMID y post citado lo devolvió una herramienta
  store.py      RawMention: el contrato con el triaje
  tools.py      herramientas del agente, con los controles dentro
  prompts.py    un prompt por modo: vigilancia, hilos, dirigido
  agent.py      el agente: el bucle ReAct de la fase 4 con otras herramientas y cerebros
  systems.py    los cuatro sistemas que se comparan: fija, fija_hilos, hibrido, agente
  collect.py    construye la muestra congelada
  annotate.py   lote para anotar, pre-anotación, kappa, gold
  evaluate.py   recall por tipo de mención, precisión y coste de cada sistema
  eval_dirigido.py  evaluación del modo dirigido (preguntas en bench/dirigido.jsonl)
  monitor.py    vigilancia periódica
  ANOTACION.md  guía de anotación
```

### El agente

Es el `ReActAgent` de [`agent/react_agent.py`](../agent/react_agent.py) sin reescribir el bucle
(una acción por paso, timeouts, errores como observación, detección de bucles, memoria). Cambian
las herramientas, el prompt y el cerebro:

| Cerebro | Qué es |
|---|---|
| `27b` | Qwen3.8-27B-FP8 del servidor de los profesores en la DGX (`common/llm.py: RemoteChatModel`, API compatible con OpenAI) |
| `27b-thinking` | El mismo con razonamiento |
| `base` | Qwen3-1.7B local, para medir si rastrear necesita un modelo grande |

**Herramientas** ([`tools.py`](tools.py)). La regla es que lo que se puede garantizar con código
no se le pide al modelo:

| Herramienta | Qué hace | Control en el código |
|---|---|---|
| `nombres_comerciales` | Marcas de CIMA | Dato oficial, no memoria del modelo |
| `buscar_bluesky` | Búsqueda por palabras, 100 resultados | Presupuesto de búsquedas; solo el periodo de la tarea |
| `leer_hilo` | La conversación entera, por partes | Presupuesto de hilos; marca los posts fuera del periodo |
| `guardar_menciones` | Guarda de una vez todos los posts relevantes que acaba de ver | El modelo da **ids**, el texto se copia de la fuente: solo puede guardar posts que ha visto, sin duplicados; un id inventado se rechaza sin perder los demás |
| `proponer_termino` | Una forma nueva de nombrar el fármaco | Se añade a las búsquedas del vigilante |
| `buscar_pubmed` | Literatura (solo modo dirigido) | — |

**Modos:** `vigilancia` (el agente solo), `hilos` (la mitad del híbrido: lee los hilos donde la
búsqueda fija encontró el fármaco) y `dirigido` (preguntas del analista).

### Privacidad

Los posts son públicos, pero son datos de salud de personas. Por eso:
- el texto se anonimiza al leerlo (`@usuario`, `[enlace]`, `[correo]`);
- los posts se identifican con un hash: el AT-URI (que contiene el identificador del autor) no
  llega nunca al modelo, a la muestra ni a los informes;
- todo se guarda en `data/`, que está en `.gitignore`. A `reports/` solo van cifras.

## Cómo se usa (en la DGX)

```bash
uv sync --extra train --extra rag --extra agent --extra crawler   # o make setup
cp .env.example .env    # BSKY_HANDLE y BSKY_APP_PASSWORD (contraseña de aplicación)
```

| Paso | Comando | GPU / 27B |
|---|---|---|
| 1. Muestra congelada (8 fármacos desde 2025-01-01, ~1 h, reanudable) | `make crawler-muestra` | No |
| 2. Lote a anotar | `make crawler-lote` | No |
| 3. Pre-anotación (opcional) | `make crawler-prenotar` | 27B |
| 4. Anotar (ver [`ANOTACION.md`](ANOTACION.md)) y consolidar | `make crawler-gold` | No |
| 5. Evaluación | `make crawler-eval` | 27B (híbrido y agente) |
| 6. Vigilancia periódica | `make crawler-vigilar` | 27B (o `--sin-agente`) |

El servidor del 27B escucha en `0.0.0.0:8001` con clave API (`ARCA_REMOTE_LLM_API_KEY` en `.env`,
la del correo de los profesores) y solo es accesible desde una sesión de la DGX; es compartido (5 peticiones simultáneas en total, ~50 tokens/s), así que la evaluación
del agente puede tardar. `ARCA_REMOTE_LLM_CONCURRENCY` (2 por defecto) limita cuántas lanzamos.

Probar el agente a mano y leer su traza (se guarda en `data/bluesky/demo/`):

```bash
uv run --extra crawler python -m crawler.demo vigilar sertralina --dias 7          # agente solo, en vivo
uv run --extra crawler python -m crawler.demo hilos lorazepam --dias 14             # híbrido, en vivo
uv run --extra crawler python -m crawler.demo investigar sertralina "¿caídas en ancianos?"
uv run python -m crawler.demo vigilar sertralina --muestra data/bluesky/sample.jsonl  # sobre la muestra
```

Tras `crawler.evaluate`, en `data/bluesky/eval/` quedan, por sistema, las menciones, las
**trazas** del agente y `*_errores.json`: las menciones anotadas que se perdió (`perdidos`) y
lo que envió sin que hablara del fármaco (`sobrantes`), con el texto, para el análisis de fallos.

Modo dirigido, desde Python:

```python
from crawler.agent import investigar
from crawler.cima import load_brands
from crawler.sources import LiveSource

run = investigar("¿Hay casos de caídas en ancianos?", "sertralina", LiveSource(), load_brands())
print(run.response.final_answer)
```

La herramienta `buscar_evidencia` (en `crawler/agent.py`) hace lo mismo y se puede registrar en
el analista: `agent.registry.register(buscar_evidencia)`. Devuelve `resumen`, `posts`,
`articulos` y `referencias_sin_respaldo`: los PMID o posts que el resumen cita pero que ninguna
búsqueda devolvió (inventados); el analista no debe citarlos.

### Evaluación del modo dirigido

[`bench/dirigido.jsonl`](bench/dirigido.jsonl): 12 preguntas del analista con respuesta en la
literatura (hiponatremia con sertralina, síndrome serotoninérgico con tramadol, acidosis láctica
con metformina…) y 2 controles sin ella (¿cambia la sertralina el color de los ojos?). Cada
pregunta lleva los términos del fármaco y de la reacción en inglés y 3 PMID de referencia,
buscados en PubMed por título y comprobados el 2026-11-23. Se comparan la búsqueda fija (una
consulta en inglés, los 10 primeros artículos, sin modelo) y el agente.

```bash
uv run python -m crawler.eval_dirigido --sistemas fija                   # sin modelo, en cualquier sitio
uv run --extra crawler python -m crawler.eval_dirigido --brain 27b       # DGX: fija + agente
uv run --extra crawler python -m crawler.eval_dirigido --ids caidas_zolpidem --sistemas agente
```

| Medida | Qué es |
|---|---|
| **Citas sin respaldo** | PMID y posts citados en la respuesta que ninguna herramienta devolvió. Lo más grave: llegarían al informe como evidencia |
| Relevancia | Artículo con un término del fármaco y uno de la reacción en el título o el resumen. Aproximada, automática e igual para los dos sistemas |
| Referencia encontrados | Cuántos de los 3 PMID de referencia aparecen. Hay muchos otros artículos válidos: sirve para comparar, no como nota absoluta |
| Controles | Si la respuesta admite que no hay evidencia (heurística: leedlas) |
| Coste | Tokens, pasos, segundos, búsquedas en PubMed y Bluesky |

Las cifras van a `reports/crawler_dirigido_eval.json` (sin texto); respuestas, trazas y posts, a
`data/bluesky/dirigido/<cerebro>/`. Búsqueda fija (2026-11-23): **76 % de artículos relevantes
[68-83]** y 16 de 36 de referencia (44 %).

## La evaluación

Los cuatro sistemas se ejecutan sobre la **misma muestra congelada** (mismos posts, misma
búsqueda, siempre) y se puntúan solo sobre los posts **anotados**:

- **Recall** = menciones anotadas que el sistema encuentra, también **por tipo**: `nombre`,
  `marca`, `errata`, `indirecta` y `jerga`. Los dos últimos son donde el agente tiene que ganar
  para justificarse.
- **Precisión** = de lo que envía al triaje (entre lo anotado), cuánto habla del fármaco.
- **Coste**: posts enviados al triaje, tokens, pasos, hilos leídos y tiempo.

La muestra **no es circular**: además de los nombres y las marcas, se construye con términos
amplios ("antidepresivos", "me han recetado", "efectos secundarios"…) y con los hilos completos
de lo encontrado, así que contiene menciones que la búsqueda fija no puede ver.

## Lo que sabemos de la API (probado el 2026-11-18)

- Con cuenta, la búsqueda pagina con cursor. Sin cuenta solo da la primera página (la segunda
  devuelve 403); `LiveSource` lo resuelve partiendo el rango de fechas por la mitad mientras
  una ventana vuelva llena.
- `getPostThread` funciona sin cuenta.
- La búsqueda es por palabras, no por significado: "sertralina" no encuentra "la sertra".
- El volumen es bajo: unos 530 posts en español con "sertralina" desde mediados de 2025.

## Limitaciones conocidas

- Solo Bluesky. Otras fuentes HTML (foros) necesitarían una herramienta `leer_pagina` y comprobar
  la cita literal, porque allí no hay ids de post.
- Los términos amplios con mucho volumen ("efectos secundarios") se cortan en 1 500 posts.
- Las marcas de CIMA excluyen los genéricos EFG con nombre de fantasía y las marcas que son
  palabras corrientes ("Captor"); las extranjeras (Zoloft) solo las encuentra el agente.
