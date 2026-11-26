# Rastreador · menciones de fármacos en Bluesky

**Qué hace:** encuentra en Bluesky los posts en español que hablan de un fármaco vigilado,
también los que **no lo nombran** ("a mí también me dejó KO", respondiendo a alguien que toma
sertralina), y se los pasa al triaje como `RawMention`. **No valora nada**: decidir si un post
describe una reacción adversa es trabajo del triaje.

**Su segundo papel:** dos herramientas para el analista. `buscar_literatura` pregunta a PubMed
qué se ha publicado sobre un fármaco y una reacción (si está descrita, en quién, el mecanismo) y
devuelve un resumen con PMID; `buscar_mas_casos` busca en Bluesky más personas que cuenten esa
reacción con ese fármaco.

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
  pubmed.py     PubMed (E-utilities), para buscar_literatura
  citas.py      comprueba que cada PMID y post citado lo devolvió una herramienta
  store.py      RawMention: el contrato con el triaje
  tools.py      herramientas del agente, con los controles dentro
  prompts.py    un prompt por modo: vigilancia, hilos, literatura, casos
  agent.py      el agente: el bucle ReAct de la fase 4 con otras herramientas y cerebros
  systems.py    los cuatro sistemas que se comparan: fija, fija_hilos, hibrido, agente
  collect.py    construye la muestra congelada
  annotate.py   lote para anotar, pre-anotación, kappa, gold
  evaluate.py   recall por tipo de mención, precisión y coste de cada sistema
  eval_literatura.py  evaluación de buscar_literatura (bench/literatura*.jsonl)
  eval_casos.py      evaluación de buscar_mas_casos (bench/casos*.jsonl)
  stats.py           proporciones con intervalo de Wilson
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
| `buscar_pubmed` | Literatura (solo modo `literatura`) | Las citas se comprueban (`citas.py`) |

**Modos:** `vigilancia` (el agente solo), `hilos` (la mitad del híbrido: lee los hilos donde la
búsqueda fija encontró el fármaco) y, para el analista, `literatura` (solo PubMed) y `casos`
(solo Bluesky, sin `proponer_termino`).

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

**Todo lo de un fármaco desde una fecha** (un fármaco nuevo en la lista, o rellenar un hueco):
`uv run --extra crawler python -m crawler.monitor --farmacos lorazepam --desde 2026-01-01`. Es
el vigilante con el inicio fijado: búsqueda fija desde esa fecha, sus hilos y el agente sobre
ellos, y las menciones van a `data/bluesky/monitor/menciones.jsonl`. Las siguientes ejecuciones
siguen desde ahora. Con meses de posts el agente lee muchos hilos (tokens), y cada término de
búsqueda trae como mucho 2000 posts.

El servidor del 27B escucha en `0.0.0.0:8001` con clave API (`ARCA_REMOTE_LLM_API_KEY` en `.env`,
la del correo de los profesores) y solo es accesible desde una sesión de la DGX; es compartido (5 peticiones simultáneas en total, ~50 tokens/s), así que la evaluación
del agente puede tardar. `ARCA_REMOTE_LLM_CONCURRENCY` (2 por defecto) limita cuántas lanzamos.

Probar el agente a mano y leer su traza (se guarda en `data/bluesky/demo/`):

```bash
uv run --extra crawler python -m crawler.demo vigilar sertralina --dias 7          # agente solo, en vivo
uv run --extra crawler python -m crawler.demo hilos lorazepam --dias 14             # híbrido, en vivo
uv run --extra crawler python -m crawler.demo literatura sertralina "¿hiponatremia en ancianos?"
uv run --extra crawler python -m crawler.demo casos sertralina bruxismo   # Bluesky en vivo
uv run python -m crawler.demo vigilar sertralina --muestra data/bluesky/sample.jsonl  # sobre la muestra
```

Tras `crawler.evaluate`, en `data/bluesky/eval/` quedan, por sistema, las menciones, las
**trazas** del agente y `*_errores.json`: las menciones anotadas que se perdió (`perdidos`) y
lo que envió sin que hablara del fármaco (`sobrantes`), con el texto, para el análisis de fallos.

## Las herramientas del analista

El analista valida señales (GVP módulo IX): si la reacción es conocida, si es plausible, en quién
es más probable. La ficha técnica (RAG) dice si está en la ficha; FAERS, cuántas notificaciones
hay. El rastreador aporta dos cosas más, cada una como herramienta aparte:

| Herramienta (en `crawler/agent.py`) | Cuándo | Devuelve |
|---|---|---|
| `buscar_literatura(farmaco, pregunta)` | La reacción no está en la ficha, o para valorar plausibilidad y grupos de riesgo | `resumen` con PMID, `articulos`, `referencias_sin_respaldo` (PMID citados que ninguna búsqueda devolvió: no citarlos) |
| `buscar_mas_casos(farmaco, reaccion)` | La señal viene de redes y la reacción es algo que el paciente nota y cuenta (no un análisis) | `resumen` y `casos`: los posts guardados, con su texto real |

**Decisión (2026-11-23): `buscar_mas_casos` queda aparcada.** El sistema solo vigila fármacos de la
lista, y para ellos los casos ya están en la base de menciones que recoge el vigilante y clasifica
el triaje: contar cuántos hay de una reacción es una consulta a esa base, que hace el analista,
no una búsqueda nueva con un agente (buscar palabras en las menciones recogidas ya encuentra el
84 % de los casos). Sigue implementada y con su evaluación preparada por si se vigilan fármacos
nuevos o hace falta buscar antes del inicio de la vigilancia, pero no se ha medido.

Se registra en el analista con `registry.register(buscar_literatura)`.
Desde Python: `investigar_literatura(pregunta, farmaco)` y `buscar_casos(farmaco, reaccion,
fuente, marcas)`.

**Por qué dos y no una.** Hasta el 2026-11-23 era una sola (`buscar_evidencia`, Bluesky y PubMed
a la vez). En 14 preguntas no inventó ninguna cita (0 de 81 PMID), pero gastaba los pasos en
Bluesky (8,1 búsquedas por pregunta, casi todas vacías: nadie escribe "tengo el sodio bajo") y
buscaba una vez en PubMed: encontró 6 de 36 artículos de referencia frente a 16 de una búsqueda
fija. Son dos preguntas distintas, con fuentes y evaluaciones distintas. Lo que encuentren se
guarda en el expediente de la señal, no en el RAG (PubMed se consulta en vivo; los posts son
datos de salud y no van a ningún índice).

### Evaluación de `buscar_literatura`

[`bench/literatura.jsonl`](bench/literatura.jsonl) (desarrollo): 12 preguntas con respuesta en
la literatura (hiponatremia con sertralina, síndrome serotoninérgico con tramadol…) y 2
controles sin ella. [`bench/literatura_test.jsonl`](bench/literatura_test.jsonl) (examen): 10
preguntas y 2 controles distintos, escritos antes del primer cambio; solo para la nota final.
Cada pregunta lleva términos en inglés y 3 PMID de referencia comprobados el 2026-11-23.

| Medida | Qué es |
|---|---|
| **PMID sin respaldo** | Citados en la respuesta sin que ninguna búsqueda los devolviera. Lo más grave |
| Relevancia | Título o resumen con un término del fármaco y uno de la reacción (aproximada) |
| Referencia encontrados | De los 3 PMID por pregunta. Hay otros válidos: para comparar sistemas |
| Controles | Si admite que no hay evidencia |
| Coste | Tokens, pasos, segundos, búsquedas |

Sistema de comparación (`fija`): una búsqueda en PubMed con una consulta fija, los 10 primeros.

**Desarrollo (2026-11-23, 27B):**

| | Búsqueda fija | Bluesky + PubMed (v1) | `buscar_literatura` (v2) |
|---|---|---|---|
| PMID citados sin respaldo | — | 0 de 81 | **0 de 129** [0-3 %] |
| Relevantes (resúmenes cortados) | 77 % de lo entregado | 93 % de lo citado | **94 % de lo citado** [89-97] (78 % de todo lo visto) |
| De referencia encontrados | 15-16 de 36 | 6 de 36 | **15 de 36** (14 citados) |
| Controles | — | "no hay evidencia" | "no hay evidencia" |
| Coste por pregunta | 0 tokens | 63 000 tokens, 13,6 pasos | **11 600 tokens, 4 pasos, 24 s** |

Separar PubMed devolvió al agente lo que perdía frente a la búsqueda fija (encuentra lo mismo:
15 de 36) con una quinta parte del coste, y no inventa referencias.

**Corrección (2026-11-23).** Hasta aquí los resúmenes se cortaban a 900 caracteres, para el agente
y para la medida de relevancia. Leyendo respuestas a mano: el agente citó una revisión
(PMID 1684924) para decir que en ancianos el lorazepam se elimina peor, y el resumen completo dice
lo contrario (es la excepción); esa frase estaba después del corte. Con resúmenes completos
(hasta 2500 caracteres) la búsqueda fija pasa del 77 % al **93 % de relevantes** en desarrollo
(85 % en el examen): **la ventaja de "filtrado" del agente era sobre todo un efecto del corte**.
Lo que el agente añade es el resumen (tipo de evidencia, grupos de riesgo, mecanismo), y eso hay
que medirlo como **fidelidad**: cada afirmación con PMID, ¿está en ese artículo? v3: resúmenes
completos y la regla en el prompt de no atribuir a un artículo lo que no dice.

v3 en desarrollo: 0 de 113 PMID sin respaldo, 99 % de lo citado relevante (104 de 105), 15 de 36
de referencia, 13 400 tokens por pregunta. Revisadas a mano todas las cifras de la respuesta del
lorazepam: todas están en sus artículos, y ahora explica la excepción del aclaramiento y marca el
mecanismo general como "sin cita". Un fallo: en caídas con zolpidem devolvió una respuesta vacía
tras ver 17 artículos (la respuesta no cabía en 2048 tokens de salida). v3.1: 4096 tokens en
modo literatura, y una respuesta final vacía vuelve al modelo como error en vez de terminar.

**Examen (`literatura_test`, 2026-11-23, v3.1, una sola ejecución):**

| | Búsqueda fija | `buscar_literatura` |
|---|---|---|
| PMID citados sin respaldo | — | **0 de 86** [0-4 %] |
| Relevantes | 88 % de lo entregado [78-93] | **95 % de lo citado** [88-98] (86 % de lo visto) |
| De referencia encontrados | 14 de 30 | 13 de 30 (11 citados) |
| Controles | — | los dos: "no hay evidencia"; lo que cita lo presenta como otra cosa |
| Coste por pregunta | 0 tokens | 15 200 tokens, 4,2 pasos, 25 s, 3,2 búsquedas |

La primera puntuación dio 2 PMID sin respaldo: eran **años** ("PMID 27722027, 2016"), que la
comprobación leía como una lista de PMID. Corregida la comprobación (`citas.py`), las mismas
respuestas se puntuaron otra vez sin ejecutar el agente (`--recalcular`).

**Fidelidad, a mano:** en tres respuestas del examen (hipoglucemia con tramadol, levotiroxina y
hueso, lorazepam con opioides) se buscó cada cifra y dato atribuido a un PMID en su resumen
original: **26 artículos, todo lo comprobado está en su artículo** (p. ej. 97 639 notificaciones y
ROR 1,54; OR 1,52 y 2,61; 63 estudios, 31/23/9; HR 1,83 y 2,96). Además separa bien lo que es
del grupo y no del fármaco ("no encontré estudios que aíslen el lorazepam; extrapolo a su clase")
y marca lo que no tiene cita. Límites: 3 respuestas de 10, revisadas por Claude, no por un
farmacólogo, y con la parte de mecanismo comprobada solo en sus cifras.

**Conclusión:** encuentra lo mismo que una búsqueda simple, no inventa referencias y lo que añade
es un resumen fiel y ordenado (evidencia, grupos de riesgo, mecanismo) por ~15 000 tokens.

### Evaluación de `buscar_mas_casos`

Sobre la **muestra congelada de sertralina**, con la **hoja de soluciones del triaje** como
respuestas: de las 725 menciones que el híbrido envió al triaje, los posts anotados como efecto
adverso cuya reacción encaja con la pregunta (expresión regular revisada a mano,
`eval_casos construir --ver`). [`bench/casos.jsonl`](bench/casos.jsonl) (desarrollo): calor,
insomnio, bruxismo, piernas inquietas y mareo (58 casos) y un control (caída del pelo, 0).
[`bench/casos_test.jsonl`](bench/casos_test.jsonl) (examen): problemas sexuales, temblores,
estómago, embotamiento emocional y somnolencia (44 casos) y un control (sangrado, 0).

- **Recall**: casos encontrados / casos de la hoja. **Precisión**: de lo guardado que la hoja
  cubre, cuántos son casos. Lo guardado fuera de las 725 se cuenta aparte (sin juzgar).
- Sistema de comparación (`fija`, sin modelo): las menciones ya recogidas por el rastreador cuyo
  texto contiene una palabra de la reacción. Desarrollo: **recall 84 % [73-92], precisión 50 %
  [41-60]**; examen: 84 % [71-92] y 37 % [28-46]. Encuentra casi todo, pero la mitad o más son
  bromas, preguntas u otras reacciones: el agente tiene que ganar en precisión sin perder recall.

```bash
uv run python -m crawler.eval_casos construir --ver      # donde está la hoja del triaje → data/casos/
uv run python -m crawler.eval_casos evaluar --sistemas fija
ARCA_CRAWLER_DATA=data/bluesky_sertralina uv run --extra crawler python -m crawler.eval_casos evaluar --sistemas agente
```

Cifras (sin texto) en `reports/crawler_casos_eval[_casos_test].json`; posts guardados, perdidos y
trazas en `data/bluesky/casos/<banco>_<cerebro>/`.

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
