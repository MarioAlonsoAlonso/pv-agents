# Triaje · ¿es un efecto adverso?

**Qué hace:** recibe las menciones del rastreador (`RawMention`, contrato en
[`crawler/CONTRATO_TRIAJE.md`](../crawler/CONTRATO_TRIAJE.md)) y, para cada una, decide si el
post describe un **efecto adverso del fármaco**. Si lo describe, extrae los datos del caso:
reacciones con las palabras del autor, paciente, edad, sexo, gravedad y desenlace. Es lo que
recibe después el codificador MedDRA.

Las respuestas usan exactamente los campos y valores de la hoja de soluciones
([`evaluacion_triaje/ANOTACION_TRIAJE.md`](../evaluacion_triaje/ANOTACION_TRIAJE.md)):

```json
{"id": "M-56a92a5028", "es_efecto_adverso": "si", "motivo_no": "",
 "reacciones": "me paso el día con náuseas", "negadas": "", "paciente": "autor",
 "edad": "desconocido", "sexo": "desconocido", "sexo_fuente": "", "grave": "no_consta",
 "criterio_gravedad": "", "desenlace": "desconocido", "notas": "", "estado": "ok"}
```

## Diseño

Como en el rastreador, **lo que se puede garantizar con código no se le pide al modelo**:

| Paso | Quién | Por qué |
|---|---|---|
| Agrupar las menciones por conversación (en trozos de 8) | Código | Las respuestas de un hilo se entienden juntas; una petición por hilo, no por post |
| Leer y decidir | 27B | Es la parte que necesita criterio: bromas, respuestas indirectas, otros fármacos |
| Quedarse solo con los ids enviados y los valores exactos de la hoja | Código | Un id inventado se ignora; `true` → `si`, criterio sin `grave = si` → se limpia |
| Si falta una respuesta o no es válida, volver a preguntar **una vez** diciendo qué falló | Código | Una lista cortada o un valor mal escrito no deberían perder la mención |
| Si aun así no hay respuesta válida: **a revisión humana como posible efecto adverso** (`estado = "sin_respuesta"`) | Código | En farmacovigilancia perder un caso es peor que revisar uno de más |

No es un bucle ReAct con herramientas: el triaje no necesita buscar nada fuera del post y su
conversación, así que una petición estructurada por conversación basta. Si las pruebas lo
pidieran (por ejemplo, consultar la ficha técnica en los casos dudosos), el bucle de
[`agent/react_agent.py`](../agent/react_agent.py) ya existe.

**Cerebros** (`--brain`):

| Cerebro | Qué es |
|---|---|
| `palabras` | Línea base **sin modelo**: "sí" si el post contiene un síntoma de una lista fija. Mide si hace falta un LLM |
| `27b` | Qwen3.8-27B-FP8 del servidor del curso (solo desde la DGX) |
| `27b-thinking` | El mismo, con razonamiento |

**Prompts** (`--version`, en [`prompts.py`](prompts.py)):

| Versión | Qué cambia |
|---|---|
| `v0` | El prompt con el que el 27B pre-anotó la hoja de soluciones, tal cual. **Ojo:** la hoja se construyó revisando los desacuerdos entre Claude y el 27B con este prompt, así que está algo favorecido |
| `v1` | v0 + reglas para las trampas conocidas (broma con síntoma real, otro fármaco, síntomas de un mensaje anterior, retirada, falta de eficacia) + ejemplos inventados |
| `v2` | v1 + los casos que v1 perdía: pregunta que cuenta un síntoma propio, queja vaga atribuida al fármaco, otro fármaco en el post pero algo atribuido a este, tomas olvidadas; «ante la duda, a revisión»; no suponer «hombre» por defecto |

`--por mencion` hace una petición por post en lugar de por conversación (más caro; sirve para
medir si agrupar contagia el tema del hilo a las respuestas, como le pasaba al 27B al anotar).

## Cómo se usa

En tu ordenador (sin modelo), para comprobar que todo funciona:

```bash
make triage-base        # = uv run python -m triage.run --brain palabras
```

En la DGX (los datos de `data/` y la clave en `.env`, ver [`COLABORAR.md`](../COLABORAR.md)):

```bash
uv run python scripts/check_remote_llm.py                        # ¿responde el 27B?
uv run python -m triage.run --version v1 --limite 5              # prueba con 5 conversaciones
uv run python -m triage.run --version v1                         # desarrollo: 287 menciones
uv run python -m triage.run --version v0                         # la alternativa, para comparar
uv run python -m triage.run --version v1 --brain 27b-thinking    # con razonamiento
uv run python -m triage.evaluate comparar \
    data/triaje/eval/v0_27b_conversacion_dev_respuestas.jsonl \
    data/triaje/eval/v1_27b_conversacion_dev_respuestas.jsonl    # McNemar entre dos versiones
```

Si la sesión se corta, relanza el mismo comando y **sigue donde se quedó**. `--de-nuevo` empieza
desde cero.

Cada ejecución deja:

| Fichero | Qué tiene | ¿Git? |
|---|---|---|
| `reports/triage_eval.json` | Las cifras (detección, precisión, datos del caso, coste), sin texto | Sí |
| `data/triaje/eval/<nombre>_respuestas.jsonl` | Una respuesta por mención | **No** |
| `data/triaje/eval/<nombre>_errores.json` | Los fallos con su texto: `perdidos`, `falsas_alarmas`, `datos_del_caso`, `motivo_no` y las `reacciones` lado a lado | **No** |

**El examen**, una sola vez y al final, con la versión elegida en desarrollo:
`uv run python -m triage.run --version vN --examen`. Después de mirarlo no se cambia nada
([`GUIA_TRIAJE.md`](../evaluacion_triaje/GUIA_TRIAJE.md), apartado 5).

## Qué se mide

Lo de la [guía](../evaluacion_triaje/GUIA_TRIAJE.md), apartado 6, con intervalos de Wilson al
95 %:

- **Detección** (*recall*): de los efectos adversos de la hoja, cuántos marca `si`.
- **Precisión**: de lo que marca `si`, cuánto lo es (falsas alarmas = trabajo para el revisor).
- **Descarte de errores del rastreador**: de los `no_es_mencion`, cuántos marca `no`.
- **Datos del caso**, en los efectos adversos que ambos marcan: `paciente`, `edad`, `sexo`,
  `grave` y `desenlace`.
- Todo lo anterior **por origen**: menciones de la búsqueda fija y del agente del rastreador
  (las del agente son más difíciles: respuestas que no nombran el fármaco).
- **Coste**: tokens, peticiones, reparaciones y segundos.

En desarrollo hay 287 menciones y 72 efectos adversos. **Ninguno es grave y en todos el
paciente es el autor**, así que esos dos campos no se pueden afinar en desarrollo.

## Resultados en desarrollo

| Versión | Cerebro | Detección | Precisión | Descarta `no_es_mencion` | Tokens |
|---|---|---|---|---|---|
| `palabras` | — | 36 % [26–48] | 57 % [42–70] | 88 % [77–94] | 0 |
| `v0` | 27b | **82 % [72–89]** | 75 % [64–83] | 84 % [73–92] | 594 / mención |
| `v1` | 27b | 75 % [64–84] | **86 % [75–92]** | **95 % [86–98]** | 1 176 / mención |
| `v2` | 27b | 88 % [78–93] | 88 % [78–93] | 93 % [84–97] | 1 372 / mención |
| `v2` (repetición) | 27b | 89 % [80–94] | 86 % [77–92] | 93 % [84–97] | 1 399 / mención |
| `v2` | 27b-thinking | 92 %\* | 61 %\* | 71 %\* | 2 632 / mención |
| **`v2`, `--por mencion`** | **27b** | **90 % [81–95]** | **92 % [83–96]** | **98 % [91–100]** | 2 384 / mención |

\* Con razonamiento, 53 respuestas se cortaron por el límite de tokens antes del JSON y fueron
a revisión humana (cuentan como «sí»). En las 234 que contestó acierta lo mismo que sin
razonamiento (221), con el doble de tokens y 10 veces más tiempo (65 min frente a 7).

**Configuración elegida para el examen: `v2`, `--por mencion`, `27b` sin razonamiento.**

v2 frente a v0 en el acierto sí/no: 269 frente a 254 de 287 (21 casos solo los acierta v2, 6
solo v0), McNemar p = 0,006. v0 frente a v1: 254 frente a 260, p = 0,34 (no significativo). v1 es más estricta: la mitad de falsas alarmas, pero pierde más casos. En v1,
5 de los 75 % fueron a revisión humana solo porque el modelo dejó vacía la reacción (desde v2 el
código acepta el «sí» con `[no especificada]`).

## Piezas

```
triage/
  prompts.py    los prompts (v0, v1) y cómo se presenta cada conversación
  agent.py      agrupar, preguntar, validar, reparar, revisión humana; y la línea base
  evaluate.py   cifras con Wilson, fallos con texto, comparación con McNemar
  run.py        la línea de comandos: ejecutar sobre desarrollo o examen y puntuar
```

Tests: [`tests/test_triage.py`](../tests/test_triage.py), con un modelo simulado: no necesitan
red ni `data/`.
