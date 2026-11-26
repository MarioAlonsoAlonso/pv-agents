# Coordinador · el 27B con todo el sistema como herramientas

**Qué hace:** atiende al técnico de farmacovigilancia. Recibe una petición en lenguaje natural
("¿qué casos de sertralina debería revisar primero?", "prepárame el resumen para el comité"),
decide qué agentes y componentes necesita, los llama y redacta la respuesta. Si hay que
preparar un informe oficial (ICSR), lo deja **pendiente de confirmación humana**: nunca lo
registra solo.

```
                 Técnico de farmacovigilancia
                              │
              COORDINADOR · Qwen3.8-27B · bucle ReAct
   ┌───────────┬────────────┬───────────┬────────────┬────────────┬─────────────┐
 listar_casos codificar_   search_     faers_       buscar_      create_icsr_
 ver_caso     meddra       knowledge_  reaction_    literatura   report
              agrupar_por_ base        counts +                  ⏸ confirmación
              meddra                   PRR                       humana
   │            │            │           │            │
 rastreador   modelo de    RAG: fichas  FAERS       sub-agente del
 + triaje     razonamiento técnicas de  (openFDA)   rastreador
 (casos ya    MedDRA 1,7B  CIMA y guías             (PubMed)
 filtrados)   (fase 1)     de la EMA
```

## Diseño

**Lo que se puede hacer con código se hace antes** (la lección del rastreador y del triaje):
rastrear Bluesky y decidir qué posts son efectos adversos lo hacen el rastreador y el triaje
sobre cientos de posts, sin el coordinador. `coordinator/cases.py` junta sus salidas, se queda
con los efectos adversos y los **ordena por prioridad**: graves primero (plazo legal de 15 días
en lugar de 90), luego los más completos (sexo y desenlace conocidos, más cerca de un ICSR
válido), luego los más recientes. El 27B coordina sobre casos, no sobre posts.

**El bucle no se reescribe:** es el de [`agent/react_agent.py`](../agent/react_agent.py) con las
llamadas nativas a herramientas del 27B que añadió el rastreador
([`crawler/agent.py`](../crawler/agent.py)): una acción por paso, tiempos máximos, errores como
observación, detección de bucles, memoria y confirmación humana de las acciones con efecto.

**Herramientas** ([`tools.py`](tools.py)):

| Herramienta | Qué hay detrás | Para qué |
|---|---|---|
| `listar_casos`, `ver_caso` | Rastreador + triaje | Qué se ha encontrado, por prioridad |
| `codificar_meddra` | **Modelo de razonamiento** de la fase 1 (Qwen3-1.7B + LoRA, SFT + GRPO) | Reacciones → *Preferred Terms*; se guardan para no codificar dos veces |
| `agrupar_por_meddra` | Código | Casos equivalentes: cuántos comparten cada término |
| `search_knowledge_base` | **RAG** de la fase 3 (BM25 + denso sobre fichas de CIMA y guías de la EMA) | ¿Es conocida la reacción? ¿Con qué frecuencia? ¿Qué plazo marca la norma? |
| `faers_reaction_counts`, `disproportionality_prr` | Herramientas de la fase 2 | Notificaciones en FAERS y señal (PRR, regla de Evans) |
| `buscar_literatura` | Sub-agente del rastreador (PubMed) | ¿Está descrita la reacción? ¿Qué evidencia hay? |
| `estado_del_sistema` | Las etapas ([`etapas.py`](etapas.py)) | Qué se ha rastreado y triado de cada fármaco, y qué falta |
| `listar_menciones` | Rastreador | Lo que encontró, que es lo que recibe el triaje |
| `listar_descartes` | Triaje | Lo que dijo que no es un efecto adverso, y por qué |
| `rastrear`, `triar` | Rastreador (vigilante) y triaje | Lanzarlos desde la conversación; los trabajos grandes esperan la confirmación |
| `create_icsr_report` | Herramienta de la fase 2, con `requires_confirmation` | Borrador del ICSR, pendiente de confirmar. Antes de dejarlo pendiente, el código comprueba que sería un ICSR válido (paciente identificable, términos del diccionario): un borrador que nunca se podría registrar no llega a la cola |

**Fuera a propósito: Naranjo.** Sus diez preguntas (reexposición, placebo, niveles en sangre…)
no se pueden contestar desde un post y el modelo las rellenaría. La causalidad aquí es lo que
pedía la propuesta a ese agente: si hay casos equivalentes, si la reacción es conocida y si hay
señal.

## Cómo se usa (en la DGX)

Hace falta una sesión **con GPU** (el modelo MedDRA y el buscador del RAG corren en local) y
el 27B (clave en `.env`). La primera vez:

```bash
uv sync --extra train --extra rag --extra agent --extra crawler   # = make setup
make rag-corpus                    # corpus del RAG: fichas de CIMA y guías de la EMA (~10 min)
```

`.env` tiene que tener `ARCA_RLM_ADAPTER=Matteofermar/arca-meddra-qwen3-1.7b` (viene en
`.env.example`): es el modelo de razonamiento, que se descarga del Hub.

Los casos salen de `data/`: las menciones del rastreador y las respuestas del triaje
(`data/triaje/eval/v2_27b_mencion_{dev,test}_respuestas.jsonl`).

```bash
uv run python -m coordinator.demo --casos                                   # resumen de los casos
uv run python -m coordinator.demo "¿Qué casos de sertralina debería revisar primero?"
uv run python -m coordinator.demo "Prepara el resumen para el comité sobre la sertralina"
uv run python -m coordinator.evaluate                                       # banco + ablaciones
```

La demo imprime la traza (pensamiento, acción, observación real) y la guarda en
`data/coordinador/demo/`.

## Evaluación

**Banco** ([`bench/tareas.jsonl`](bench/tareas.jsonl)): 18 peticiones con comprobación
automática, de dos tipos:

- **13 dirigidas** (`c01`–`c13`): la petición dice qué hacer («codifica…», «calcula el PRR…»).
  Miden si el coordinador ejecuta bien las herramientas.
- **5 abiertas** (`a01`–`a05`): lo que diría de verdad un técnico, sin decir cómo («Buenos días,
  ¿qué tengo pendiente hoy con la sertralina?», «¿Hay algo que tenga que notificar ya?», «Lo de
  la manía, ¿es nuevo o ya se conocía?», «Prepara lo que haya que mandar a la AEMPS», «¿Estamos
  viendo algo raro?»). Miden si sabe **decidir** los pasos: por ejemplo, que «notificar ya»
  significa buscar los casos graves y el plazo de las guías de la EMA, o que los dos casos
  graves no pueden ir a un ICSR porque no tienen ni edad ni sexo.

Las dirigidas: Cubre casos y prioridad, codificación MedDRA, ficha técnica (con cita
del fragmento), señal en FAERS (PRR calculado en vivo), normativa, ICSR pendiente de
confirmación, evidencia en redes y PubMed, un resumen completo para el comité y dos peticiones
imposibles (otra red social y otro año; un fármaco sin casos), donde tiene que decir que no lo
sabe. Las peticiones nombran casos con huecos (`{caso_prioritario}`…) que se rellenan con los
datos, así que el banco no contiene ningún post.

**Métricas:** éxito, uso de las herramientas necesarias, pasos, tokens, tiempo, **números sin
respaldo** (cifras de la respuesta que no aparecen en ninguna observación: recuentos, PRR o
frecuencias inventados) y **citas sin respaldo** (ids de fragmentos de la ficha técnica que la
respuesta cita pero que ninguna búsqueda devolvió).

**Ablaciones** (`--variantes`): `completo`, `sin_rag`, `sin_meddra` (el 27B tiene que poner los
términos MedDRA él solo) y `sin_evidencia`. Responden a "¿aporta cada pieza?".

Las cifras van a `reports/coordinator_eval.json`; las respuestas y las trazas, a
`data/coordinador/eval/` (pueden parafrasear posts: fuera de git). Si se corta la sesión,
relanzar el mismo comando sigue donde se quedó.

## Resultados

**Primera prueba con el 27B** (2026-11-23): «¿Hay casos graves de sertralina? Codifica en MedDRA
el más urgente y dime si esa reacción aparece en la ficha técnica y si hay señal en FAERS.»

7 pasos, 32 000 tokens: `listar_casos` (graves) → `ver_caso` → `codificar_meddra` (Insomnia,
Mania) → `search_knowledge_base` (manía: frecuente, y advertencia de la sección 4.4) →
`faers_reaction_counts` (Mania) → `disproportionality_prr` (PRR 5,85, IC 95 % 5,49–6,25: señal)
→ respuesta. Todas las cifras salen de las observaciones y las tres citas de la ficha técnica
están en lo que devolvió el RAG. Ofrece preparar el ICSR en lugar de hacerlo sin pedírselo.

En el primer intento el coordinador solo consultó el insomnio: la herramienta daba «Mania» por
no reconocida porque no está en el subconjunto de 90 PT del repositorio, y el 27B lo leyó como
que el modelo de razonamiento la había descartado. Desde entonces la herramienta explica que
esos términos son válidos para FAERS.

**Banco completo, variante `completo`** (2026-11-23, 27B, 159 casos de 725 menciones):

| Éxito | Herramientas necesarias | Con números sin respaldo | Con citas sin respaldo | Pasos | Tokens/tarea | Tiempo |
|---|---|---|---|---|---|---|
| **13/13** | 12/13 | 0 | 0 | 2–17 (media 4,2) | 24 700 | 3 s – 13 min |

- Las dos peticiones imposibles (otra red social y otro año; un fármaco sin casos) las contesta
  diciendo qué no cubre el sistema, sin inventar.
- El ICSR queda pendiente de confirmación. La primera llamada falló la validación (puso la edad
  como el texto `"None"`) y el 27B la corrigió solo con el mensaje de error.
- La única petición sin todas las herramientas previstas es el ICSR: el 27B puso él mismo el
  término MedDRA (Insomnia, el correcto) en lugar de llamar al modelo de razonamiento.
- El coste está en el modelo de razonamiento: codificar un caso tarda ~2,5 minutos (Qwen3-1.7B
  con `generate` de Hugging Face en una GPU de 16 GB), así que las dos peticiones que codifican
  cinco casos tardan ~13 minutos; el resto, segundos.

**Fallos encontrados al revisar las trazas, y arreglados:**
1. El modelo de razonamiento a veces no termina su respuesta (razona hasta el límite de tokens;
   en la fase 1 pasaba en 19 de 200). La herramienta guardaba ese razonamiento como si fueran
   términos (`"<think>"`, `"Okay, let's tackle…"`). El 27B lo detectó y lo descartó, pero quedaba
   en la caché. Ahora se reintenta una vez y, si no, la herramienta devuelve un error y no
   guarda nada; la caché descarta entradas corruptas al cargar.
2. El detector de números sin respaldo daba tres falsos positivos («IC95 5,49» leído como 95 549)
   y no aceptaba restas evidentes («el resto, 157» = 159 − 2). Corregido y vuelto a puntuar sin
   llamar al modelo (`--solo-puntuar`).

**Ablaciones** (18 peticiones: 13 dirigidas y 5 abiertas; una ejecución por variante):

| Variante | Dirigidas | Abiertas | Total | Números / citas sin respaldo | Tokens/petición | c07 (codifica 5 casos) |
|---|---|---|---|---|---|---|
| **Completo** | 13/13 | 4/5 | **17/18** | 0 / 0 | 26 800 | 13 min |
| Sin RAG | 10/13 | 3/5 | **13/18** | 0 / 0 | 23 800 | 7 min |
| Sin modelo MedDRA | 13/13 | 5/5 | **18/18** | 0 / 0 | 25 100 | 16 s |
| Sin sub-agente de evidencia | 11/13 | 5/5 | 16/18 | 0 / 0 | 28 300 | 22 s |

- **El RAG es lo que más aporta** (−4). Sin él, el coordinador o se niega («no dispongo de la
  herramienta… prefiero no inventar») o contesta de memoria con errores de concepto: da bien el
  plazo de 15 días pero lo llama «SUSAR» (un término de ensayos clínicos), o concluye que la
  manía «ya se conocía» a partir de la señal de FAERS, que no dice nada de la ficha técnica.
- **Sin el sub-agente** falla lo que lo necesita (c12). El otro fallo (c06) no es de la ablación:
  openFDA devolvió la mitad de los recuentos de sertralina (111 098 en lugar de 222 019) y el
  PRR salió 6,73 en lugar de 2,66. La comprobación lo detectó; el coordinador se lo creyó.
- **Sin el modelo de razonamiento MedDRA no se pierde nada en el banco** y es unas 50 veces más
  rápido: el 27B codifica él mismo y, en estos casos, de forma parecida («Mania», «Insomnia»,
  «Suicidal ideation»). Ninguno de los dos acierta siempre el *Preferred Term* exacto (el 27B
  escribe «Decreased libido» y el modelo pequeño «Lack of sexual desire»; el PT es «Libido
  decreased»). El banco no demuestra que el modelo entrenado aporte calidad frente al 27B; su
  ventaja es otra: es pequeño, corre en local y su acierto está medido con un verificador (56,5 %
  en el test de la fase 1).
- **Peticiones abiertas:** con todo activado acierta 4 de 5. La que falla («¿Estamos viendo algo
  raro?») la contesta bien pero sin investigar: lista los casos y mira FAERS, sin ficha técnica
  ni señal. En las tres ablaciones la misma petición sí pasa, con más pasos: con una ejecución
  por variante, una petición de diferencia es ruido.

**Cambios en las comprobaciones tras ver las trazas** (para leer las cifras con cuidado): c12
aceptaba una negativa que nombraba «PubMed» (ahora exige un PMID o un id de post); c09 aceptaba
solo el ICSR pendiente, y en dos ejecuciones el 27B se negó con un buen motivo (el post nombra
tres antidepresivos: no hay un sospechoso claro), así que ahora acepta las dos cosas; y la
primera ablación sin MedDRA no valía porque `ver_caso` enseñaba la codificación guardada.

**Banco v2** (105 peticiones con fuentes esperadas; ver «Evaluación»). Desarrollo (46): línea
base 42/46 → 43/46 tras las mejoras (búsqueda de casos por reacción y filtros, `listar_casos` sin
fármaco, `codificar_notificacion`), con las fuentes necesarias consultadas en el 100 % de las
peticiones (antes 89 %) y fuentes de más en el 15 % (antes 28 %).

**Examen** (59 peticiones, medido una vez con el sistema cerrado en desarrollo, 2026-11-23):

| Variante | Aciertos | Ficha | Normativa | MedDRA | Literatura | Fuentes necesarias | Números sin respaldo | s/petición |
|---|---|---|---|---|---|---|---|---|
| **Completo** | **54/59 (92 %)** | 13/14 | 4/4 | 5/6 | 4/4 | 98 % | 1 | 44 |
| Sin RAG | 30/59 (51 %) | 0/14 | 0/4 | 3/6 | 4/4 | 58 % | 6 | 34 |
| Sin modelo MedDRA | 52/59 (88 %) | 13/14 | 4/4 | 3/6 | 4/4 | 88 % | 3 | 14 |
| Sin literatura | 51/59 (86 %) | 13/14 | 4/4 | 4/6 | 2/4 | 95 % | 1 | 55 |

McNemar de la variante completa frente a cada ablación: sin RAG, 24 peticiones solo las acierta
la completa y 0 al revés (p < 0,0001); sin MedDRA, 4 frente a 2 (p = 0,69); sin literatura, 5
frente a 2 (p = 0,45). **El RAG es imprescindible**: sin él falla toda la ficha técnica y la
normativa y aparecen cifras sin respaldo en 6 respuestas. El modelo MedDRA y la literatura
aportan justo en sus categorías (MedDRA 5/6 frente a 3/6; literatura 4/4 frente a 2/4), pero
con tan pocas peticiones de cada una la diferencia no es significativa. Sin el modelo MedDRA el
coordinador es tres veces más rápido.

**Corrección de la medida después del examen** (las mismas para las cuatro variantes; las
respuestas se leyeron a mano): dos comprobaciones demasiado literales rechazaban respuestas
correctas (f14 decía «contraindicación expresa» y se pedía «contraindicad»; g02 decía «resumen
de características del producto» y se pedía «resumen de las características»), y el detector de
números juntaba los de una lista JSON («14.979, 16.234») y marcaba como inventado un intervalo
copiado de la herramienta. Sin corregir, la variante completa daba 52/59 (88 %).

**Fallos del coordinador en el examen** (variante completa):
- **x05, grave:** «¿Hay señal de hepatotoxicidad con la xelofrina?» (no existe). FAERS no la
  conoce; el coordinador decidió que era el celecoxib y respondió «sí, hay señal» con el PRR del
  celecoxib. Igual en las cuatro variantes: sustituye un nombre desconocido por uno parecido sin
  base. Haría falta una garantía en código (no aceptar un fármaco distinto del pedido sin
  confirmación del técnico).
- **f07:** la sobredosis de ciprofloxacino la buscó en nuestros casos y no en la ficha técnica.
- **k09 y o04:** sin filtro de edad conocida, y sin forma de ver las reacciones de los 159 casos
  a la vez, recorrió los casos por fechas hasta agotar los 20 pasos.
- **m04:** «Lack of appetite» en lugar del PT «Decreased appetite» (el sinónimo frente al término
  preferente; un MedDRA completo con licencia lo traduciría).
## Todas las etapas, desde la conversación (2026-11-26)

Hasta aquí el coordinador solo veía los casos: no podía decir qué encontró el rastreador, qué
descartó el triaje ni que un fármaco no se había rastreado nunca (respondía «no hay casos»).
Ahora se le puede preguntar por cualquier etapa, y lanzar las dos primeras:

| El técnico pide… | Herramienta |
|---|---|
| Lo que encontró el rastreador (= lo que recibe el triaje) | `listar_menciones`; una entera, con su conversación, `ver_mencion` |
| Lo que descartó el triaje y por qué | `listar_descartes` |
| En qué punto está cada etapa y qué falta | `estado_del_sistema` |
| Los casos, MedDRA, ficha, FAERS, literatura, ICSR | lo de antes |
| Algo que necesita una etapa que no se ha hecho («casos de tramadol») | `estado_del_sistema` → explica qué falta y ofrece `rastrear` / `triar` |
| «Busca todo lo del lorazepam desde enero», «solo septiembre» | `rastrear` (desde y, si hace falta, hasta), después `triar` |

**El almacén.** [`cases.py`](cases.py) lee la muestra congelada de la evaluación y, además, el
almacén del vigilante (`data/bluesky/monitor/menciones.jsonl` y `triaje_respuestas.jsonl`), y
guarda todas las menciones y todas las respuestas del triaje, no solo los casos. Una mención que
esté en los dos sitios cuenta una vez. Los bancos usan solo la muestra (`include_monitor=False`),
para que sus cifras no cambien.

**Lanzar etapas: lo que se cuida** ([`etapas.py`](etapas.py)):

- **Coste y permiso.** El 27B es el servidor compartido del curso. Con lo medido en sertralina (una
  mención al día, ~3 100 tokens por mención en el rastreo con agente y ~2 400 en el triaje), un
  rastreo de más de **31 días** o un triaje de más de **100 menciones** no lo lanza el modelo: queda
  pendiente, con su estimación, hasta que el técnico dice «sí» (el mismo mecanismo que el ICSR).
  Los trabajos pequeños se hacen al momento.
- **Tiempo.** Lo confirmado lo ejecuta el código fuera del bucle del modelo, sin su tiempo
  máximo (meses de Bluesky pueden ser una hora); el resultado vuelve al coordinador como el turno
  siguiente.
- **Un solo escritor.** El vigilante periódico, un rastreo pedido en la conversación y un triaje
  usan el mismo bloqueo (`en_marcha.lock`): el segundo se para con un error en vez de escribir a
  la vez los mismos ficheros.
- **Solo los fármacos vigilados**, y fechas desde que Bluesky tiene posts públicos.
- **En los exámenes nunca se lanza nada** (`hold_stages`): `rastrear` y `triar` solo se proponen.

**Conversación.** `Conversacion` ([`agent.py`](agent.py)) mantiene la memoria entre turnos y
confirma o rechaza lo pendiente:

```bash
uv run --extra crawler python -m coordinator.chat          # make coordinador-chat
técnico> ¿Hay casos graves de tramadol?
  → nunca se ha rastreado; ¿busco en Bluesky? ¿desde cuándo?
técnico> Sí, desde enero de 2026
  ⏸ [1] Rastrear tramadol en Bluesky desde 2026-01-01. Rastrear 329 días: unas 345 menciones…
técnico> sí
```

**Primera prueba con el 27B (2026-11-26, DGX, sin datos previos).** «Busca información de la
sertralina en septiembre de 2026»: propuso el rastreo con su coste, esperó el «sí», encontró 128
menciones (81 de la búsqueda fija, 47 del agente) y propuso el triaje, otra vez con confirmación
(128 > 100). Tres fallos, corregidos: `rastrear` no tenía fecha final (para «solo septiembre»
rastreó hasta hoy, 86 días en vez de 30); el técnico no podía leer un post entero (no había
herramienta para menciones que no son casos, y el prompt le hacía resumir siempre); y la
estimación usaba el ritmo de la muestra congelada (90 previstas, 128 reales: ahora 1,5 al día).

**Evaluación:** banco v5 ([`bench/v5.py`](bench/v5.py), 24 peticiones: menciones, descartes,
estado, etapa que falta y lanzar), `make coordinador-eval-etapas` (completo frente a
`sin_etapas`). Pendiente de ejecutar en la DGX.

## Piezas

```
coordinator/
  cases.py      casos = menciones del rastreador + respuestas del triaje, por prioridad;
                guarda también las menciones y los descartes (todas las etapas)
  etapas.py     qué ha hecho cada etapa, coste de lanzarlas y cuándo piden permiso
  tools.py      las herramientas: cada agente del sistema
  prompts.py    el prompt del coordinador
  agent.py      el bucle (el del rastreador con otras herramientas y tiempos)
  evaluate.py   el banco, las comprobaciones y las ablaciones
  demo.py       probarlo a mano y ver la traza (una pregunta)
  chat.py       conversación en la terminal, con «sí» / «no» a lo pendiente
  bench/        los bancos (v1–v5)
```

Tests: [`tests/test_coordinator.py`](../tests/test_coordinator.py) y
[`tests/test_etapas.py`](../tests/test_etapas.py), con un cerebro, un Bluesky y un triaje
simulados: sin red, GPU ni `data/`.
