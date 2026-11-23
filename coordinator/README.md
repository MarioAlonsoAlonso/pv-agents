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
 ver_caso     meddra       knowledge_  reaction_    evidencia    report
              agrupar_por_ base        counts +                  ⏸ confirmación
              meddra                   PRR                       humana
   │            │            │           │            │
 rastreador   modelo de    RAG: fichas  FAERS       sub-agente del
 + triaje     razonamiento técnicas de  (openFDA)   rastreador
 (casos ya    MedDRA 1,7B  CIMA y guías             (Bluesky + PubMed)
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
| `buscar_evidencia` | Sub-agente dirigido del rastreador | Más casos en Bluesky y PubMed |
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

Ablaciones (`sin_rag`, `sin_meddra`, `sin_evidencia`): pendiente (DGX).

## Piezas

```
coordinator/
  cases.py      casos = menciones del rastreador + respuestas del triaje, por prioridad
  tools.py      las herramientas: cada agente del sistema
  prompts.py    el prompt del coordinador
  agent.py      el bucle (el del rastreador con otras herramientas y tiempos)
  evaluate.py   el banco, las comprobaciones y las ablaciones
  demo.py       probarlo a mano y ver la traza
  bench/        las 13 peticiones
```

Tests: [`tests/test_coordinator.py`](../tests/test_coordinator.py), con un cerebro simulado: sin
red, GPU ni `data/`.
