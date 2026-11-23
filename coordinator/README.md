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
| `create_icsr_report` | Herramienta de la fase 2, con `requires_confirmation` | Borrador del ICSR, pendiente de confirmar |

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

**Banco** ([`bench/tareas.jsonl`](bench/tareas.jsonl)): 13 peticiones de un técnico, cada una con
comprobación automática. Cubre casos y prioridad, codificación MedDRA, ficha técnica (con cita
del fragmento), señal en FAERS (PRR calculado en vivo), normativa, ICSR pendiente de
confirmación, evidencia en redes y PubMed, un resumen completo para el comité y dos peticiones
imposibles (otra red social y otro año; un fármaco sin casos), donde tiene que decir que no lo
sabe. Las peticiones nombran casos con huecos (`{caso_prioritario}`…) que se rellenan con los
datos, así que el banco no contiene ningún post.

**Métricas:** éxito, uso de las herramientas necesarias, pasos, tokens, tiempo y **números sin
respaldo**: cifras de la respuesta que no aparecen en ninguna observación (recuentos, PRR o
frecuencias inventados).

**Ablaciones** (`--variantes`): `completo`, `sin_rag`, `sin_meddra` (el 27B tiene que poner los
términos MedDRA él solo) y `sin_evidencia`. Responden a "¿aporta cada pieza?".

Las cifras van a `reports/coordinator_eval.json`; las respuestas y las trazas, a
`data/coordinador/eval/` (pueden parafrasear posts: fuera de git). Si se corta la sesión,
relanzar el mismo comando sigue donde se quedó.

## Resultados

Pendiente (DGX).

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
