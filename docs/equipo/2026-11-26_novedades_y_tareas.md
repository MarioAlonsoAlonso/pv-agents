# Novedades del 26-11 y tareas pendientes

Para el equipo (triaje y coordinador). Todo lo de este documento está en la rama `rastreador`,
pendiente de un *pull request* a `main` que tenéis que revisar, porque toca código vuestro.

**En cinco líneas:**

1. El coordinador ya puede responder sobre **cualquier etapa** (lo que encontró el rastreador, lo
   que descartó el triaje, qué falta) y **lanzar el rastreador y el triaje** desde la
   conversación, con confirmación si el trabajo es grande.
2. Hay **chat en la terminal y en una página web**, con memoria entre turnos y botones para
   confirmar.
3. Las **notificaciones por correo o teléfono** pasan por el mismo triaje que los posts.
4. Hay un **catálogo único de fármacos** y un **único sitio para las rutas de `data/`**.
5. Quedan tres tareas vuestras: **bloque 0** (revisar y comprobar que no se ha roto nada),
   **cambio A** (codificar MedDRA con el 27B) y **cambio G** (mejorar el triaje).

---

## 1. Qué ha cambiado

### En el coordinador (`coordinator/`)

| Cambio | Qué es | Dónde |
|---|---|---|
| **Herramientas de las etapas** | `estado_del_sistema` (qué se ha rastreado, triado y codificado de cada fármaco, y qué falta), `listar_menciones` (lo que encontró el rastreador), `listar_descartes` (lo que el triaje dijo que no, y por qué), `rastrear` (el vigilante, desde y hasta una fecha), `triar` (las menciones que esperan) | `tools.py`, `etapas.py` |
| **`ver_mencion` sustituye a `ver_caso`** | Lee cualquier post, sea caso o no; si es caso, añade todo lo del triaje y los términos MedDRA. **La evaluación cuenta `ver_mencion` como `ver_caso`**, así que vuestras trazas guardadas puntúan igual | `tools.py`, `evaluate.py` (`ALIASES`) |
| **Confirmación de los trabajos grandes** | Un rastreo de más de 31 días o un triaje de más de 100 menciones no lo lanza el modelo: queda pendiente con su coste estimado, como el ICSR | `agent.py` (`hold_if_big`), `etapas.py` |
| **En los exámenes no se lanza nada** | `coordinar(..., hold_stages=True)` en `evaluate.py`: `rastrear` y `triar` solo se proponen | `agent.py`, `evaluate.py` |
| **Conversación** | `Conversacion`: memoria entre turnos; confirmar o rechazar lo pendiente; los rastreos y triajes confirmados van **en segundo plano** | `agent.py` |
| **Chat y web** | `make coordinador-chat` (terminal) y `make coordinador-web` (página con botones; en la DGX, túnel `ssh -L 7860:localhost:7860`) | `chat.py`, `web.py` |
| **El almacén lee más fuentes** | Además de la muestra, el almacén del vigilante y las notificaciones. Guarda **todas** las menciones y respuestas del triaje, no solo los casos. **Los bancos usan solo la muestra** (`include_monitor=False`), para que sus cifras no cambien | `cases.py` |
| **Casos que pueden ser el mismo** | `resumen_casos` dice cuántas conversaciones distintas hay y cuántos casos comparten conversación o repiten texto; `listar_casos` da `misma_conversacion` | `cases.py`, `tools.py` |
| **Notificaciones** | `registrar_notificacion(texto, farmaco, origen)`: correo o teléfono → mismo triaje → caso | `etapas.py`, `tools.py` |
| **Prompt** | Instrucciones de las herramientas nuevas, «si falta una etapa, dilo y ofrécete a lanzarla», «N casos en M conversaciones», y que el técnico puede leer un post entero | `prompts.py` |
| **Banco v5** | 24 peticiones sobre las etapas (10 de práctica y 14 de examen), comparando con la variante `sin_etapas` | `bench/v5.py` |

### En el triaje (`triage/`)

**El triaje no cambia de comportamiento.** Solo se ha movido código de sitio:

- Los campos, sus valores y la comprobación (`LABELS`, `VALUES`, `normalise`, `validate`) están
  ahora en `triage/esquema.py`. Antes estaban en `evaluacion_triaje/lote.py`, que ahora los
  importa de ahí: el sistema ya no depende de la herramienta de evaluación.
- El prompt v0 está en `triage/prompts.py`, **idéntico letra a letra**: lo comprobé con el hash.
  v1 y v2 tampoco cambian.
- `triage/run.py` lee las rutas de `common/rutas.py` (son las mismas).
- **Ahora el triaje también se usa en producción**, con su configuración final (v2, un post cada
  vez):
  - desde el vigilante: `crawler.monitor --triar`;
  - desde el coordinador: `triar` y `registrar_notificacion`.

### Piezas comunes (`common/`)

| | Qué |
|---|---|
| `common/farmacos.py` | **Catálogo único** de los 35 fármacos: nombre, nombre en FAERS, grupo, si está vigilado y si tiene ficha. Lo leen el rastreador, FAERS (`tool_use`) y el coordinador. Un test avisa si deja de coincidir con los fármacos de la fase 1 |
| `common/rutas.py` | **Todas las rutas de `data/`**: las mismas de antes, escritas en un solo sitio |
| `common/stats.py` | El intervalo de Wilson, uno solo (vuestras cifras no cambian) |

### En el rastreador y el contrato

- **Contrato 1.2.** `fuente` admite `correo`, `telefono` y `otro`; `encontrado_por` admite
  `tecnico`. Solo se añaden valores: los ficheros 1.1 se siguen leyendo igual.
- **El vigilante** acepta `--desde`, `--hasta` y `--triar`, y tiene un bloqueo para que dos
  ejecuciones no escriban a la vez.
- **Cambio F, importante para el triaje.** Según la hoja de soluciones del triaje, de las 302
  menciones que guardó el agente del rastreador, **165 no hablaban del fármaco**: «gracias»,
  «lo siento», antidepresivos en general, otro fármaco… El triaje las recibía como ruido, y por
  eso en el examen solo acierta el **54 %** de las veces que dice «sí» a una respuesta indirecta.
  Hay dos arreglos:
  - **un filtro en el código** que rechaza las respuestas de solo cortesía. Es seguro, pero quita
    pocas: 3 de 109 errores en el examen y ninguna mención buena;
  - **reglas nuevas en el prompt** del agente para el resto. Todavía sin medir: lo mediré
    volviendo a ejecutar el híbrido en la DGX.

  Si funciona, **al triaje le llegará menos ruido y debería acertar más con las respuestas
  indirectas sin tocar nada vuestro**. Pero no sustituye mejorar el triaje (cambio G).
- **`buscar_mas_casos`** (la herramienta aparcada) se ha borrado.

### En la fase 1 (`rlm/`)

- **`rlm/evaluate_pista.py`** (`make rlm-pista`). ¿Codifica mejor el 27B si el triaje le da antes
  las reacciones y lo que el paciente niega? El experimento:
  1. el triaje lee las 200 notificaciones del test de la fase 1;
  2. se codifica cada notificación sin y con esa ayuda;
  3. se comparan los resultados con McNemar.

  En el coordinador está **apagado** (`ARCA_MEDDRA_PISTA=1` para encenderlo) hasta medirlo.

---

## 2. Tarea: bloque 0 (antes de nada)

1. **Revisad el *pull request*** `rastreador` → `main`. Ficheros vuestros tocados:
   - del coordinador: `coordinator/{agent,cases,tools,prompts,evaluate,etapas,chat,web,drugs}.py`
     y `coordinator/bench/v5.py`;
   - del triaje: `triage/{agent,prompts,evaluate,run,esquema}.py`;
   - de la fase 1: `rlm/evaluate_pista.py`;
   - de las fases 2 y 4: `tool_use/tools.py` (los nombres de FAERS salen del catálogo) y
     `agent/react_agent.py` (de antes).
2. **Pasad los tests:** `uv run pytest`. Aquí pasan todos salvo `test_grpo_step` y
   `test_code_agent`, que necesitan `torch` y `datasets`.
3. **Comprobad que el coordinador no ha empeorado.** He cambiado su prompt y sus herramientas.
   Repetid **solo la práctica** de vuestros bancos (**los exámenes no se repiten**) y comparad con
   vuestras cifras de práctica anteriores:
   ```bash
   uv run python -m coordinator.evaluate --bench coordinator/bench/tareas_v2.jsonl --conjunto dev --variantes completo --etiqueta tras_etapas
   uv run python -m coordinator.evaluate --bench coordinator/bench/tareas_v4.jsonl --conjunto dev --variantes completo --etiqueta tras_etapas
   ```
4. **Evaluad lo nuevo:** `make coordinador-eval-etapas` (la práctica del v5, completo frente a
   `sin_etapas`). Si sale bien, `make coordinador-examen-etapas` **una sola vez**.
5. **Pasadme vuestras respuestas del triaje** (`data/triaje/eval/v2_27b_mencion_{dev,test}_respuestas.jsonl`)
   por un canal privado: mi copia de la DGX las necesita para tener los 159 casos.

---

## 3. Tarea: cambio A (codificar MedDRA con el 27B, y mejorar su precisión)

**Por qué.** Ya está medido que el 27B codifica mejor que el modelo pequeño de la fase 1:

| Test de la fase 1 | 1,7B (SFT + GRPO) | 27B sin razonamiento | 27B con razonamiento |
|---|---|---|---|
| Examen (200), corrección estricta | **56,5 %** | 38 % | 46 % |
| Examen (200), corrección tolerante | 56,5 % | 60 % | **67,5 %** |
| Términos nunca vistos (100), tolerante | 38 % | 57 % | **74 %** |

Además, en la ablación del coordinador, quitar el modelo pequeño apenas cambia el resultado
(52 de 59 frente a 54 de 59) y el 27B es unas 50 veces más rápido. Con el 27B se podrían **codificar
todos los casos al terminar el triaje**, en minutos y no en horas.

**La propuesta:**
- que `codificar_meddra` y `codificar_notificacion` usen el 27B, con las mismas reglas de la fase 1;
- dejar el 1,7B como **respaldo local** (sin internet, verificable): sigue siendo lo que defendemos
  de la fase 1.

**Cómo mejorar la precisión del 27B**, de su error más frecuente al menos:

1. **El formato.** Con la corrección estricta, casi todos sus fallos son respuestas buenas sin
   `<answer>…</answer>`. Pedirle el formato explícitamente, o leer su respuesta corta como hace
   `lenient_correct`, recupera la mayor parte.
2. **El subconjunto de 90 términos.** A veces da un término de MedDRA válido que no está en el
   repositorio, o un sinónimo (*Lack of appetite* en vez de *Decreased appetite*). Dos opciones:
   - darle la lista en el prompt;
   - comprobarlo en el código con `MedDRADictionary.to_pt` y, si no está, devolverle el error
     para que lo corrija, como hace el coordinador con las citas.
3. **Ejemplos.** Unos cuantos ejercicios resueltos de `rlm/data/train.jsonl` en el prompt, sobre
   todo de las familias que más falla: diagnóstico que absorbe síntomas y varias reacciones.
4. **Razonamiento.** Con razonamiento acierta más (67,5 % frente a 60 %), pero gasta muchos más
   tokens. Decidid con números.
5. **La ayuda del triaje** (cambio I): `make rlm-pista` dice si darle las reacciones del triaje
   ayuda.

**Ojo con el examen.** El test de la fase 1 ya se ha usado con el 27B. Para probar prompts,
**iterad con una parte de `rlm/data/train.jsonl`** (tiene las respuestas) y medid **una vez** al
final en el test, con `rlm/evaluate_remote.py`: estricto, tolerante y McNemar frente al 1,7B.

---

## 4. Tarea: cambio G (mejorar el triaje)

**El problema.** En el examen, el triaje detecta el **79 %** de los efectos adversos y acierta el
**80 %** de las veces que dice «sí». **El 21 % de casos perdidos es el error caro**: nadie los
vuelve a ver. Y con las respuestas indirectas solo acierta el 54 % de sus «sí».

**La propuesta.** La que tú mismo apuntaste en tus conclusiones: **ser más permisivo cuando el
modelo duda**. Tres formas de hacerlo:

| Forma | Cómo |
|---|---|
| Una tercera respuesta, «dudoso» | Lo dudoso va a revisión humana |
| Pedirle la seguridad | Un «no» poco seguro se trata como «sí» |
| Una regla en el código | Si el post cuenta un síntoma (una lista de palabras) y el modelo dijo «no» por broma o «sin efecto», va a revisión |

Va de la mano de una **cola de revisión humana**: el técnico ve los dudosos y los confirma o los
corrige. Sin esa cola, ser más permisivo solo llena los casos de falsas alarmas.

**Cómo medirlo (sin tocar tu examen, que ya está usado):** hace falta un **lote nuevo anotado**.

1. Tengo **128 menciones nuevas** de sertralina, de septiembre a noviembre de 2026, rastreadas
   desde el chat. Están en mi DGX, en `data/bluesky/monitor/menciones.jsonl`; te las paso por un
   canal privado.
2. Anótalas con la misma guía (`evaluacion_triaje/ANOTACION_TRIAJE.md`). Para ir más rápido,
   preanótalas con `python -m evaluacion_triaje.lote prenotar --menciones <fichero>` y revisa.
3. Compara la v2 actual con la versión permisiva en ese lote: detección, precisión y cuántas van a
   revisión.

**Junto con el cambio F.** Si el nuevo prompt del rastreador funciona, al triaje le llegarán menos
«gracias» y respuestas de otros fármacos. Eso debería subir su precisión con las respuestas
indirectas sin cambiar nada del triaje. Pero los casos perdidos no se arreglan así: para eso es el
cambio G.

**Un detalle:** las instrucciones del triaje hablan de «posts de Bluesky», y ahora también le
llegan notificaciones por correo o teléfono. Sería bueno añadir una línea al prompt, o probarlo
con unas pocas notificaciones de ejemplo.

---

## 5. Lo que hago yo (DGX)

- Volver a ejecutar el híbrido para medir el cambio F (`reports/crawler_eval_sertralina_v5.json`).
- `make rlm-pista`, si lo prefieres hacer tú dentro del cambio A, dímelo.
- Pasaros las 128 menciones nuevas para el lote del cambio G.

Dudas, por el grupo. La guía de ramas y *pull requests* es [`COLABORAR.md`](../../COLABORAR.md).
