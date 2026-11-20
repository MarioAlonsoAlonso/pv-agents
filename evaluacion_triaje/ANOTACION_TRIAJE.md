# Guía de anotación · triaje

Hoja de soluciones para evaluar el **agente de triaje**: para cada mención que entrega el
rastreador (contrato en [`crawler/CONTRATO_TRIAJE.md`](../crawler/CONTRATO_TRIAJE.md)), qué
debería contestar un buen triaje. Cómo usar la hoja para construir y evaluar el agente:
[`GUIA_TRIAJE.md`](GUIA_TRIAJE.md). Se anota **lo que dice el post**, leído con su conversación
(`conversacion` + `texto`), sin suponer nada que no esté escrito.

En la hoja (`lote_triaje.csv`, separador `;`) cada fila es una mención. Las columnas `texto` y
`conversacion` son para leer; se rellenan las de la derecha.

## 1. `es_efecto_adverso`: `si` / `no`

`si` cuando **el autor cuenta que a él o a alguien concreto le pasó algo malo, o algo distinto de
lo esperado, con el fármaco**:

- síntomas o molestias atribuidos al fármaco: "me paso el día con náuseas", "no pego ojo";
- síntomas al **dejarlo o bajar la dosis** (retirada): "al dejarla me mareaba";
- **falta de eficacia**: "llevo dos meses y no me hace nada" (en MedDRA es *Drug ineffective*);
- también si la atribución es dudosa ("no sé si es la sertralina o el estrés"): cuenta, con la
  duda en `notas`.

`no` en todo lo demás, y entonces se rellena `motivo_no`:

| `motivo_no` | Ejemplos |
|---|---|
| `no_es_mencion` | No habla del fármaco: "gracias 🫂", "ánimo", el personaje de una novela (error del rastreador) |
| `pregunta_o_consejo` | "¿la sertralina engorda?", "pregunta a tu médico", "a mí me la recetaron por la ansiedad" sin más |
| `broma_opinion_noticia` | "Shrek ha curado más depresiones que la sertralina", un estudio, una opinión sobre los psicofármacos |
| `sin_reaccion` | Experiencia sin nada malo: "me va genial", "llevo dos años y bien", solo que la toma |

## 2. `reacciones` y `negadas`

- `reacciones`: cada reacción **con las palabras del autor**, separadas por ` | `:
  `me paso el día con náuseas | no pego ojo`.
- Si el post **no dice cuál** y solo se sabe por la conversación ("a mí también me pasó"), se
  escribe la reacción que se deduce, entre corchetes: `[náuseas]`.
- `negadas`: lo que el autor dice que **no** tuvo: `náuseas no he tenido` → `náuseas`.
- Vacías si `es_efecto_adverso = no`.

## 3. El paciente

| Columna | Valores |
|---|---|
| `paciente` | `autor` (le pasa a quien escribe) · `familiar_o_conocido` ("mi madre", "mi hermana") · `desconocido` |
| `edad` | El número si se dice ("tengo 34 años", "mi madre de 78") · si no, `desconocido` |
| `sexo` | `mujer` · `hombre` · `desconocido` |
| `sexo_fuente` | `dicho` ("soy mujer", "mi madre") · `deducido` (por el género gramatical: "estoy **cansada**") · vacío si `desconocido` |

Solo para `es_efecto_adverso = si`; si no, se dejan vacías. No se deduce la edad.

## 4. Gravedad y desenlace

| Columna | Valores |
|---|---|
| `grave` | `si` si el post dice uno de los criterios · si no, `no_consta` |
| `criterio_gravedad` | `muerte` · `riesgo_vital` · `hospitalizacion` (ingreso, urgencias con ingreso) · `discapacidad` · `otro_importante` (p. ej. intento de suicidio, convulsiones) · vacío |
| `desenlace` | `recuperado` ("se me quitó") · `recuperandose` ("ya voy mejor") · `no_recuperado` ("sigo igual") · `desconocido` |

Solo para `es_efecto_adverso = si`. Ir a urgencias **sin** ingreso no es hospitalización.

## 5. `notas`

Dudas, y el motivo de cualquier decisión no evidente. La preanotación de Claude escribe aquí
`prenotado claude`.

## Cómo se hace la hoja de soluciones

Las menciones son las que entrega el rastreador (`hibrido_27b_menciones.jsonl`, formato 1.1). Los
ficheros van en `data/triaje/`, fuera de git.

1. **Hoja en blanco** (opcional; la de Claude ya está hecha):
   `uv run python -m evaluacion_triaje.lote crear --menciones data/hibrido_27b_menciones.jsonl`
2. **Claude preanota** las menciones siguiendo esta guía → `lote_triaje_claude.csv`.
3. **El 27B anota** las mismas por su cuenta (en la DGX):
   `uv run python -m evaluacion_triaje.lote prenotar --menciones data/hibrido_27b_menciones.jsonl`
   → `lote_triaje_prenotado_27b.csv`.
4. **Se comparan**:
   `uv run python -m evaluacion_triaje.lote acuerdo data/triaje/lote_triaje_claude.csv data/triaje/lote_triaje_prenotado_27b.csv`
   (kappa de cada decisión).
5. **Revisión humana, solo de lo que cambia la respuesta** (efecto adverso sí/no y gravedad; el
   motivo del "no" se deja, porque sus categorías se solapan):
   `uv run python -m evaluacion_triaje.lote revisar data/triaje/lote_triaje_claude.csv data/triaje/lote_triaje_prenotado_27b.csv`
   → `revision.csv`. Una persona del equipo rellena `decision` en cada fila: `a` (Claude), `b`
   (27B) o `descartar` (sin respuesta clara). La columna `razon_a` explica la decisión de Claude.
   Después: `uv run python -m evaluacion_triaje.lote aplicar data/triaje/lote_triaje_claude.csv data/triaje/lote_triaje_prenotado_27b.csv data/triaje/revision.csv`
   → `lote_triaje_final.csv`.
6. **Hoja de soluciones**:
   `uv run python -m evaluacion_triaje.lote consolidar data/triaje/lote_triaje_final.csv`
   → `gold_triaje.jsonl`, con la que se corrige el agente de triaje.
7. **Desarrollo y test** (opcional, recomendado):
   `uv run python -m evaluacion_triaje.lote partir data/triaje/gold_triaje.jsonl --menciones data/hibrido_27b_menciones.jsonl`
   → respuestas `gold_triaje_dev.jsonl` y `gold_triaje_test.jsonl`, y las preguntas partidas igual,
   `hibrido_27b_menciones_dev.jsonl` y `hibrido_27b_menciones_test.jsonl`. Desarrollo (40 % de las
   conversaciones) es para practicar y mejorar el triaje mirando sus fallos; test (60 %) se usa una
   sola vez, al final, y durante la práctica el triaje ni siquiera se ejecuta sobre sus preguntas.
   Las menciones de una misma conversación quedan siempre en el mismo lado.

Los datos (`data/`) no se suben al repositorio: son textos reales sobre la salud de personas.
