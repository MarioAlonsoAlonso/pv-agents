# Guía de anotación · rastreador de Bluesky

Anotamos **posts**, pero se leen **dentro de su hilo**: en `lote.csv` cada fila trae el texto
del post al que responde (`texto_responde_a`) y las filas de un mismo hilo van seguidas, en
orden de lectura (`posicion`). Abrid el CSV con Excel o LibreOffice (separador `;`).

Se rellenan cuatro columnas:

| Columna | Valores | Cuándo |
|---|---|---|
| `farmaco` | `sertralina`, `escitalopram`, `lorazepam`, `alprazolam`, `zolpidem`, `tramadol`, `metformina`, `levotiroxina`, `otro`, `ninguno` | Siempre |
| `tipo_mencion` | `nombre`, `marca`, `errata`, `indirecta`, `jerga` | Solo si `farmaco` es uno de los ocho |
| `es_experiencia` | `si`, `no` | Solo si `farmaco` es uno de los ocho |
| `notas` | texto libre | Dudas, casos raros |

## 1. ¿De qué fármaco habla el post? (`farmaco`)

Un post **habla del fármaco** si se refiere a él, lo nombre o no, **sin valorar nada más**:
cuenta igual una experiencia, una pregunta, una opinión, una broma o una noticia.

- *"Llevo un mes con sertralina y duermo fatal"* → `sertralina`.
- *"Shrek ha curado más depresiones que la sertralina"* → `sertralina` (es una broma, pero habla de ella).
- Respuesta a lo anterior: *"A mí también me dejó KO las primeras semanas"* → `sertralina`: por el hilo se sabe de qué habla.
- Respuesta en el mismo hilo: *"Qué bonito día hace hoy"* → `ninguno`.
- *"Me han recetado antidepresivos"*, sin decir cuál ni poder saberlo por el hilo → `otro`. Habla de un medicamento, pero no podemos decir que sea uno de los ocho.
- *"Tómate un ibuprofeno"* → `otro`.
- Un post sin relación con ningún medicamento → `ninguno`.

Si un post habla de dos fármacos vigilados, poned el más importante y el otro en `notas`.

## 2. ¿Cómo lo nombra? (`tipo_mencion`)

| Tipo | Qué es | Ejemplo |
|---|---|---|
| `nombre` | El principio activo, bien escrito | "la sertralina" |
| `marca` | Un nombre comercial (los de CIMA están en `crawler/data/marcas.json`, pero vale cualquiera, también extranjero: Zoloft) | "el Besitran", "Orfidal" |
| `errata` | Mal escrito o abreviado | "setralina", "la sertra", "lorazepan" |
| `indirecta` | No lo nombra; se sabe por el hilo | "a mí también me pasó" respondiendo a un post sobre sertralina |
| `jerga` | Lo describe sin nombrarlo, y se sabe cuál es por lo que dice | "las pastillas de la tiroides" (levotiroxina), "la pastilla para la diabetes que me da diarrea" (metformina) |

Si dudáis entre `indirecta` y `jerga`: es `indirecta` si **hace falta el hilo** para saberlo, y
`jerga` si **basta con el propio post**.

## 3. ¿Cuenta una experiencia? (`es_experiencia`)

`si` cuando alguien cuenta **lo que le pasó a él o a alguien cercano** con el fármaco (bueno o
malo, con síntomas o sin ellos). `no` para preguntas sin experiencia, opiniones generales,
bromas, noticias, consejos. El rastreador no usa esta columna; la usará el triaje.

- *"Con el Orfidal duermo como un bebé"* → `si`.
- *"¿La sertralina engorda?"* → `no`.
- *"Mi madre toma tramadol y está todo el día mareada"* → `si`.

## Cómo se reparte el trabajo

1. `make crawler-lote` genera `lote.csv` (todo el lote) y `validacion.csv` (100 posts al azar
   del mismo lote).
2. **Anotación principal**: el lote entero, repartido por hilos entre los tres. Si queréis
   ir más rápido, partid de la pre-anotación (`make crawler-prenotar`, en la DGX) y corregidla,
   pero entonces la validación ciega del paso 3 es imprescindible.
3. **Validación ciega**: una persona que **no** haya anotado esos posts anota `validacion.csv`
   sin mirar el lote. Después:
   ```bash
   uv run python -m crawler.annotate acuerdo data/bluesky/anotacion/lote.csv data/bluesky/anotacion/validacion.csv
   ```
   da la kappa de Cohen (≥ 0,8 muy bueno; < 0,6 la tarea es ambigua: revisad la guía con
   los desacuerdos, que se guardan en un JSON al lado).
4. Resolved los desacuerdos en `lote.csv` y `make crawler-gold`.

Los datos (`data/`) **no se suben al repositorio**: son textos públicos, pero hablan de la
salud de personas. Compartidlos entre vosotros por un canal privado.
