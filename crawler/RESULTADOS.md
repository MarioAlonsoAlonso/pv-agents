# Resultados · Rastreador de menciones en Bluesky

**Pregunta:** ¿aporta algo un agente (Qwen3.8-27B) frente a una búsqueda por palabras clave para
encontrar los posts que hablan de un fármaco, también cuando no lo nombran?

**Respuesta corta:** sí, si se usa donde hace falta criterio. Una búsqueda fija encuentra el 70 %
de las menciones y ninguna indirecta; el **híbrido** (búsqueda fija + agente que lee las
conversaciones) encuentra el **97 %**, con un coste de ~2,3 M de tokens por fármaco y pasada.
**Es el sistema elegido** para el vigilante (`crawler.monitor`). El agente solo llega al 91 % con
mejor precisión, pero casi al doble de coste.

## 1. Cómo se ha medido

**Muestra congelada.** 18 548 posts en español de Bluesky publicados entre el 1 de enero de 2025
y el 18 de noviembre de 2026 (`crawler.collect`): búsquedas por "sertralina" y sus marcas de CIMA
(Aremis, Besitran, Semonic), por términos generales que no nombran ningún fármaco
("antidepresivos", "ansiolíticos", "me han recetado", "efectos secundarios"…) y por "ibuprofeno"
como ruido, más las conversaciones completas de lo encontrado. Los términos generales evitan que
la evaluación sea circular: la muestra contiene posts que la búsqueda fija no puede encontrar.

**Hoja de soluciones.** 576 posts en 61 conversaciones completas elegidas por estratos
(`crawler.annotate seleccionar`): 30 conversaciones y 10 posts sueltos de sertralina, 15 hilos de
términos generales y 6 de ibuprofeno. Anotados por Claude siguiendo [`ANOTACION.md`](ANOTACION.md)
y contrastados con una anotación independiente del 27B:

- Acuerdo: κ = 0,77 en "habla o no de un fármaco vigilado" y 0,72 en "de cuál".
- De los 83 desacuerdos, 39 son `otro` frente a `ninguno` (irrelevantes aquí) y 29 son errores
  claros del 27B: se inventa el tramadol en un hilo sobre pastillas para el dolor (nadie lo
  nombra), contagia el tema del hilo a respuestas como un "gracias" y confunde el nombre de un
  personaje de un libro con la marca Aremis.
- 9 posts genuinamente dudosos se descartaron: no cuentan ni a favor ni en contra.

Quedan **567 posts con 66 menciones de sertralina** (46 por nombre, 19 indirectas, 1 por una marca
que no está en CIMA) y 501 negativos. Los negativos son imprescindibles: sin ellos un sistema que
lo guarda todo sacaría un 100 %.

**Métricas** (solo sobre los posts anotados; cada sistema busca en la muestra entera):
*recall* = menciones anotadas encontradas / 66, también por tipo de mención; precisión = de lo
que envía entre los anotados, cuánto es de verdad sertralina; posts enviados al triaje (todos);
tokens y tiempo. Intervalos de Wilson al 95 %.

## 2. Los cuatro sistemas

| Sistema | Qué hace | Modelo |
|---|---|---|
| `fija` | Busca el principio activo y las marcas de CIMA y guarda todo lo encontrado | No |
| `fija_hilos` | Lo anterior + todos los posts de esas conversaciones | No |
| `hibrido` | `fija` + el agente lee esas conversaciones y guarda las respuestas que hablan del fármaco | 27B |
| `agente` | El agente solo, ventanas de 30 días: decide qué buscar, qué leer y qué guardar | 27B |

## 3. Resultado final (versión 4)

| Sistema | *Recall* | Nombre | Indirecta | Marca | Precisión | Al triaje | Tokens | Tiempo |
|---|---|---|---|---|---|---|---|---|
| `fija` | 70 % [58–79] | 46/46 | 0/19 | 0/1 | 98 % [89–100] | 423 | 0 | 1 s |
| `fija_hilos` | 100 % [94–100] | 46/46 | 19/19 | 1/1 | 20 % [16–25] | 2 712 | 0 | 1 s |
| **`hibrido`** | **97 % [90–99]** | 46/46 | **17/19** | 1/1 | 89 % [80–94] | 708 | 2,26 M | 11 min |
| `agente` | 91 % [82–96] | 46/46 | 13/19 | 1/1 | **98 % [91–100]** | 526 | 4,14 M | 13 min |

Fuente: [`reports/crawler_eval_sertralina_v4.json`](../reports/crawler_eval_sertralina_v4.json).

**Lectura.**

- **La búsqueda fija es imbatible en lo fácil y ciega en lo difícil.** Todas las menciones por
  nombre, ninguna indirecta. Su único error es un personaje de una novela que se llama casi igual
  que una marca.
- **Leer las conversaciones enteras sin criterio no sirve.** `fija_hilos` lo encuentra todo, pero
  4 de cada 5 posts que envía no hablan del fármaco: el triaje recibiría 2 712 posts en lugar de
  ~500.
- **El híbrido se queda con lo mejor de los dos.** 17 de las 19 indirectas, y además descubre por
  su cuenta la marca extranjera y términos como "sertra" o "certralina" (`proponer_termino`). Su
  precio es algo de ruido: de las 25 menciones anotadas que guardó el agente, 7 eran respuestas
  de cortesía o apoyo ("gracias", emojis) dentro de conversaciones sobre la sertralina (el
  octavo error es el personaje de novela de la búsqueda fija).
- **El agente solo** comete menos errores de precisión (1 frente a 8), pero pierde más indirectas
  (6 frente a 2) y gasta casi el doble, porque la búsqueda fija del híbrido hace gratis la parte
  fácil. La diferencia de *recall* entre ambos no es significativa con 66 casos (McNemar exacto,
  6 frente a 2 discordantes, p = 0,29).
- **Decisión:** híbrido. Cuesta la mitad, no pierde nada de lo que encuentra la búsqueda fija
  (las menciones por nombre las guarda el código, no el modelo) y su ruido es fácil de descartar
  para el triaje.

## 4. Cómo se llegó aquí: cuatro versiones

Todas se midieron sobre la misma muestra y la misma hoja de soluciones.

| Versión | Cambio | Híbrido: *recall* / precisión / tokens | Agente: *recall* / precisión / tokens |
|---|---|---|---|
| v1 | Primera versión: guardar un post por paso | 92 % / 98 % / 0,37 M\* | 70 % / 98 % / 5,35 M |
| v2 | Guardar varios posts en un paso (lista de objetos) | 74 % / 89 % / 2,13 M | 23 % / 94 % / 2,36 M |
| v3 | La lista como texto "id tipo; id tipo" + el código reenvía los hilos no leídos | 95 % / 85 % / 2,06 M | 65 % / 98 % / 3,61 M |
| v4 | Sin tope de 30 posts por llamada + 2 048 tokens de respuesta + "guarda al acabar cada página o hilo" | **97 % / 89 % / 2,26 M** | **91 % / 98 % / 4,14 M** |

\* v1 del híbrido leyó solo las 32 conversaciones con posts anotados; desde v2, las 235.

Qué falló en cada una (el detalle está en las trazas de `data/…/eval/`):

1. **v1 · Sin pasos para leer.** El agente solo tiene 40 pasos por mes y guardaba los posts de
   uno en uno: en 11 de los 23 meses agotó los pasos. Abrió 94 conversaciones, solo 3 de las 7
   anotadas con menciones indirectas, y encontró 3 de 19.
2. **v2 · Una orden que se perdía.** Para guardar varios posts a la vez se pasaba una lista de
   objetos. El servidor del curso, al traducir la llamada a herramienta, a veces la perdía: 25
   llamadas llegaron sin lista, el agente las repetía y la detección de bucles cortó 10 meses.
   Además, en 3 de las 40 tareas del híbrido el modelo respondió "he leído los hilos" tras abrir
   1 o 2 de 6, y se saltó justo las dos conversaciones anotadas con más indirectas (11 de 19).
   Con solo 6 tareas, en v1 ese fallo no había aparecido.
3. **v3 · Fallos de fontanería.** Con la lista como texto y el código comprobando qué hilos se
   leyeron de verdad, el híbrido sube al 95 %. El agente solo perdió 14 posts que nombraban el
   fármaco aunque los había visto: un tope de 30 posts por llamada que descartaba el resto sin
   avisar (error de implementación), dos meses en los que acumuló la lista para el final y la
   respuesta se cortó a los 768 tokens antes de escribir la llamada, y un id mal copiado repetido
   hasta la detección de bucles.
4. **v4 · Arreglados** los dos primeros. Los ids mal copiados siguen apareciendo de forma
   aislada (2 en 23 meses), pero ya no bloquean ningún mes.

**Lo que enseñan las cuatro versiones:** con el mismo modelo, el resultado se movió entre el 23 %
y el 91 % solo por el diseño de las herramientas y por lo que se garantiza en el código. Las
garantías que más pesaron no están en el prompt: que el texto guardado se copie de la fuente
(nunca lo escribe el modelo), que solo se puedan guardar posts que el agente ha visto, que el
código compruebe qué hilos se han leído y que las menciones explícitas las guarde el código.

## 5. Coste

- **Híbrido:** 2,26 M de tokens y 11 minutos por fármaco para 23 meses de Bluesky. En uso diario
  (`crawler.monitor`) solo se leen las conversaciones nuevas o con respuestas nuevas: un
  orden de magnitud menos.
- **Agente solo:** 4,14 M de tokens y 13 minutos.
- La mayor parte del coste son tokens de entrada: en cada paso se reenvía toda la conversación
  (instrucciones, búsquedas, hilos leídos y razonamientos anteriores).
- El servidor del curso es compartido (5 peticiones simultáneas para todos los grupos);
  `ARCA_REMOTE_LLM_CONCURRENCY` limita las nuestras a 2.

## 6. Límites (para leer las cifras con cuidado)

- **Ajuste al examen.** Las cuatro versiones se corrigieron mirando los fallos sobre los mismos
  567 posts. La versión final debería validarse, sin tocar nada, con un lote nuevo o con otro
  fármaco (el lote ya incluye 8 menciones de lorazepam, pocas para concluir).
- **Pocos casos.** 66 menciones y solo 19 indirectas: los intervalos son anchos (el *recall* del
  híbrido, 90–99 %; el del agente, 82–96 %).
- **Un fármaco, una red, un idioma.** Sertralina, Bluesky, español.
- **Anotación de un modelo.** La hoja de soluciones la hizo Claude, contrastada con el 27B; los
  9 casos dudosos los descartó el equipo. No hubo una doble anotación humana completa.
- **El modelo varía.** El 27B muestrea (temperatura 0,7): repetir la misma versión cambia unos
  puntos. Una sola ejecución por versión.

## 7. Reproducir

```bash
export ARCA_CRAWLER_DATA=data/bluesky_sertralina
uv run --extra crawler python -m crawler.collect --farmacos sertralina --max-hilos 1000
uv run python -m crawler.annotate seleccionar --conversaciones 30 --sueltos 10 --amplios 15
# anotar (ANOTACION.md) → anotacion/lote_final.csv
uv run python -m crawler.annotate consolidar $ARCA_CRAWLER_DATA/anotacion/lote_final.csv
uv run python -m crawler.evaluate --systems fija fija_hilos --farmacos sertralina
uv run python -m crawler.evaluate --systems hibrido --brain 27b --farmacos sertralina --todos-los-hilos
uv run python -m crawler.evaluate --systems agente --brain 27b --farmacos sertralina
```

Los posts y las anotaciones no están en el repositorio (datos de salud de personas); los informes
de [`reports/`](../reports/) solo tienen cifras y los términos descubiertos.
