# Guía para construir y evaluar el agente de triaje

Para quien desarrolla el triaje. Resume qué recibes, qué tienes que construir, cómo practicar,
cómo examinarte y qué contar en el informe.

## 1. La idea en una frase

Te damos un **examen**: menciones reales de Bluesky que encontró el rastreador (las preguntas) y,
para cada una, lo que un buen triaje debería contestar (las respuestas). Tu agente tiene que
acertar lo más posible, **practicando con una parte y examinándose una sola vez con la otra**.

## 2. Qué recibes (por un canal privado, nunca por GitHub)

| Cuándo | Fichero | Qué es |
|---|---|---|
| Al empezar | `hibrido_27b_menciones_dev.jsonl` | 287 **preguntas de práctica** (menciones, formato del [contrato](../crawler/CONTRATO_TRIAJE.md)) |
| Al empezar | `gold_triaje_dev.jsonl` | Sus 287 **respuestas** (72 efectos adversos) |
| **Al final** | `hibrido_27b_menciones_test.jsonl` | 438 **preguntas del examen** |
| **Al final** | `gold_triaje_test.jsonl` | Sus 438 **respuestas** (89 efectos adversos) |

Ya están en el repositorio: el [contrato](../crawler/CONTRATO_TRIAJE.md) (qué es cada campo de una
mención), la [guía de anotación](ANOTACION_TRIAJE.md) (qué significa cada respuesta, léela
entera: es la definición de "acertar") y 7 ejemplos inventados en
[`crawler/contrato/ejemplo_menciones.jsonl`](../crawler/contrato/ejemplo_menciones.jsonl).

Las preguntas y las respuestas de práctica y examen están partidas **por conversaciones**: las
menciones de un mismo hilo están siempre en el mismo lado.

## 3. Qué tienes que construir

Un agente que, para cada mención, conteste **exactamente las preguntas de la guía de anotación,
con los mismos nombres de campo y los mismos valores**:

```json
{"id": "M-56a92a5028", "es_efecto_adverso": "si", "motivo_no": "",
 "reacciones": "me paso el día con náuseas", "negadas": "",
 "paciente": "autor", "edad": "desconocido", "sexo": "desconocido", "sexo_fuente": "",
 "grave": "no_consta", "criterio_gravedad": "", "desenlace": "desconocido"}
```

- **Entrada:** las menciones, leídas con `crawler.store.read_mentions` (valida el formato).
  Para entender una respuesta indirecta ("a mí también me pasó"), usa `conversacion` + `texto`.
- **Salida:** un fichero JSONL con una línea por mención y su `id`. Con los mismos nombres que
  la hoja de soluciones, compararlas es inmediato.
- **Dónde:** en tu carpeta, `triage/`, en tu rama de git (`triaje`).

## 4. Practicar (todas las veces que quieras, solo con desarrollo)

El ciclo es:

1. Ejecutas el agente sobre las **287 de práctica**.
2. Lo comparas con `gold_triaje_dev.jsonl` y sacas los números (apartado 6).
3. **Miras los fallos uno a uno, con su texto**: los efectos adversos que se le escaparon y los
   que se inventó.
4. **Buscas el patrón**: ¿confunde las bromas con efectos adversos? ¿Cuenta síntomas de otro
   fármaco? ¿No lee la conversación? ¿Pone "hospitalización" cuando solo fue a urgencias?
5. **Cambias UNA cosa**: el prompt, una regla, un ejemplo, cómo le pasas la conversación.
6. Vuelves al paso 1 y compruebas si mejora **y si no empeora otra cosa**.
7. Apuntas en `EXPERIMENTS.md` qué cambiaste y qué pasó, con fecha. Es material directo para el
   informe.

Aunque el 27B no se entrene, cada cambio que decides mirando estos ejemplos **ajusta el sistema a
ellos**. Por eso existe la parte de examen.

## 5. Examinarte (una sola vez, al final)

- **No abras las preguntas ni las respuestas del examen hasta haber terminado.** Si las miras y
  cambias el agente por lo que ves, el examen deja de medir nada.
- Cuando el agente esté terminado, **sin tocar nada más**, ejecútalo una vez sobre las 438 del
  examen y compara con `gold_triaje_test.jsonl`. **Esa es tu nota para el informe.**
- Si comparas alternativas (otro prompt, otro modelo, con y sin razonamiento), **elige en
  práctica** y mide en el examen solo la elegida (o todas, pero sin volver a cambiar nada).

## 6. Qué medir

Tu programa de evaluación (en `triage/`) empareja tu salida con la hoja por `id` y calcula:

| Medida | Cálculo | Por qué importa |
|---|---|---|
| **Detección** (*recall*) | De los efectos adversos de la hoja, cuántos marcas `si` | Perder un caso real es lo grave en farmacovigilancia |
| **Falsas alarmas** (precisión) | De los que marcas `si`, cuántos son `si` en la hoja | Cada falsa alarma es trabajo inútil para el revisor humano |
| **Descarte de errores del rastreador** | De los que la hoja marca `no_es_mencion`, cuántos marcas `no` | El rastreador te pasará "gracias" y personajes de novela |
| **Datos del caso** | En los efectos adversos que los dos marcáis `si`: % de acierto en `paciente`, `sexo`, `grave`, `desenlace`, `edad` | Lo que necesitan el codificador y el informe oficial |
| **Reacciones** | Texto libre: no se compara letra a letra. Revisa a mano las de práctica, o (opcional) pásalas por el codificador MedDRA y compara los términos | "me da muchísimo sueño" y "mucho sueño" son la misma reacción |
| **Coste** | Tokens y segundos por mención | El servidor del 27B es compartido |

Da cada porcentaje con su **margen de error** (intervalo de Wilson): con 89 efectos adversos en el
examen, un 90 % puede ser en realidad algo entre el 82 % y el 95 %. Y guarda la **lista de fallos
con su texto** (en `data/`, no en git): sin ella no hay análisis.

Como modelo de código sirve [`crawler/evaluate.py`](../crawler/evaluate.py): `score` calcula los
números y `errors` lista los fallos con su texto, para el rastreador.

## 7. Lo que ya sabemos de este examen (trampas)

- **Solo el 22 % son efectos adversos** (161 de 725). La mayoría son bromas, opiniones, gente que
  la toma y está bien, preguntas… y **168 son errores del rastreador** (no hablan de la sertralina).
- **El tono de broma no anula un síntoma real**: "primer día con sertralina… solo quiero vomitar"
  es un efecto adverso. Fue el fallo más frecuente del 27B al anotar.
- **Otro fármaco o un antidepresivo sin identificar → `no` (`no_es_mencion`)**: náuseas con la
  duloxetina no son un efecto adverso de la sertralina. Fue el otro gran fallo del 27B.
- **Los síntomas de un mensaje anterior no son de este post**, salvo que el post diga "a mí
  también".
- **Retirada y falta de eficacia sí cuentan** ("al dejarla me mareaba", "no me hace nada").
- **La edad casi nunca aparece** (1 de 161); el **sexo** suele deducirse del género gramatical
  ("estoy cansada"); **grave** casi nunca consta (3 de 161) y urgencias sin ingreso no es
  hospitalización.
- **Hay posts en portugués, catalán e inglés**: el campo `idioma` lo pone la aplicación del autor y
  no siempre es correcto.

## 8. Qué contar en el informe

- Cómo funciona tu agente y por qué lo diseñaste así.
- Las cifras del **examen**, con su margen de error, y las de práctica para comparar.
- **Qué alternativas probaste** en práctica y por qué elegiste una (lo que más valora la rúbrica).
- **En qué falla**, con ejemplos (parafraseados: los posts son de personas reales).
- **Cuánto cuesta.**
- Los límites: un fármaco, una red social, una hoja de soluciones hecha por un modelo y revisada
  por una persona en los desacuerdos.

## 9. Git

Trabaja en tu rama (`triaje`); antes de empezar cada día, trae lo nuevo de `main`
(`git checkout main && git pull && git checkout triaje && git merge main`); para entregar, `git
push` de tu rama y un *pull request* en GitHub. **Nada de `data/` en git**: está en `.gitignore`.
