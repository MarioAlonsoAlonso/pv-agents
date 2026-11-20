# Cómo trabajamos en el repositorio

Para todo el equipo. Cada persona trabaja en **su rama** y en **su carpeta**; lo terminado se
junta en `main` con un *pull request*.

## Quién toca qué

| Rama | Carpeta | Quién |
|---|---|---|
| `rastreador` | `crawler/` | rastreador |
| `triaje` | `triage/` | triaje |
| `main` | todo | nadie directamente: solo entra lo que llega por *pull request* |

Las carpetas comunes (`common/`, `agent/`, `README.md`, `pyproject.toml`…) se pueden tocar, pero
**avisad antes al resto**: si dos personas cambian las mismas líneas, git no sabe cuál quedarse.

## La primera vez

1. Pedid acceso al repositorio a su dueño (GitHub → *Settings* → *Collaborators*).
2. Descargad el proyecto, entrad en vuestra rama e instalad las dependencias:

   ```bash
   git clone https://github.com/MarioAlonsoAlonso/pv-agents.git
   cd pv-agents
   git checkout triaje          # o rastreador: vuestra rama
   uv sync --extra agent
   cp .env.example .env         # y rellenad la clave del 27B (os la pasa el equipo, nunca por git)
   ```

3. Cread la carpeta `data/` y meted ahí los datos que os pasen (ver abajo).

## Cada día

Antes de empezar, traed lo nuevo de `main` a vuestra rama:

```bash
git checkout main
git pull
git checkout triaje            # vuestra rama
git merge main
```

Mientras trabajáis, guardad y subid vuestros cambios:

```bash
git status                     # qué ha cambiado; comprobad que no aparece nada de data/ ni .env
git add triage/                # solo lo vuestro
git commit -m "triage: qué habéis hecho"
git push
```

## Cuando algo esté terminado

1. Traed lo nuevo de `main` otra vez (los cuatro comandos de arriba) y comprobad que los tests
   pasan: `uv run pytest`.
2. En GitHub, botón **Compare & pull request**, de vuestra rama a `main`. Explicad qué cambia.
3. Otra persona del equipo lo mira y lo acepta (**Merge**).
4. Seguid trabajando en vuestra rama; antes, `git merge main` como cada día.

**Si git dice `CONFLICT`** al hacer `merge`: abrid los ficheros que nombra, buscad las marcas
`<<<<<<<`, `=======`, `>>>>>>>`, dejad la versión buena (o una mezcla), borrad las marcas y
haced `git add` + `git commit`. Si no está claro, preguntad a quien hizo el otro cambio.

## Lo que nunca va a git

- **`data/`**: posts de Bluesky (personas reales, datos de salud), hojas de soluciones y
  resultados con texto. Está en `.gitignore`; se pasa por un canal privado.
- **`.env`**: la clave del servidor del 27B.
- Pesos de modelos y ficheros grandes.

Si alguno aparece en `git status` como listo para subir, **no hagáis commit** y avisad.
