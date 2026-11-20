# Instrucciones para Claude Code en este repo

Lee `ESPECIFICACION_ENTORNO.md` (la especificación original) y `README.md` (lo que de verdad se
construyó y por qué se desvió de la especificación en algunos puntos).

## Reglas
- `sim/` nunca importa pygame ni ursina. Las vistas solo leen el modelo.
- Ningún número mágico en el código: todo parámetro va en `config/base.yaml`.
- Nada de `random` suelto: usar `self.model.random` (o el `rng` que recibe la clase).
- Código, nombres y docstrings en español. Los docstrings explican el **porqué**, no el qué.
- Después de cualquier cambio: `pytest -q` debe pasar y la simulación con semilla fija debe dar
  el mismo resultado dos veces.

## Entorno
- Python 3.12 en `%USERPROFILE%\venvs\domicilios-rl` (fuera de OneDrive).
- Pasar código Python por PowerShell rompe las comillas: escribir un script y ejecutarlo.
