# Ciudad de domicilios multiagente con aprendizaje por refuerzo

Proyecto de Simulación de Sistemas. Una ciudad donde **habitantes** con hambre piden
domicilios, **restaurantes** los cocinan y **repartidores** deciden qué pedido aceptar.
El repartidor aprende con **Q-learning** y el habitante aprende qué restaurante le
conviene con un **bandit ε-greedy**.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CarlZeichenfolge/simulacion-domicilios-rl/blob/main/notebooks/informe_domicilios.ipynb)

**Informe ejecutable:** [notebooks/informe_domicilios.ipynb](notebooks/informe_domicilios.ipynb) — el informe completo con
mapa, animación del almuerzo, entrenamiento de Q-learning y los 4 experimentos en versión rápida (~2-3 min en Colab).
El enlace de Colab solo funciona si el repositorio es público; si es privado, el notebook explica cómo subir el `.zip`.

## Instalación

```powershell
py -3.12 -m venv $env:USERPROFILE\venvs\domicilios-rl     # fuera de OneDrive
& $env:USERPROFILE\venvs\domicilios-rl\Scripts\Activate.ps1
pip install -r requirements.txt
```

> El entorno virtual va fuera de OneDrive porque OneDrive bloquea archivos mientras pip instala.
> En VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → elegir ese `python.exe`.

## Uso

```powershell
pytest -q                                                    # pruebas (deben pasar todas)
python main.py --vista ninguna --episodios 300               # entrena Q-learning, guarda resultados/tabla_q.npz
python main.py --vista 2d                                    # demo con Pygame (usa la tabla entrenada)
python main.py --vista 2d --politica cercano                 # demo con otra política
python -m experimentos.correr config/experimentos/01_politicas.yaml   # un experimento
python -m experimentos.correr (Get-ChildItem config/experimentos/*.yaml)   # todos
python -m experimentos.prueba_final                          # prueba final del exp. 1 con días nunca vistos
python -m experimentos.prediccion                            # predictor supervisado (regresión + clasificación)
python -m experimentos.graficas                              # PNG en resultados/
```

Controles de la vista 2D: `ESPACIO` pausa, `↑/↓` velocidad, `D` un minuto, `ESC` salir.
Vista 3D (`--vista 3d`): clic derecho + arrastrar gira la cámara, rueda hace zoom, `ESPACIO` pausa, `↑/↓` velocidad.

## Estructura

| Carpeta | Qué hay |
|---|---|
| `config/base.yaml` | **Todos** los parámetros. Ningún número mágico vive en el código. |
| `config/experimentos/` | Un YAML por experimento: solo lo que cambia respecto a la base. |
| `sim/` | La lógica. **Nunca importa Pygame ni Ursina**: así se entrena sin ventana. |
| `sim/agentes/` | `Habitante`, `Restaurante`, `Repartidor` (ciclo percibir → decidir → actuar). |
| `sim/politicas/` | Políticas intercambiables: aleatoria, cercano, urgente, Q-learning, ε-greedy. |
| `vistas/` | Vista 2D (Pygame) y 3D (Ursina). Solo leen el modelo. |
| `experimentos/` | Correr experimentos en paralelo, prueba final, predictor supervisado y gráficas. |
| `tests/` | pytest: reglas del entorno, ciclo del pedido, políticas y reproducibilidad. |

## El entorno

- **Tiempo**: 1 tick = 1 minuto, 1 día = 1440 ticks.
- **Ciudad**: cuadrícula 10×10 manzanas como grafo de NetworkX (cuadras de 400 m). Edificios:
  casas, oficinas, una universidad y 3 restaurantes (Pizza rápida y barata, Hamburguesas
  intermedia, Sushi lento y caro).
- **Lluvia**: cadena de Markov por hora. Baja la velocidad al 70 % y castiga la utilidad de salir a
  comer, por lo que **la gente pide más domicilios sin que lo impongamos desde afuera**.
- **Trancones**: en hora pico (7-9 y 17-19) el 30 % de las calles multiplica su tiempo ×1.5 a ×3.
  Los repartidores replanifican con A*.
- **Pedido**: CREADO → EN_PREPARACION → LISTO → EN_CAMINO → ENTREGADO, o CANCELADO a los 60 min.
- **Propiedades**: multiagente, estocástico, dinámico, parcialmente observable, discreto en el tiempo y secuencial.

## Los agentes

| Agente | Tipo | Percibe | Decide | Aprende |
|---|---|---|---|---|
| Habitante | Basado en utilidad | Hambre, energía, dinero, actividad, lluvia, menú | Cocinar, salir o pedir (y a quién) | Bandit ε-greedy: satisfacción real de cada restaurante |
| Restaurante | Basado en modelo | Su cola y sus fogones | Qué empezar a cocinar (FIFO con capacidad) | Guarda la demanda por hora (para un predictor futuro) |
| Repartidor | Basado en metas + aprendizaje | Pedidos sin dueño, distancia, edad, pago, hora, lluvia | Qué pedido aceptar o esperar; ruta con A* | Q-learning tabular (SMDP) |

### Q-learning del repartidor

- **Estado (contexto)**: franja horaria (6), lluvia (2), carga de pedidos disponibles (4).
- **Acción**: el *tipo* de pedido que acepta = distancia al restaurante (4) × distancia de entrega (3)
  × edad del pedido (4) × ¿está listo? (2), o **esperar**.
- **Recompensa**: `pago − 80 $/min × minutos`, −5000 si el pedido se le cancela, −4000 por rechazar
  (esperar habiendo pedidos), escalada a miles.
- **Actualización SMDP**: `Q ← Q + α [r + γ^k max Q(s′) − Q]`, con k = minutos que duró la acción.
- Una tabla compartida por todos los repartidores; ε decae por episodio.

### Decisiones que salieron de los experimentos (para la sustentación)

1. **"Más cercano" no fue la mejor heurística.** Deja vencer los pedidos lejanos. Por eso se agregó
   la heurística **"urgente"** (el más viejo) como otra línea base.
2. **Acciones = reglas fijas no funcionó.** Con acciones "cercano / mejor pagado / esperar" y un estado
   agregado, el agente no distinguía un pedido salvable de uno perdido. Describir el pedido con sus
   características hizo que aprendiera de verdad.
3. **Recompensa mal alineada.** Sin castigo por rechazar, el agente maximiza *su* ganancia escogiendo
   solo pedidos cómodos y la cancelación de la ciudad sube de ~15 % a ~28 %. Es lo mismo que pasa con
   los repartidores reales; por eso las apps miden la tasa de aceptación. El experimento 1 incluye
   la variante `qlearning_sin_rechazo` para mostrarlo.
4. **Una recompensa "cooperativa" (multa por cada cancelación de la ciudad durante la acción) fue peor**:
   como las multas se acumulan por minuto, esperar (3 min) acumulaba menos que entregar (30 min) y el
   agente aprendía a no hacer nada. Se descartó.
5. **Calibración de "salir a comer".** Al principio casi nadie salía (utilidad ≈ −0.8), así que la
   lluvia no tenía a quién quitarle salidas y el aumento de pedidos venía solo de re-pedir tras
   cancelaciones. Con gusto por salir propio de cada persona (media 7.5) la gente sale ~7 veces al
   día en seco y casi nunca con lluvia: el mecanismo ahora se ve en el experimento 3.
6. **Q-learning no le gana a "urgente" (FIFO), y se intentó.** Optimiza bien su objetivo (su recompensa
   supera a todas las heurísticas y entrega más rápido), pero en cancelaciones solo empata con aleatoria y
   cercano (13.0 % vs 14.0 %, dentro del ruido entre semillas), y "atender primero al más viejo" cancela
   bastante menos (9.1 %). Se probó (3 semillas cada uno):
   - incentivo de pago que sube con la espera (`pedidos.incentivo_por_minuto`, queda en 0): Q 12.4 % vs urgente 8.2 %;
   - agregar al tipo de pedido "¿es el más viejo?", para que la tabla pueda representar exactamente
     "urgente": Q 13.8-15.3 % vs 8.2 % (más tipos = aprende más lento). Se revirtió.

   Explicación: la ventaja de FIFO es **del sistema** (menos cancelaciones para todos), pero la
   recompensa del repartidor es **individual**. Un agente que optimiza su propia ganancia no tiene por
   qué descubrir la regla que mejor sirve a la ciudad. Por eso en sistemas reales la asignación la
   hace la plataforma (un despachador central) y no cada repartidor por su cuenta.

## Experimentos

| # | YAML | Pregunta |
|---|---|---|
| 1 | `01_politicas.yaml` | ¿Q-learning le gana a aleatoria, cercano y urgente? (+ curva de aprendizaje) |
| 2 | `02_cantidad_repartidores.yaml` | 1, 3 y 5 repartidores contra la espera de los habitantes |
| 3 | `03_lluvia.yaml` | Sin lluvia, lluvia estocástica y lluvia todo el día |
| 4 | `04_preferencias.yaml` | 14 días: ¿cambian las preferencias con el aprendizaje? (control: α = 0) |

Cada experimento se corre con 5 semillas. Los episodios de evaluación usan **las mismas semillas para
todas las variantes** (comparación pareada) y Q-learning se evalúa sin explorar ni aprender.

### Validación y prueba

Hay tres bloques de semillas que nunca se cruzan (`DESPLAZAMIENTO_FASE` en `experimentos/correr.py`):

| Bloque | Para qué |
|---|---|
| Entrenamiento | Q-learning aprende (300 días por semilla) |
| Evaluación = **validación** | Con estos 30 días por semilla se tomaron las decisiones de diseño (recompensa, estado, calibración) |
| **Prueba** | Se corrió **una sola vez** al final, con las tablas Q congeladas (`python -m experimentos.prueba_final`) |

Prueba final (5 semillas × 30 días nunca vistos):

| Política | Cancelación | Tiempo de entrega (min) | Recompensa por día |
|---|---|---|---|
| Urgente | **8.8 % ± 0.5** | 36.2 | 209.5 |
| Q-learning | 11.1 % ± 1.8 | 33.9 | **235.8** |
| Aleatoria | 11.1 % ± 1.8 | 34.5 | 217.0 |
| Más cercano | 12.6 % ± 1.6 | 35.1 | 197.2 |
| Q-learning sin castigo por rechazo | 21.3 % ± 3.0 | 33.5 | 231.1 |

Las conclusiones de la validación se sostienen con días nuevos.

## Predictor supervisado (`experimentos/prediccion.py`)

Cada pedido guarda una *foto* de lo que se sabía al crearlo (`CiudadModel._foto_pedido`). Con eso se
responden dos preguntas sobre el mismo dataset: **¿cuánto tardará?** (regresión) y **¿se cancelará?**
(clasificación). Partición por día (60 / 20 / 20), Pipeline ajustado solo con entrenamiento, hiperparámetros
elegidos en validación y una sola medición en prueba.

| Regresión (prueba) | MAE (min) | R² |
|---|---|---|
| Línea base (media) | 8.83 | −0.01 |
| Regresión lineal | 5.58 | 0.57 |
| KNN (K = 15) | 5.44 | 0.57 |
| **Random Forest** | **5.24** | **0.61** |

| Clasificación (prueba) | Exactitud | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Línea base ("siempre no") | 0.91 | 0.00 | 0.00 | 0.50 |
| Regresión logística | 0.89 | **0.76** | 0.56 | 0.92 |
| KNN (K = 5) | 0.92 | 0.37 | 0.46 | 0.85 |
| **Random Forest** | 0.92 | 0.73 | **0.62** | **0.94** |

La línea base acierta 91 % sin detectar ni una cancelación: con clases desbalanceadas la exactitud engaña.

## Por qué aprendizaje por refuerzo (y no supervisado) para el repartidor

- Supervisado exige la respuesta correcta de cada ejemplo durante el entrenamiento. Para "¿qué pedido
  acepto?" nadie la conoce: solo se sabe después si le fue bien o mal. Eso es refuerzo (Géron, cap. 1 y el
  capítulo de aprendizaje por refuerzo, donde está Q-learning).
- No es no supervisado: no se buscan grupos, se buscan decisiones.
- Donde sí existe la respuesta correcta (tiempo de entrega, cancelación) usamos supervisado: el predictor de arriba.
- Por qué tabla y no red neuronal: 48 × 97 = 4.656 valores, se entrena en minutos y se puede leer celda por
  celda. Una red se justifica con entradas complejas o enormes; aquí no hace falta.

## Reproducibilidad

Todo lo aleatorio usa `model.random` (sembrado por Mesa con `rng=semilla`). El mapa usa su propia
semilla fija (`ciudad.semilla_mapa`) para que el escenario sea el mismo entre episodios.
`tests/test_agentes.py::test_misma_semilla_mismo_resultado` lo verifica para cada política.
