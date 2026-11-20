# Proyecto Simulación de Sistemas. Ciudad de domicilios multiagente

Este documento es la especificación del entorno para construirlo con Claude Code. Léelo completo antes de escribir código. La prioridad es que el entorno funcione sin gráficos, sea reproducible con semilla y deje listas las interfaces para que los agentes perciban, decidan, actúen y aprendan.

> Nota: esta es la especificación original. Las desviaciones que se tomaron al construirlo (acción
> "urgente", Q-learning por tipo de pedido, penalización por rechazo) están explicadas en el README.

## 1. Stack tecnológico

| **Pieza** | **Tecnología** | **Para qué** |
|---|---|---|
| Lenguaje | Python 3.11 | Todo el proyecto |
| Simulación basada en agentes | Mesa 3.x | Clases Model y Agent, activación de agentes, DataCollector para métricas |
| Mapa de calles | NetworkX | La ciudad como grafo, A* con pesos dinámicos por trancón y lluvia |
| Cálculo | NumPy | Tablas Q, aleatoriedad con semilla |
| Datos y gráficas | Pandas y Matplotlib | Resultados de experimentos y gráficas comparativas |
| Configuración | Archivo YAML (PyYAML) | Parámetros de cada experimento sin tocar código |
| Vista 2D de depuración | Pygame | Ver la ciudad mientras se desarrolla |
| Vista 3D para la demo | Ursina | Ciudad low poly para la sustentación (fase final) |
| Pruebas | pytest | Verificar reglas del entorno |
| Control de versiones | Git y GitHub | Trabajo en paralelo de 3 personas |

Regla de arquitectura obligatoria. La lógica (carpeta `sim/`) nunca importa nada de Pygame ni Ursina. Las vistas solo leen el estado del modelo y lo dibujan. Así se puede entrenar sin ventana, miles de pasos por segundo.

## 2. Estructura de carpetas

```
proyecto/
  config/
    base.yaml              # parámetros por defecto
    experimentos/          # un yaml por experimento
  sim/
    model.py               # CiudadModel (mesa.Model)
    ciudad.py              # construcción del grafo de calles y edificios
    reloj.py               # tiempo simulado, hora del día
    eventos.py             # lluvia y trancones
    pedidos.py             # clase Pedido y su ciclo de vida
    agentes/
      base.py              # AgenteBase con percibir, decidir, actuar
      habitante.py
      restaurante.py
      repartidor.py
    politicas/
      base.py              # interfaz Politica
      repartidor_aleatoria.py
      repartidor_cercano.py
      repartidor_qlearning.py
      habitante_egreedy.py
    metricas.py            # DataCollector y cálculos
  vistas/
    vista_2d.py            # Pygame
    vista_3d.py            # Ursina
  experimentos/
    correr.py              # corre N episodios sin gráficos y guarda CSV
    graficas.py            # genera las gráficas comparativas
  resultados/              # CSV y PNG generados (no se versionan)
  tests/
  main.py                  # punto de entrada con argumentos
  requirements.txt
  README.md
```

## 3. El entorno (CiudadModel)

### 3.1 Tiempo
Un paso (tick) equivale a 1 minuto simulado. Un día son 1440 ticks. Un episodio por defecto dura 1 día, configurable. El reloj expone la hora actual para que los agentes y eventos la usen.

### 3.2 Ciudad
Cuadrícula de N x N manzanas (por defecto 10 x 10) convertida en grafo de NetworkX. Cada intersección es un nodo con coordenadas (x, y). Cada calle es una arista con atributos `longitud` (metros), `tiempo_base` (minutos) y `factor_trancon` (1.0 normal, mayor cuando hay trancón).

Sobre los nodos se ubican edificios de tipo casa, oficina, universidad y restaurante. Los 3 restaurantes son Pizza (rápida y barata), Hamburguesas (intermedia) y Sushi (lenta y cara). Las posiciones salen de la configuración o se generan con la semilla.

### 3.3 Eventos estocásticos
Lluvia. Cada hora hay una probabilidad configurable de que empiece o pare. Con lluvia la velocidad de los repartidores baja (factor configurable, por defecto 0.7) y la probabilidad de pedir domicilio sube.

Trancones. En horas pico (por defecto 7 a 9 y 17 a 19) un porcentaje de aristas aleatorias multiplica su `factor_trancon`. Fuera de hora pico vuelven a 1.0.

Todo evento usa el generador aleatorio del modelo (`self.random` de Mesa) para que una misma semilla dé siempre el mismo resultado.

### 3.4 Propiedades del entorno (para el documento)
Multiagente, estocástico, dinámico, parcialmente observable, discreto en tiempo y secuencial. El repartidor no conoce los pedidos futuros y el habitante no conoce el tiempo real de preparación de cada restaurante, solo su experiencia.

### 3.5 Pedido
Clase con id, habitante, restaurante, repartidor asignado, precio, y marcas de tiempo `creado`, `aceptado_rest`, `listo`, `recogido`, `entregado`, `cancelado`. Estados posibles: CREADO, EN_PREPARACION, LISTO, EN_CAMINO, ENTREGADO, CANCELADO. Un pedido se cancela si pasa un tiempo máximo de espera sin entregarse (configurable, por defecto 60 minutos).

El modelo mantiene una lista de pedidos disponibles (listos o en preparación sin repartidor) que los repartidores pueden ver.

## 4. Contrato de los agentes

Todos los agentes heredan de `AgenteBase(mesa.Agent)` con este ciclo en cada tick.

```python
def step(self):
    percepcion = self.percibir()      # solo lo que el agente puede ver
    accion = self.decidir(percepcion)  # delega en su política
    self.actuar(accion)                # cambia el entorno
```

La decisión se delega a un objeto `Politica` con métodos `elegir(percepcion)` y `aprender(percepcion, accion, recompensa, nueva_percepcion)`. Esto permite cambiar de política desde el YAML sin tocar el agente, que es justo lo que se necesita para comparar en los experimentos.

### 4.1 Habitante (agente basado en utilidad)
Estado con hambre (0 a 100), energía, dinero, ubicación (casa) y actividad actual (dormir, trabajar, estudiar, ejercicio, libre) según un horario. El hambre sube por tick con una tasa que depende de la actividad. Cuando supera un umbral decide entre cocinar, salir a comer o pedir domicilio. Si pide, elige restaurante con utilidad igual a gusto menos peso_precio por precio menos peso_tiempo por tiempo esperado. Al recibir el pedido calcula su satisfacción y actualiza su estimación del restaurante (bandit epsilon greedy). Si el pedido se cancela la recompensa es negativa.

### 4.2 Restaurante (agente basado en modelo)
Cola de pedidos, capacidad de cocina (pedidos simultáneos), tiempo de preparación medio y su desviación, precio medio. Cada tick avanza la preparación y marca pedidos como LISTO. Guarda el historial de demanda por hora para que más adelante se le pueda poner un predictor.

### 4.3 Repartidor (agente basado en metas con aprendizaje)
Estado con ubicación (nodo), estado (LIBRE, YENDO_A_RESTAURANTE, YENDO_A_CLIENTE), pedido actual, ruta y ganancias. Cuando está libre percibe los pedidos disponibles y decide cuál aceptar o esperar. Las rutas se calculan con `networkx.astar_path` usando como peso tiempo_base por factor_trancon dividido por el factor de lluvia, con heurística de distancia euclidiana. Avanza por la ruta según su velocidad. Recompensa por pedido igual a pago menos costo por minuto, con penalización si el pedido termina cancelado.

Estado para Q-learning (discreto) con distancia al restaurante en rangos, franja horaria, lluvia sí o no, y cantidad de pedidos disponibles en rangos. Las acciones son aceptar el más cercano, aceptar el mejor pagado, o esperar.

Políticas a implementar para comparar: aleatoria, más cercano y Q-learning.

## 5. Métricas (DataCollector)

Por tick o por episodio se guardan tiempo promedio de entrega, pedidos entregados, pedidos cancelados, ganancia por repartidor, satisfacción promedio de habitantes, elección de restaurante por habitante a lo largo del tiempo, y recompensa acumulada por episodio del Q-learning (curva de aprendizaje).

## 6. Ejecución

```
python main.py --config config/base.yaml --vista ninguna --episodios 500 --semilla 42
python main.py --config config/base.yaml --vista 2d
python main.py --config config/base.yaml --vista 3d
```

Con `--vista ninguna` corre a máxima velocidad y guarda resultados en `resultados/`. La tabla Q se guarda en un archivo para reusarla en la demo con vista.

## 7. Experimentos previstos

1. Política del repartidor aleatoria contra más cercano contra Q-learning.
2. 1, 3 y 5 repartidores contra el tiempo de espera de los habitantes.
3. Con lluvia y sin lluvia.
4. Evolución de las preferencias de los habitantes en el tiempo.

Cada experimento es un YAML en `config/experimentos/` y se corre varias veces con distintas semillas para reportar promedio y desviación.

## 8. Orden de construcción para Claude Code

Fase 1. Ciudad, reloj, eventos y pedidos con pruebas en pytest. Modelo que corre sin agentes.
Fase 2. Agentes con políticas simples (aleatoria y más cercano) y el ciclo completo de un pedido funcionando de punta a punta.
Fase 3. Vista 2D en Pygame para verificar visualmente.
Fase 4. Q-learning del repartidor y epsilon greedy del habitante, con curva de aprendizaje.
Fase 5. Script de experimentos y gráficas.
Fase 6. Vista 3D en Ursina reutilizando el mismo modelo.

Criterio para pasar de fase. Las pruebas pasan, la simulación de un día corre sin errores con semilla fija y da el mismo resultado dos veces seguidas.

## 9. Convenciones
Código y nombres en español, con docstrings cortos que expliquen el porqué de cada decisión (el profesor pregunta por qué, no qué). Todos los números mágicos van en el YAML. Nada de `random` de Python suelto, todo con el generador del modelo.
