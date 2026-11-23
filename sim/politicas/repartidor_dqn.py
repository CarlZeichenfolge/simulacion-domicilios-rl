"""Deep Q-Network: la tabla Q reemplazada por una red neuronal (PyTorch).

Qué cambia respecto a la tabla (repartidor_qlearning.py):
  - La tabla parte el mundo en casillas ("3 a 6 cuadras") y aprende un número
    por casilla. La red recibe los valores reales (5.7 cuadras, 23 minutos,
    $6.300) y aprende una FUNCIÓN continua Q(situación, pedido). Dos pedidos
    parecidos reciben valores parecidos aunque nunca se hayan visto: generaliza.
  - La entrada describe la pareja (situación, acción) y la salida es UN número.
    Por qué así y no "una salida por acción": las acciones son los pedidos que
    haya en ese minuto (cantidad variable), así que se evalúa cada candidato
    con la misma red y se escoge el de mayor Q.

Las dos piezas que hacen estable a DQN (Mnih et al., 2015; Géron, cap. de RL):
  - Experience replay: las transiciones se guardan en una memoria y se
    entrena con minilotes al azar. Sin esto, la red aprende de decisiones
    seguidas que se parecen mucho entre sí y se "olvida" del resto.
  - Red objetivo: una copia congelada de la red calcula r + γ^k·max Q(s',a').
    Si el objetivo saliera de la misma red que se está moviendo, sería como
    perseguir la propia sombra.

La actualización es la misma idea que la tabla, escrita como regresión:
  pérdida = Huber( Q_red(s,a) , r + γ^k · max_a' Q_objetivo(s',a') )
y se minimiza con Adam (descenso de gradiente). En la tabla, el paso
Q ← Q + α(objetivo − Q) es exactamente un paso de gradiente de (objetivo − Q)²/2.
"""
from __future__ import annotations

import copy
import math
import random
from collections import deque
from pathlib import Path

import numpy as np

from sim.politicas.base import Politica, accion_aceptar, accion_esperar

# Orden de las 12 entradas de la red (se guarda con el modelo para leerlo después).
ENTRADAS = ["hora_sen", "hora_cos", "lluvia", "pedidos_disponibles",
            "cuadras_al_restaurante", "cuadras_de_entrega", "edad_pedido", "listo", "pago",
            "es_el_mas_viejo", "es_el_mas_cercano", "es_esperar"]
N_CONTEXTO = 4   # las primeras 4 entradas describen la situación; las demás, el pedido
N_PEDIDO = 7     # entradas propias de cada pedido (sin contar es_esperar)


def contexto_y_pedidos(percepcion: dict, escalas: dict) -> tuple[list[float], list[list[float]]]:
    """Ingeniería de características (Frente 9 del curso), compartida por la red y por Gymnasium.

      - La hora va como seno y coseno: así las 23 h y las 0 h quedan cerca.
      - Todo se divide por una escala para que quede cerca de [0, 1] (Frente 7).
      - "es el más viejo" y "es el más cercano" le permiten a la red
        representar las reglas urgente y cercano si le convienen.
    """
    e = escalas
    pedidos = percepcion["pedidos"]
    angulo = 2 * math.pi * percepcion["hora"] / 24
    contexto = [math.sin(angulo), math.cos(angulo), float(percepcion["lluvia"]), len(pedidos) / e["pedidos"]]
    filas = []
    if pedidos:
        edad_max = max(p["edad"] for p in pedidos)
        dist_min = min(p["distancia"] for p in pedidos)
        filas = [[p["distancia"] / e["cuadras"], p["distancia_entrega"] / e["cuadras"],
                  p["edad"] / e["minutos"], float(p["listo"]), p["pago"] / e["pago"],
                  float(p["edad"] == edad_max), float(p["distancia"] == dist_min)] for p in pedidos]
    return contexto, filas


def filas_candidatas(contexto: list[float], pedidos: list[list[float]]) -> np.ndarray:
    """Una fila por acción posible: cada pedido y, al final, esperar."""
    filas = [contexto + p + [0.0] for p in pedidos]
    filas.append(contexto + [0.0] * N_PEDIDO + [1.0])
    return np.asarray(filas, dtype=np.float32)


def caracteristicas(percepcion: dict, escalas: dict) -> np.ndarray:
    return filas_candidatas(*contexto_y_pedidos(percepcion, escalas))


def construir_red(n_entradas: int, capas_ocultas: list[int], semilla: int):
    """MLP con ReLU e inicialización He (la recomendada para ReLU en RN-01)."""
    import torch
    from torch import nn

    capas, anterior = [], n_entradas
    for ancho in capas_ocultas:
        capas += [nn.Linear(anterior, ancho), nn.ReLU()]
        anterior = ancho
    capas.append(nn.Linear(anterior, 1))  # salida lineal: Q puede ser negativo
    with torch.random.fork_rng(devices=[]):  # pesos iniciales reproducibles sin tocar el azar global
        torch.manual_seed(semilla)
        red = nn.Sequential(*capas)
        for capa in red:
            if isinstance(capa, nn.Linear):
                nn.init.kaiming_normal_(capa.weight, nonlinearity="relu")
                nn.init.zeros_(capa.bias)
    return red


class PoliticaRepartidorDQN(Politica):
    """Misma interfaz que la tabla: elegir() decide y aprender() recibe la
    recompensa cuando la acción termina (SMDP, descuento γ^minutos)."""

    aprende = True
    extension = ".pt"

    def __init__(self, cfg: dict):
        super().__init__()
        import torch

        torch.set_num_threads(1)  # red pequeña: un hilo es más rápido y da resultados idénticos
        self.torch = torch
        self.cfg = cfg
        self.gamma = cfg["gamma"]
        self.epsilon = cfg["epsilon_inicial"]
        self.epsilon_min = cfg["epsilon_min"]
        self.decaimiento = cfg["decaimiento_epsilon"]
        self.escalas = cfg["escalas"]
        self.lote = cfg["lote"]
        self.minimo = cfg["minimo_para_entrenar"]
        self.entrenar_cada = cfg["entrenar_cada"]
        self.actualizar_objetivo_cada = cfg["actualizar_objetivo_cada"]
        self.max_norma = cfg["max_norma_gradiente"]

        self.red = construir_red(len(ENTRADAS), cfg["capas_ocultas"], cfg["semilla"])
        self.objetivo = copy.deepcopy(self.red)
        self.optimizador = torch.optim.Adam(self.red.parameters(), lr=cfg["tasa_aprendizaje"])
        self.perdida = torch.nn.SmoothL1Loss()  # Huber: como MSE, pero un error enorme no la desestabiliza
        self.memoria: deque = deque(maxlen=cfg["memoria"])
        self.rng_memoria = random.Random(cfg["semilla"])  # separado del modelo: no altera la simulación
        self.decisiones = 0
        self.pasos_gradiente = 0
        self.perdidas_episodio: list[float] = []
        self.historial_perdida: list[float] = []  # pérdida media por episodio (curva de entrenamiento)
        self.explorar = True

    # -------------------------------------------------------- características
    def caracteristicas(self, percepcion: dict) -> np.ndarray:
        return caracteristicas(percepcion, self.escalas)

    def valores_q(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.red(self.torch.from_numpy(x)).squeeze(1).numpy()

    # ----------------------------------------------------------------- decidir
    def elegir(self, percepcion: dict) -> dict:
        x = self.caracteristicas(percepcion)
        if self.explorar and self.rng.random() < self.epsilon:
            i = self.rng.randrange(len(x))
        else:
            i = int(np.argmax(self.valores_q(x)))
        pedidos = percepcion["pedidos"]
        accion = accion_esperar() if i == len(pedidos) else accion_aceptar(pedidos[i], i)
        accion["x"] = x[i]  # la fila exacta que se evaluó: es lo que se guarda en la memoria
        return accion

    def valorar(self, percepcion: dict) -> list[float]:
        return [float(v) for v in self.valores_q(self.caracteristicas(percepcion))]

    # ---------------------------------------------------------------- aprender
    def aprender(self, percepcion, accion, recompensa, nueva_percepcion, duracion=1):
        self.recordar(accion["x"], recompensa, duracion, self.caracteristicas(nueva_percepcion))

    def recordar(self, x: np.ndarray, recompensa: float, duracion: int, siguientes: np.ndarray) -> None:
        """Guarda una transición (fila elegida, r, γ^k, candidatos de s') y entrena cada tanto.

        Separado de aprender() para que un agente de Gymnasium, que solo ve
        vectores de observación y no percepciones de Mesa, entrene la misma red.
        """
        if not self.explorar:
            return
        self.memoria.append((x, recompensa, self.gamma ** duracion, siguientes))
        self.decisiones += 1
        if len(self.memoria) >= self.minimo and self.decisiones % self.entrenar_cada == 0:
            self._paso_de_gradiente()

    def _paso_de_gradiente(self) -> None:
        torch = self.torch
        lote = self.rng_memoria.sample(self.memoria, self.lote)
        x = torch.from_numpy(np.stack([t[0] for t in lote]))
        r = torch.tensor([t[1] for t in lote], dtype=torch.float32)
        descuento = torch.tensor([t[2] for t in lote], dtype=torch.float32)
        siguientes = [t[3] for t in lote]
        with torch.no_grad():
            # Todos los candidatos de s' en un solo pase por la red objetivo; luego el max de cada uno.
            q_sig = self.objetivo(torch.from_numpy(np.concatenate(siguientes))).squeeze(1)
            maximos = torch.stack([q.max() for q in torch.split(q_sig, [len(s) for s in siguientes])])
            objetivo = r + descuento * maximos
        prediccion = self.red(x).squeeze(1)
        perdida = self.perdida(prediccion, objetivo)
        self.optimizador.zero_grad()
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(self.red.parameters(), self.max_norma)
        self.optimizador.step()
        self.pasos_gradiente += 1
        self.perdidas_episodio.append(perdida.item())
        if self.pasos_gradiente % self.actualizar_objetivo_cada == 0:
            self.objetivo.load_state_dict(self.red.state_dict())

    def fin_episodio(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decaimiento)
        media = float(np.mean(self.perdidas_episodio)) if self.perdidas_episodio else float("nan")
        self.historial_perdida.append(media)
        self.perdidas_episodio = []

    # -------------------------------------------------------------- persistencia
    def guardar(self, ruta: str | Path) -> None:
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({"red": self.red.state_dict(), "epsilon": self.epsilon,
                         "capas_ocultas": list(self.cfg["capas_ocultas"]), "entradas": ENTRADAS,
                         "historial_perdida": self.historial_perdida}, ruta)

    def cargar(self, ruta: str | Path) -> None:
        datos = self.torch.load(ruta, weights_only=True)
        if datos["entradas"] != ENTRADAS or datos["capas_ocultas"] != list(self.cfg["capas_ocultas"]):
            raise ValueError("La red guardada no coincide con las entradas o capas del YAML actual")
        self.red.load_state_dict(datos["red"])
        self.objetivo.load_state_dict(datos["red"])
        self.epsilon = float(datos["epsilon"])
        self.historial_perdida = list(datos["historial_perdida"])
