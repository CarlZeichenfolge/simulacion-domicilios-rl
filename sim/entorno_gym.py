"""La ciudad de domicilios como entorno Gymnasium (la misma interfaz que CartPole o MuJoCo).

    env = EntornoDomicilios()
    obs, info = env.reset(seed=0)
    while True:
        obs, r, terminado, truncado, info = env.step(accion)

Quién es el agente: el repartidor 0. Los demás siguen una regla fija
(config gym.politica_otros), así que para el agente son parte del entorno,
igual que la lluvia o los trancones.

Qué es un "paso": NO es un minuto, es una DECISIÓN. La simulación corre
sola minuto a minuto (habitantes, cocinas, motos) y solo se detiene cuando
el repartidor 0 está libre y hay pedidos. Por qué: en los minutos en que va
manejando no hay nada que decidir; pedirle una acción en cada minuto sería
llenar el aprendizaje de pasos vacíos. Cada paso informa en info["minutos"]
cuánto duró, para quien quiera descontar con γ^minutos (SMDP).

Observación (vector de 4 + 8·K números, K = gym.max_pedidos):
  [0:4]          contexto: seno y coseno de la hora, lluvia, pedidos disponibles
  [4:4+7K]       7 números por casilla de pedido (los mismos que ve la red DQN)
  [4+7K:4+8K]    máscara: 1 si la casilla tiene un pedido de verdad
Acción: Discrete(K+1). 0..K-1 = aceptar el pedido de esa casilla, K = esperar.
Una casilla vacía se trata como esperar (y lo marca info["accion_valida"]).

Recompensa: lo que cambia la recompensa acumulada del repartidor 0 (pago
menos costos, castigos por cancelación o por rechazar), en miles de pesos.
Fin: el día termina (truncated=True; es un límite de tiempo, no un estado final).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.config import cargar_config, fusionar
from sim.model import CiudadModel
from sim.politicas.base import accion_aceptar, accion_esperar
from sim.politicas.externa import PoliticaExterna
from sim.politicas.repartidor_dqn import N_CONTEXTO, N_PEDIDO, contexto_y_pedidos


class EntornoDomicilios(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None, politica_otros: str | None = None):
        super().__init__()
        self.config = config or cargar_config(cambios={"metricas": {"recolectar_cada": 0}})
        g = self.config["gym"]
        self.k = g["max_pedidos"]
        self.politica_otros = politica_otros or g["politica_otros"]
        self.escalas = self.config["repartidores"]["dqn"]["escalas"]
        self.limite = g["limite_observacion"]
        n = N_CONTEXTO + self.k * (N_PEDIDO + 1)
        self.observation_space = spaces.Box(-1.0, self.limite, shape=(n,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.k + 1)
        self.modelo: CiudadModel | None = None
        self.externo = None
        self._percepcion: dict | None = None
        self._terminado = True

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """`options={"semilla_dia": s}` fija el día exacto (lluvia, pedidos): así
        se puede evaluar en los mismos días que el resto de experimentos."""
        super().reset(seed=seed)
        semilla_dia = (options or {}).get("semilla_dia")
        if semilla_dia is None:
            semilla_dia = int(self.np_random.integers(2**31 - 1))
        cfg = fusionar(self.config, {"repartidores": {"politica": self.politica_otros}})
        self.modelo = CiudadModel(cfg, semilla=semilla_dia)
        self.externo = self.modelo.repartidores[0]
        self.externo.politica = PoliticaExterna()
        self._minutos = self._recorrer()
        self._terminado = False
        self._avanzar()
        return self._observacion(), self._info()

    def step(self, accion):
        if self._terminado:
            raise RuntimeError("El día ya terminó: llama a reset()")
        pedidos = self._percepcion["pedidos"][:self.k]
        accion = int(accion)
        valida = accion < len(pedidos) or accion == self.k
        if accion < len(pedidos):
            self.externo.politica.accion = accion_aceptar(pedidos[accion], accion)
        else:
            self.externo.politica.accion = accion_esperar()
        antes, tick = self.externo.recompensa_acumulada, self.modelo.reloj.tick
        self._avanzar()
        recompensa = self.externo.recompensa_acumulada - antes
        info = self._info()
        info.update(minutos=self.modelo.reloj.tick - tick, accion_valida=valida)
        return self._observacion(), float(recompensa), False, self._terminado, info

    @property
    def dia_terminado(self) -> bool:
        return self._terminado

    def action_masks(self) -> np.ndarray:
        """Acciones legales ahora (convención de MaskablePPO): pedidos visibles y esperar."""
        mascara = np.zeros(self.k + 1, dtype=bool)
        mascara[:min(self.k, len(self._percepcion["pedidos"])) if self._percepcion else 0] = True
        mascara[self.k] = True
        return mascara

    # ------------------------------------------------------------- por dentro
    def _recorrer(self):
        while self.modelo.running:
            yield from self.modelo.minuto(externo=self.externo)

    def _avanzar(self) -> None:
        """Corre la simulación hasta que el repartidor 0 tenga que decidir otra vez."""
        try:
            next(self._minutos)
            self._percepcion = self.externo.percibir()
        except StopIteration:
            self._terminado = True
            self._percepcion = None

    def _observacion(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        if self._percepcion is None:
            return obs
        contexto, pedidos = contexto_y_pedidos(self._percepcion, self.escalas)
        obs[:N_CONTEXTO] = contexto
        inicio_mascara = N_CONTEXTO + self.k * N_PEDIDO
        for i, fila in enumerate(pedidos[:self.k]):
            obs[N_CONTEXTO + i * N_PEDIDO:N_CONTEXTO + (i + 1) * N_PEDIDO] = fila
            obs[inicio_mascara + i] = 1.0
        return np.clip(obs, -1.0, self.limite)

    def _info(self) -> dict:
        return {"minuto": self.modelo.reloj.tick, "mascara_acciones": self.action_masks()}


ID_ENTORNO = "Domicilios-v0"
if ID_ENTORNO not in gym.registry:  # gymnasium.make("Domicilios-v0"), igual que "CartPole-v1"
    gym.register(id=ID_ENTORNO, entry_point="sim.entorno_gym:EntornoDomicilios")


def separar_observacion(obs: np.ndarray, k: int) -> tuple[list[float], list[list[float]]]:
    """Inverso de _observacion: (contexto, pedidos visibles). Lo usa el agente DQN de Gymnasium."""
    contexto = obs[:N_CONTEXTO].tolist()
    inicio_mascara = N_CONTEXTO + k * N_PEDIDO
    pedidos = [obs[N_CONTEXTO + i * N_PEDIDO:N_CONTEXTO + (i + 1) * N_PEDIDO].tolist()
               for i in range(k) if obs[inicio_mascara + i] > 0.5]
    return contexto, pedidos
