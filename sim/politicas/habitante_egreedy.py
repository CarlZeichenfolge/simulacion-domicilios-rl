"""Bandit epsilon-greedy del habitante para escoger restaurante."""
from __future__ import annotations

from sim.politicas.base import Politica


class PoliticaHabitanteEGreedy(Politica):
    """Cada restaurante es un "brazo". Q[r] estima la satisfacción de pedirle a r,
    es decir gusto - peso_precio*precio - peso_tiempo*tiempo, pero con el
    tiempo y el precio REALES que ha vivido, no los que le prometen.

    Por qué bandit y no Q-learning completo: la elección de restaurante no
    cambia el estado futuro del habitante (pedir pizza hoy no cambia el
    menú de mañana); un problema sin estado es justamente un bandit.

    Por qué alpha constante y no promedio simple: el tiempo de un restaurante
    cambia con la lluvia, los trancones y la cola; el paso constante le da
    más peso a lo reciente y sigue esos cambios.
    """

    aprende = True

    def __init__(self, cfg: dict, valores_iniciales: dict[str, float]):
        super().__init__()
        self.epsilon = cfg["epsilon"]
        self.alpha = cfg["alpha"]
        self.q = dict(valores_iniciales)
        self.conteos = {nombre: 0 for nombre in valores_iniciales}

    def elegir(self, percepcion: dict) -> dict:
        opciones = percepcion["restaurantes_asequibles"]
        if not opciones:
            return {"tipo": "ninguno"}
        if self.rng.random() < self.epsilon:
            nombre = self.rng.choice(opciones)
        else:
            nombre = max(opciones, key=lambda r: self.q[r])
        return {"tipo": "restaurante", "nombre": nombre}

    def mejor_valor(self, opciones: list[str]) -> float:
        return max(self.q[r] for r in opciones)

    def aprender(self, percepcion, accion, recompensa, nueva_percepcion, duracion=1):
        nombre = accion["nombre"]
        self.conteos[nombre] += 1
        self.q[nombre] += self.alpha * (recompensa - self.q[nombre])
