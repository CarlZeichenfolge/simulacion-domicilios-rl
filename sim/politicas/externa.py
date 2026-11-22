"""Política "de afuera": la decisión la toma quien llama a env.step(accion).

Por qué existe: en Gymnasium el agente vive FUERA del entorno. El repartidor
sigue percibiendo y actuando como siempre dentro de Mesa; lo único que cambia
es que, cuando le toca escoger, usa la acción que dejó el entorno Gymnasium
en vez de calcularla él.
"""
from __future__ import annotations

from sim.politicas.base import Politica


class PoliticaExterna(Politica):
    def __init__(self):
        super().__init__()
        self.accion: dict | None = None

    def elegir(self, percepcion: dict) -> dict:
        if self.accion is None:
            raise RuntimeError("El repartidor externo tuvo que decidir sin que nadie le diera una acción")
        accion, self.accion = self.accion, None
        return accion
