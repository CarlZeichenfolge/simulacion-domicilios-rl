"""Línea base: acepta cualquier pedido disponible al azar."""
from __future__ import annotations

from sim.politicas.base import Politica, accion_aceptar, accion_esperar


class PoliticaRepartidorAleatoria(Politica):
    """Por qué existe: es el piso de comparación. Si Q-learning no le gana a
    esto, no aprendió nada útil. Nunca espera si hay pedidos, para que la
    diferencia con las demás sea solo CUÁL pedido toma, no si trabaja.
    """

    def elegir(self, percepcion: dict) -> dict:
        pedidos = percepcion["pedidos"]
        if not pedidos:
            return accion_esperar()
        return accion_aceptar(self.rng.choice(pedidos), indice=-1)
