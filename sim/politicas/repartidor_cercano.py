"""Heurística voraz: acepta el pedido cuyo restaurante está más cerca."""
from __future__ import annotations

from sim.politicas.base import CERCANO, Politica, accion_aceptar, accion_esperar, mas_cercano


class PoliticaRepartidorCercano(Politica):
    """Por qué es la línea base fuerte: minimizar el trayecto vacío es lo que
    haría una persona sensata. Su debilidad es que ignora el pago, la hora y
    la lluvia; eso es justo lo que Q-learning puede aprender a aprovechar.
    """

    def elegir(self, percepcion: dict) -> dict:
        pedidos = percepcion["pedidos"]
        if not pedidos:
            return accion_esperar()
        return accion_aceptar(mas_cercano(pedidos), indice=CERCANO)
