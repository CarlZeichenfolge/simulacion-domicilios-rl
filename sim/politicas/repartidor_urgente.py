"""Heurística: acepta siempre el pedido más antiguo (el más cerca de cancelarse)."""
from __future__ import annotations

from sim.politicas.base import URGENTE, Politica, accion_aceptar, accion_esperar, mas_urgente


class PoliticaRepartidorUrgente(Politica):
    """Por qué existe: es el opuesto de "más cercano". Minimiza cancelaciones
    pero puede cruzar media ciudad por un pedido. Q-learning debería aprender
    cuándo conviene cada criterio en vez de usar siempre uno solo.
    """

    def elegir(self, percepcion: dict) -> dict:
        pedidos = percepcion["pedidos"]
        if not pedidos:
            return accion_esperar()
        return accion_aceptar(mas_urgente(pedidos), indice=URGENTE)
