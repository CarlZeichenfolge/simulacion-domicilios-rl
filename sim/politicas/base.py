"""Interfaz común de las políticas de decisión.

Por qué separar política y agente: el agente sabe percibir y actuar en el
entorno; la política solo decide. Así comparamos aleatoria, más cercano y
Q-learning con el MISMO repartidor, cambiando una línea del YAML.
"""
from __future__ import annotations

import random

# Criterios de las heurísticas fijas (solo para identificar la acción tomada).
# "urgente" (el pedido más viejo) se agregó porque en las pruebas el más
# cercano dejaba vencer los pedidos lejanos.
CERCANO, MEJOR_PAGADO, URGENTE, ESPERAR = 0, 1, 2, 3


class Politica:
    """Contrato: elegir(percepcion) -> accion y aprender(...) tras ver el resultado."""

    aprende = False  # True solo en políticas con aprendizaje

    def __init__(self):
        self.rng: random.Random | None = None

    def vincular(self, modelo) -> None:
        """Usa el generador del modelo: misma semilla, mismas decisiones."""
        self.rng = modelo.random

    def elegir(self, percepcion: dict) -> dict:
        raise NotImplementedError

    def aprender(self, percepcion: dict, accion: dict, recompensa: float,
                 nueva_percepcion: dict, duracion: int = 1) -> None:
        """Por defecto no aprende (políticas fijas)."""

    def fin_episodio(self) -> None:
        """Gancho para cosas como bajar epsilon entre episodios."""


# --------------------------------------------------------------- utilidades
def accion_esperar() -> dict:
    return {"tipo": "esperar", "indice": ESPERAR}


def accion_aceptar(info: dict, indice: int) -> dict:
    return {"tipo": "aceptar", "pedido": info["pedido"], "indice": indice}


def mas_cercano(pedidos: list[dict]) -> dict:
    """Desempate por id: el más antiguo primero, para que sea determinista."""
    return min(pedidos, key=lambda p: (p["distancia"], p["pedido"].id))


def mejor_pagado(pedidos: list[dict]) -> dict:
    return min(pedidos, key=lambda p: (-p["pago"], p["pedido"].id))


def mas_urgente(pedidos: list[dict]) -> dict:
    """El más viejo es el que está más cerca de cancelarse."""
    return min(pedidos, key=lambda p: (-p["edad"], p["pedido"].id))
