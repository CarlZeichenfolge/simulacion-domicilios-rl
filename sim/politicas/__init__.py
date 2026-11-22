"""Fábrica de políticas: el nombre en el YAML decide qué clase se usa."""
from __future__ import annotations

from sim.politicas.base import Politica
from sim.politicas.repartidor_aleatoria import PoliticaRepartidorAleatoria
from sim.politicas.repartidor_cercano import PoliticaRepartidorCercano
from sim.politicas.repartidor_qlearning import PoliticaRepartidorQLearning
from sim.politicas.repartidor_urgente import PoliticaRepartidorUrgente


def crear_politica_repartidor(cfg_repartidores: dict) -> Politica:
    nombre = cfg_repartidores["politica"]
    if nombre == "aleatoria":
        return PoliticaRepartidorAleatoria()
    if nombre == "cercano":
        return PoliticaRepartidorCercano()
    if nombre == "urgente":
        return PoliticaRepartidorUrgente()
    if nombre == "qlearning":
        return PoliticaRepartidorQLearning(cfg_repartidores["qlearning"])
    if nombre == "dqn":
        # Import tardío: PyTorch solo se carga si de verdad se usa la red.
        from sim.politicas.repartidor_dqn import PoliticaRepartidorDQN
        return PoliticaRepartidorDQN(cfg_repartidores["dqn"])
    raise ValueError(f"Política de repartidor desconocida: {nombre}")
