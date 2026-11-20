"""Agente base con el ciclo percibir -> decidir -> actuar."""
from __future__ import annotations

import mesa


class AgenteBase(mesa.Agent):
    """Todos los agentes repiten el mismo ciclo en cada tick.

    Por qué forzar este ciclo: obliga a separar lo que el agente PUEDE VER
    (percibir, observabilidad parcial), lo que ELIGE (decidir, delegado a una
    política intercambiable) y lo que CAMBIA en el mundo (actuar). Es la
    definición de agente de Russell y Norvig hecha código.
    """

    def __init__(self, model, politica=None):
        super().__init__(model)
        self.politica = politica
        if politica is not None:
            politica.vincular(model)

    def step(self) -> None:
        percepcion = self.percibir()
        accion = self.decidir(percepcion)
        self.actuar(accion)

    def percibir(self) -> dict:
        raise NotImplementedError

    def decidir(self, percepcion: dict) -> dict:
        return self.politica.elegir(percepcion)

    def actuar(self, accion: dict) -> None:
        raise NotImplementedError

    @property
    def t(self) -> int:
        return self.model.reloj.tick
