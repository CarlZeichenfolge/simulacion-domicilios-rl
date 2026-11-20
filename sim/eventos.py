"""Eventos estocásticos del entorno: lluvia y trancones."""
from __future__ import annotations

import random

from sim.ciudad import Ciudad
from sim.reloj import Reloj


class GestorEventos:
    """Actualiza lluvia y trancones al inicio de cada hora.

    Por qué cada hora y no cada minuto: el clima y el tráfico cambian a
    escala de horas; revisarlos cada minuto haría que la lluvia parpadee.
    Todo usa el rng del modelo para que la semilla reproduzca el día.
    """

    def __init__(self, cfg: dict, ciudad: Ciudad, rng: random.Random):
        self.cfg_lluvia = cfg["lluvia"]
        self.cfg_trancones = cfg["trancones"]
        self.ciudad = ciudad
        self.rng = rng
        self.lloviendo = self.cfg_lluvia["modo"] == "siempre"
        self.trancon_activo = False
        self.aristas_con_trancon: list[tuple] = []
        # True durante el tick en que cambió el tráfico: los repartidores replanifican.
        self.cambio_trafico = False

    @property
    def factor_velocidad(self) -> float:
        return self.cfg_lluvia["factor_velocidad"] if self.lloviendo else 1.0

    def actualizar(self, reloj: Reloj) -> None:
        self.cambio_trafico = False
        if not reloj.es_inicio_de_hora:
            return
        self._actualizar_lluvia()
        self._actualizar_trancones(reloj.hora)

    def _actualizar_lluvia(self) -> None:
        """Cadena de Markov de dos estados (seco / lluvia) con paso de una hora."""
        modo = self.cfg_lluvia["modo"]
        if modo == "siempre":
            self.lloviendo = True
        elif modo == "nunca":
            self.lloviendo = False
        elif modo == "estocastica":
            if self.lloviendo:
                if self.rng.random() < self.cfg_lluvia["prob_parar_por_hora"]:
                    self.lloviendo = False
            elif self.rng.random() < self.cfg_lluvia["prob_empezar_por_hora"]:
                self.lloviendo = True
        else:
            raise ValueError(f"Modo de lluvia desconocido: {modo}")

    def es_hora_pico(self, hora: int) -> bool:
        return any(inicio <= hora < fin for inicio, fin in self.cfg_trancones["horas_pico"])

    def _actualizar_trancones(self, hora: int) -> None:
        """Al entrar a hora pico se trancan aristas al azar; al salir, todo vuelve a 1.0.

        Por qué se sortean al entrar y no cada hora: un trancón real dura toda
        la franja pico, no salta de calle en calle.
        """
        pico = self.cfg_trancones["activos"] and self.es_hora_pico(hora)
        if pico and not self.trancon_activo:
            aristas = self.ciudad.aristas()
            cantidad = round(len(aristas) * self.cfg_trancones["fraccion_aristas"])
            self.aristas_con_trancon = self.rng.sample(aristas, cantidad)
            for u, v in self.aristas_con_trancon:
                factor = self.rng.uniform(self.cfg_trancones["factor_min"], self.cfg_trancones["factor_max"])
                self.ciudad.grafo.edges[u, v]["factor_trancon"] = factor
            self.trancon_activo = True
            self.cambio_trafico = True
        elif not pico and self.trancon_activo:
            for u, v in self.aristas_con_trancon:
                self.ciudad.grafo.edges[u, v]["factor_trancon"] = 1.0
            self.aristas_con_trancon = []
            self.trancon_activo = False
            self.cambio_trafico = True
