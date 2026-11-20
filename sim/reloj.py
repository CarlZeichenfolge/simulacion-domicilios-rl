"""Tiempo simulado: un tick es un minuto."""
from __future__ import annotations

MINUTOS_POR_HORA = 60
MINUTOS_POR_DIA = 1440  # definición del tick, no un parámetro


class Reloj:
    """Convierte el número de tick en día, hora y minuto.

    Por qué una clase aparte: agentes y eventos solo necesitan preguntar
    "qué hora es"; si mañana el tick valiera 30 segundos, se cambia aquí.
    """

    def __init__(self, hora_inicio: int = 0):
        self.minuto_inicio = hora_inicio * MINUTOS_POR_HORA
        self.tick = 0

    def avanzar(self) -> None:
        self.tick += 1

    @property
    def minuto_absoluto(self) -> int:
        return self.minuto_inicio + self.tick

    @property
    def minuto_del_dia(self) -> int:
        return self.minuto_absoluto % MINUTOS_POR_DIA

    @property
    def hora(self) -> int:
        return self.minuto_del_dia // MINUTOS_POR_HORA

    @property
    def dia(self) -> int:
        return self.minuto_absoluto // MINUTOS_POR_DIA

    @property
    def es_inicio_de_hora(self) -> bool:
        return self.minuto_del_dia % MINUTOS_POR_HORA == 0

    def texto(self) -> str:
        return f"Día {self.dia + 1}  {self.hora:02d}:{self.minuto_del_dia % 60:02d}"
