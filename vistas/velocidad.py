"""Velocidad de las demos en minutos simulados POR SEGUNDO REAL (no por cuadro).

Por qué: antes la velocidad era "minutos por cuadro". Un computador que dibuja
144 cuadros por segundo corría la ciudad el doble de rápido que uno de 60, y
en la vista 3D (sin límite de cuadros) un día entero pasaba en segundos. Con un
reloj real la demo se ve igual en cualquier computador.

No importa pygame ni ursina: lo usan las dos vistas.
"""
from __future__ import annotations

from sim.model import CiudadModel
from sim.reloj import MINUTOS_POR_DIA, MINUTOS_POR_HORA

TOLERANCIA = 1e-9  # error de redondeo de punto flotante, no un parámetro de la demo


class Velocidad:
    """Acumula tiempo real y dice cuántos minutos simular en cada cuadro.

    `fraccion` (0 a 1) es cuánto falta para el siguiente minuto: las vistas la usan
    para dibujar las motos ENTRE la posición del minuto anterior y la del actual,
    así se mueven suave aunque la simulación avance de a un minuto.
    """

    def __init__(self, cfg: dict):
        self.niveles = cfg["minutos_por_segundo"]
        self.indice = self.niveles.index(cfg["velocidad_inicial"])
        self.dt_maximo = cfg["dt_maximo"]
        self.acumulado = 0.0

    @property
    def minutos_por_segundo(self) -> float:
        return self.niveles[self.indice]

    @property
    def fraccion(self) -> float:
        return self.acumulado

    def pasos(self, dt: float) -> int:
        # dt se recorta: el primer cuadro (o uno trabado) trae un dt enorme y saltaría media hora.
        self.acumulado += min(dt, self.dt_maximo) * self.minutos_por_segundo
        # La tolerancia evita que 60 sumas de 1/60 den 1.9999999 y se "pierda" un minuto por redondeo.
        n = int(self.acumulado + TOLERANCIA)
        self.acumulado = max(0.0, self.acumulado - n)
        return n

    def subir(self) -> None:
        self.indice = min(len(self.niveles) - 1, self.indice + 1)

    def bajar(self) -> None:
        self.indice = max(0, self.indice - 1)

    def texto(self) -> str:
        return f"{self.minutos_por_segundo:g} min/s"


def adelantar(modelo: CiudadModel, hora: float) -> None:
    """Corre la ciudad sin dibujar hasta `hora` del primer día.

    Por qué: el episodio empieza a medianoche y hasta las 6-7 a. m. todos duermen;
    a velocidad cómoda serían minutos de pantalla quieta. Adelantar no cambia la
    simulación: es la misma que se vería mirando desde las 0:00.
    """
    while modelo.running and modelo.reloj.tick < hora * MINUTOS_POR_HORA:
        modelo.step()


def saltar_al_dia_siguiente(modelo: CiudadModel, hora: float) -> None:
    """Corre sin dibujar hasta `hora` del día siguiente (o hasta que termine el episodio).

    Por qué: la demo sigue una semana y a velocidad cómoda serían más de 80 minutos,
    casi todos de noche con la ciudad dormida. Igual que adelantar(), no cambia la
    simulación: es la misma que se vería mirando sin saltar.
    """
    objetivo = (modelo.reloj.dia + 1) * MINUTOS_POR_DIA + hora * MINUTOS_POR_HORA
    while modelo.running and modelo.reloj.minuto_absoluto < objetivo:
        modelo.step()


def dias_del_episodio(modelo: CiudadModel) -> int:
    return round(modelo.duracion / MINUTOS_POR_DIA)
