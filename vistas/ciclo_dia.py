"""Colores del cielo y de la luz según la hora simulada (y la lluvia).

Por qué: la ciudad vive un día entero (de madrugada todos duermen, el almuerzo es
a mediodía, los trancones al amanecer y al atardecer). Si el cielo cambia con el
reloj, se LEE la hora sin mirar el marcador. Sin Ursina: se prueba con pytest.

La paleta está en config/base.yaml (vista_3d.ciclo_dia) como fotogramas clave
por hora; entre dos fotogramas se interpola suave.
"""
from __future__ import annotations

import math

from vistas.geometria import Color, hex_a_color, mezclar

CAMPOS_COLOR = ("cenit", "horizonte", "sol", "cielo", "suelo")
CAMPOS_NUMERO = ("intensidad", "noche")
HORAS_DIA = 24


def _suave(t: float) -> float:
    """Smoothstep: sin "codos" de color al pasar por cada fotograma."""
    return t * t * (3 - 2 * t)


def _fotogramas(cfg: dict) -> list[dict]:
    """Ordena por hora y repite el primero en hora+24 para que la medianoche empalme."""
    lista = sorted(cfg["ciclo_dia"], key=lambda f: f["hora"])
    return lista + [dict(lista[0], hora=lista[0]["hora"] + HORAS_DIA)]


def _normalizar(v) -> tuple[float, float, float]:
    largo = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / largo for c in v)


def luminancia(c: Color) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def direccion_sol(cfg: dict, hora: float) -> tuple[tuple[float, float, float], float]:
    """Vector unitario HACIA el sol y su elevación (radianes; negativa de noche).

    Sale por +x (este), pasa por el lado de la cámara (-z) y se pone por -x. Va del
    lado de la cámara para que las fachadas que se ven queden iluminadas.
    """
    sol = cfg["sol"]
    t = (hora - sol["salida"]) / (sol["puesta"] - sol["salida"])
    angulo = math.pi * t
    elevacion = math.sin(angulo) * math.radians(sol["elevacion_maxima"])
    horizontal = _normalizar((math.cos(angulo), 0.0, -sol["inclinacion_sur"]))
    direccion = (horizontal[0] * math.cos(elevacion), math.sin(elevacion), horizontal[2] * math.cos(elevacion))
    return _normalizar(direccion), elevacion


def estado_cielo(cfg: dict, hora: float, lluvia: float = 0.0) -> dict:
    """Todo lo que la vista necesita para pintar un instante.

    `hora` con decimales (11.5 = 11:30); `lluvia` de 0 (seco) a 1 (aguacero): la vista
    la sube y baja suave para que el cambio de clima no sea un salto.
    """
    hora = hora % HORAS_DIA
    marcos = _fotogramas(cfg)
    if hora < marcos[0]["hora"]:
        hora += HORAS_DIA
    for a, b in zip(marcos, marcos[1:]):
        if a["hora"] <= hora <= b["hora"]:
            break
    t = _suave((hora - a["hora"]) / (b["hora"] - a["hora"]))
    estado = {campo: mezclar(hex_a_color(a[campo]), hex_a_color(b[campo]), t) for campo in CAMPOS_COLOR}
    estado.update({campo: a[campo] + (b[campo] - a[campo]) * t for campo in CAMPOS_NUMERO})

    sol_dir, elevacion = direccion_sol(cfg, hora % HORAS_DIA)
    estado["sol_dir"] = sol_dir
    estado["sol_visible"] = max(0.0, min(1.0, elevacion / math.radians(cfg["sol"]["elevacion_fundido"]) + 1))
    estado["luna_dir"] = _normalizar(cfg["luna"]["direccion"])
    # La luz que proyecta sombras: el sol de día, la luna de noche. Con el sol muy bajo
    # las sombras serían infinitas: se recorta la elevación mínima. Alrededor del
    # horizonte se pasa de una a otra poco a poco, para que las sombras no salten.
    minima = math.radians(cfg["sol"]["elevacion_minima_sombra"])
    horizontal = _normalizar((sol_dir[0], 0.0, sol_dir[2]))
    e = max(elevacion, minima)
    sol_recortado = (horizontal[0] * math.cos(e), math.sin(e), horizontal[2] * math.cos(e))
    fundido = math.radians(cfg["sol"]["elevacion_fundido"])
    w = _suave(max(0.0, min(1.0, (elevacion + fundido) / (2 * fundido))))
    estado["luz_dir"] = _normalizar(tuple(l + (s - l) * w for l, s in zip(estado["luna_dir"], sol_recortado)))
    return con_lluvia(cfg, estado, lluvia) if lluvia > 0 else estado


def con_lluvia(cfg: dict, estado: dict, lluvia: float) -> dict:
    """Cielo encapotado: colores más grises y oscuros, sol más débil.

    Por qué gris "relativo" (de la misma luminancia) y no un gris fijo: de noche la
    lluvia no debe aclarar el cielo, y al atardecer debe conservar algo del tono.
    """
    c = cfg["lluvia"]
    nuevo = dict(estado)
    tinte = hex_a_color(c["tinte"])
    for campo in CAMPOS_COLOR:
        original = estado[campo]
        lum = luminancia(original) * c["oscurecer"]
        gris = (tinte[0] * lum, tinte[1] * lum, tinte[2] * lum, original[3])
        nuevo[campo] = mezclar(original, gris, lluvia * c["desaturar"])
    nuevo["intensidad"] = estado["intensidad"] * (1 - lluvia * (1 - c["factor_sol"]))
    nuevo["sol_visible"] = estado["sol_visible"] * (1 - lluvia)
    nuevo["noche"] = min(1.0, estado["noche"] + lluvia * c["luces_extra"])
    return nuevo
