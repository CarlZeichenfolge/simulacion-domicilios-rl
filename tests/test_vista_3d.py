"""Piezas de la vista 3D que no necesitan ventana: geometría, ciclo del día y maqueta."""
import math
import random

import pytest

from sim.model import CiudadModel
from vistas.ciclo_dia import estado_cielo
from vistas.geometria import Malla
from vistas.maqueta_3d import Maqueta, malla_moto, malla_trancones


def _normales_hacia_afuera(malla: Malla, centro) -> bool:
    """Cada triángulo con su normal alejándose del centro (si no, Ursina lo esconde)."""
    v, n = malla.vertices, malla.normales
    for i in range(0, len(v), 9):
        cx = sum(v[i + k] for k in (0, 3, 6)) / 3 - centro[0]
        cy = sum(v[i + k] for k in (1, 4, 7)) / 3 - centro[1]
        cz = sum(v[i + k] for k in (2, 5, 8)) / 3 - centro[2]
        if cx * n[i] + cy * n[i + 1] + cz * n[i + 2] <= 0:
            return False
    return True


def _orden_de_ursina(malla: Malla) -> bool:
    """Ursina (mano izquierda) muestra la cara cuya normal es OPUESTA a (b-a)×(c-a)."""
    v, n = malla.vertices, malla.normales
    for i in range(0, len(v), 9):
        a, b, c = v[i:i + 3], v[i + 3:i + 6], v[i + 6:i + 9]
        ab = [b[k] - a[k] for k in range(3)]
        ac = [c[k] - a[k] for k in range(3)]
        cruz = (ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0])
        if sum(cruz[k] * n[i + k] for k in range(3)) >= 0:
            return False
    return True


@pytest.mark.parametrize("construir", [
    lambda m: m.caja((1, 2, 3), (1, 2, 1), (1, 1, 1, 1), sin_base=False),
    lambda m: m.caja((1, 2, 3), (1, 2, 1), (1, 1, 1, 1), giro=30, sin_base=False),
    lambda m: m.cilindro((1, 2, 3), 0.5, 1, 8, (1, 1, 1, 1)),
    lambda m: m.cilindro((1, 2, 3), 0.5, 1, 8, (1, 1, 1, 1), eje="x"),
    lambda m: m.cilindro((1, 2, 3), 0.5, 1, 8, (1, 1, 1, 1), eje="z"),
    lambda m: m.esfera((1, 2, 3), (1, 0.5, 1), (1, 1, 1, 1)),
])
def test_solidos_cerrados_miran_hacia_afuera(construir):
    m = Malla()
    construir(m)
    assert len(m) % 3 == 0
    assert _normales_hacia_afuera(m, (1, 2, 3))
    assert _orden_de_ursina(m)


def test_rect_suelo_mira_hacia_arriba():
    m = Malla()
    m.rect_suelo(0, 0, 1, 2, 0.1, (1, 1, 1, 1))
    assert all(m.normales[i + 1] == pytest.approx(1) for i in range(0, len(m.normales), 3))
    assert _orden_de_ursina(m)


def test_el_cielo_cambia_con_la_hora(config):
    cfg = config["vista_3d"]["cielo"]
    mediodia, medianoche = estado_cielo(cfg, 12), estado_cielo(cfg, 0)
    assert mediodia["noche"] == 0 and medianoche["noche"] == 1
    assert sum(mediodia["cenit"][:3]) > sum(medianoche["cenit"][:3])
    assert mediodia["sol_dir"][1] > 0 > estado_cielo(cfg, 23)["sol_dir"][1]
    # La medianoche empalma: 23:59 y 00:00 casi iguales
    assert estado_cielo(cfg, 23.99)["cenit"] == pytest.approx(medianoche["cenit"], abs=0.01)


def test_la_luz_no_salta_al_amanecer(config):
    """Entre la luna y el sol hay transición: un minuto no gira la sombra más de unos grados."""
    cfg = config["vista_3d"]["cielo"]
    salida = cfg["sol"]["salida"]
    anterior = estado_cielo(cfg, salida - 1)["luz_dir"]
    for minuto in range(1, 121):
        actual = estado_cielo(cfg, salida - 1 + minuto / 60)["luz_dir"]
        angulo = math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(anterior, actual))))))
        assert angulo < 5
        anterior = actual


def test_la_lluvia_oscurece_sin_cambiar_la_hora(config):
    cfg = config["vista_3d"]["cielo"]
    seco, mojado = estado_cielo(cfg, 12), estado_cielo(cfg, 12, lluvia=1.0)
    assert mojado["intensidad"] < seco["intensidad"]
    assert sum(mojado["cenit"][:3]) < sum(seco["cenit"][:3])
    assert mojado["sol_dir"] == seco["sol_dir"]


def test_la_maqueta_es_reproducible_y_no_toca_el_modelo(config):
    """Misma semilla de adorno -> misma maqueta; y construirla no consume el azar del modelo."""
    modelo = CiudadModel(config, semilla=3)
    estado_rng = modelo.random.getstate()
    cfg = config["vista_3d"]
    a = Maqueta(modelo.ciudad, cfg, random.Random(cfg["semilla_decorado"]))
    b = Maqueta(modelo.ciudad, cfg, random.Random(cfg["semilla_decorado"]))
    assert a.estatico.vertices == b.estatico.vertices
    assert modelo.random.getstate() == estado_rng
    assert set(a.restaurantes) == set(modelo.ciudad.restaurantes)
    assert set(a.lotes) == set(modelo.ciudad.edificios)


def test_mallas_completas(config):
    modelo = CiudadModel(config, semilla=3)
    cfg = config["vista_3d"]
    q = Maqueta(modelo.ciudad, cfg, random.Random(0))
    for malla in (q.estatico, q.ventanas_encendibles, q.ventanas_apagadas, q.focos, q.halos):
        assert len(malla) > 0
        assert len(malla.normales) == len(malla.vertices)
        assert len(malla.colores) == len(malla) * 4
    assert q.halos.usa_uv and len(q.halos.uvs) == len(q.halos) * 2
    for malla in malla_moto(q.col):
        assert len(malla) > 0
    aristas = list(modelo.ciudad.grafo.edges)[:5]
    assert len(malla_trancones(q, aristas, random.Random(0))) > 0
