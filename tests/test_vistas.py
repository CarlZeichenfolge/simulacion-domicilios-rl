"""Piezas de las demos que no necesitan ventana: velocidad, adelantar y clima compartido."""
import pytest

from sim.config import fusionar
from sim.model import CiudadModel
from vistas.velocidad import Velocidad, adelantar


def test_velocidad_es_por_segundo_real_y_no_por_cuadro(config):
    """2 min/s: 60 cuadros de 1/60 s dan 2 minutos, igual que 30 cuadros de 1/30 s."""
    rapida, lenta = Velocidad(config["vistas"]), Velocidad(config["vistas"])
    assert sum(rapida.pasos(1 / 60) for _ in range(60)) == 2
    assert sum(lenta.pasos(1 / 30) for _ in range(30)) == 2
    assert 0 <= rapida.fraccion < 1


def test_un_cuadro_trabado_no_hace_saltar_la_ciudad(config):
    v = Velocidad(config["vistas"])
    assert v.pasos(5.0) == int(config["vistas"]["dt_maximo"] * v.minutos_por_segundo)


def test_subir_y_bajar_se_quedan_en_la_lista(config):
    v = Velocidad(config["vistas"])
    for _ in range(50):
        v.subir()
    assert v.minutos_por_segundo == config["vistas"]["minutos_por_segundo"][-1]
    for _ in range(50):
        v.bajar()
    assert v.minutos_por_segundo == config["vistas"]["minutos_por_segundo"][0]


def test_adelantar_llega_a_la_hora(config):
    modelo = CiudadModel(config, semilla=3)
    adelantar(modelo, 11)
    assert modelo.reloj.tick == 11 * 60


def _clima_del_dia(config, politica):
    modelo = CiudadModel(fusionar(config, {"repartidores": {"politica": politica}}), semilla=5)
    clima = []
    while modelo.running:
        modelo.step()
        clima.append((modelo.eventos.lloviendo, tuple(modelo.eventos.aristas_con_trancon)))
    return clima


def test_con_azar_propio_dos_politicas_viven_el_mismo_clima(config):
    """En los duelos lo único distinto entre las dos ciudades debe ser la política."""
    config = fusionar(config, {"eventos": {"azar_propio": True}})
    assert _clima_del_dia(config, "aleatoria") == _clima_del_dia(config, "urgente")


@pytest.mark.parametrize("politica", ["aleatoria", "urgente"])
def test_azar_propio_es_reproducible(config, politica):
    config = fusionar(config, {"eventos": {"azar_propio": True}})
    assert _clima_del_dia(config, politica) == _clima_del_dia(config, politica)
