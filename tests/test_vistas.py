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


def test_saltar_al_dia_siguiente_llega_a_la_manana(config):
    """Tecla N de la vista 3D: de cualquier hora del día 1 a las 7:00 del día 2."""
    from vistas.velocidad import saltar_al_dia_siguiente
    config = fusionar(config, {"tiempo": {"dias_por_episodio": 3}})
    modelo = CiudadModel(config, semilla=3)
    adelantar(modelo, 15)
    saltar_al_dia_siguiente(modelo, 7)
    assert (modelo.reloj.dia, modelo.reloj.hora, modelo.reloj.minuto_del_dia % 60) == (1, 7, 0)
    saltar_al_dia_siguiente(modelo, 7)
    saltar_al_dia_siguiente(modelo, 7)  # el tercer salto pasaría del final: se queda en el último minuto
    assert not modelo.running and modelo.reloj.tick == modelo.duracion


def test_marcador_de_una_semana(config):
    from vistas.marcador import titulo_marcador
    semana = CiudadModel(fusionar(config, {"tiempo": {"dias_por_episodio": 7}}), semilla=3)
    assert titulo_marcador(semana).startswith("HASTA AHORA (DÍA 1 DE 7)")
    un_dia = CiudadModel(config, semilla=3)
    while un_dia.running:
        un_dia.step()
    assert titulo_marcador(un_dia) == "FIN DEL DÍA 1"


def test_elegir_mapa(config):
    from main import elegir_mapa
    posibles = config["vistas"]["mapas_posibles"]
    assert elegir_mapa(config, 123) == 123
    with pytest.raises(SystemExit):
        elegir_mapa(config, posibles)
    assert all(0 <= elegir_mapa(config, None) < posibles for _ in range(50))
    fijo = fusionar(config, {"vistas": {"mapa_aleatorio": False}})
    assert elegir_mapa(fijo, None) == config["ciudad"]["semilla_mapa"]


def _plano(config, mapa):
    import random
    from sim.ciudad import Ciudad
    return Ciudad(config["ciudad"], random.Random(mapa))


def test_cada_numero_de_mapa_da_una_ciudad_valida_y_distinta(config):
    """Una muestra de los 100 000 mapas (incluye los extremos): todos se pueden recorrer,
    tienen los edificios del YAML, y no hay dos iguales."""
    import random

    import networkx as nx
    posibles = config["vistas"]["mapas_posibles"]
    muestra = [0, posibles - 1] + random.Random(0).sample(range(1, posibles - 1), 200)
    vistos = set()
    cantidades = config["ciudad"]["edificios"]
    for mapa in muestra:
        ciudad = _plano(config, mapa)
        assert nx.is_connected(ciudad.grafo)
        assert len(set(ciudad.restaurantes.values())) == len(config["ciudad"]["restaurantes"])
        assert len(ciudad.nodos_de_tipo("casa")) == cantidades["casas"]
        vistos.add(tuple(sorted(ciudad.edificios.items())))
    assert len(vistos) == len(muestra)


def test_el_mismo_numero_da_el_mismo_mapa(config):
    assert _plano(config, 48213).edificios == _plano(config, 48213).edificios
