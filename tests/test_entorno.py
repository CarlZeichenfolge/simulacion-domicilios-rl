"""Fase 1: reloj, ciudad, eventos y pedidos."""
import random

import pytest

from sim.ciudad import Ciudad
from sim.eventos import GestorEventos
from sim.model import CiudadModel
from sim.pedidos import EstadoPedido, Pedido, TransicionInvalida
from sim.reloj import Reloj


# ------------------------------------------------------------------ reloj
def test_reloj_convierte_ticks():
    reloj = Reloj(hora_inicio=0)
    for _ in range(60 * 25 + 7):
        reloj.avanzar()
    assert reloj.dia == 1 and reloj.hora == 1 and reloj.minuto_del_dia == 67


def test_reloj_detecta_inicio_de_hora():
    reloj = Reloj()
    assert reloj.es_inicio_de_hora
    reloj.avanzar()
    assert not reloj.es_inicio_de_hora


# ------------------------------------------------------------------ ciudad
def test_ciudad_tiene_grilla_y_atributos(config):
    ciudad = Ciudad(config["ciudad"], random.Random(1))
    n = config["ciudad"]["tamano"] + 1
    assert ciudad.grafo.number_of_nodes() == n * n
    for _, _, datos in ciudad.grafo.edges(data=True):
        assert datos["factor_trancon"] == 1.0
        assert datos["longitud"] == config["ciudad"]["longitud_calle_m"]


def test_restaurantes_en_nodos_distintos(config):
    ciudad = Ciudad(config["ciudad"], random.Random(1))
    nodos = list(ciudad.restaurantes.values())
    assert len(set(nodos)) == 3
    assert all(ciudad.edificios[n] == "restaurante" for n in nodos)


def test_mapa_reproducible_con_semilla(config):
    a = Ciudad(config["ciudad"], random.Random(5))
    b = Ciudad(config["ciudad"], random.Random(5))
    assert a.edificios == b.edificios


def test_ruta_astar_es_optima_sin_trancon(config):
    ciudad = Ciudad(config["ciudad"], random.Random(1))
    ruta = ciudad.ruta((0, 0), (4, 3))
    assert ruta[0] == (0, 0) and ruta[-1] == (4, 3)
    assert len(ruta) - 1 == ciudad.distancia_cuadras((0, 0), (4, 3))


def test_trancon_y_lluvia_alargan_el_tiempo(config):
    ciudad = Ciudad(config["ciudad"], random.Random(1))
    base = ciudad.tiempo_arista((0, 0), (1, 0))
    ciudad.grafo.edges[(0, 0), (1, 0)]["factor_trancon"] = 2.0
    assert ciudad.tiempo_arista((0, 0), (1, 0)) == pytest.approx(2 * base)
    assert ciudad.tiempo_arista((0, 0), (1, 0), factor_velocidad=0.5) == pytest.approx(4 * base)


def test_astar_esquiva_calle_trancada(config):
    ciudad = Ciudad(config["ciudad"], random.Random(1))
    ciudad.grafo.edges[(0, 0), (1, 0)]["factor_trancon"] = 50.0
    ruta = ciudad.ruta((0, 0), (1, 0))
    assert len(ruta) > 2  # prefiere dar la vuelta


# ----------------------------------------------------------------- eventos
def _avanzar_hasta(eventos, reloj, hora):
    while reloj.hora != hora or not reloj.es_inicio_de_hora:
        reloj.avanzar()
    eventos.actualizar(reloj)


def test_trancones_solo_en_hora_pico(config):
    ciudad = Ciudad(config["ciudad"], random.Random(1))
    eventos = GestorEventos(config["eventos"], ciudad, random.Random(2))
    reloj = Reloj()
    _avanzar_hasta(eventos, reloj, 7)
    trancadas = [d["factor_trancon"] for *_, d in ciudad.grafo.edges(data=True) if d["factor_trancon"] > 1]
    esperado = round(ciudad.grafo.number_of_edges() * config["eventos"]["trancones"]["fraccion_aristas"])
    assert len(trancadas) == esperado and eventos.cambio_trafico
    _avanzar_hasta(eventos, reloj, 9)
    assert all(d["factor_trancon"] == 1.0 for *_, d in ciudad.grafo.edges(data=True))


@pytest.mark.parametrize("modo,esperado", [("siempre", True), ("nunca", False)])
def test_modos_de_lluvia(config, modo, esperado):
    config["eventos"]["lluvia"]["modo"] = modo
    ciudad = Ciudad(config["ciudad"], random.Random(1))
    eventos = GestorEventos(config["eventos"], ciudad, random.Random(2))
    reloj = Reloj()
    for _ in range(24 * 60):
        eventos.actualizar(reloj)
        assert eventos.lloviendo is esperado
        reloj.avanzar()


# ----------------------------------------------------------------- pedidos
def _pedido():
    return Pedido(id=0, habitante=None, restaurante=None, destino=(0, 0), precio=1.0,
                  pago_repartidor=1.0, creado=0)


def test_ciclo_de_vida_valido():
    p = _pedido()
    p.iniciar_preparacion(1, 10)
    p.marcar_listo(11)
    p.asignar("rep")
    p.recoger(12)
    p.entregar(20)
    assert p.estado == EstadoPedido.ENTREGADO and p.tiempo_entrega == 20


def test_transiciones_invalidas_fallan():
    p = _pedido()
    with pytest.raises(TransicionInvalida):
        p.entregar(5)
    with pytest.raises(TransicionInvalida):
        p.recoger(5)  # sin repartidor
    p.cancelar(5)
    with pytest.raises(TransicionInvalida):
        p.iniciar_preparacion(6, 10)


def test_no_se_asigna_dos_veces():
    p = _pedido()
    p.asignar("a")
    assert not p.disponible
    with pytest.raises(TransicionInvalida):
        p.asignar("b")


# ------------------------------------------------------------ modelo vacío
def test_modelo_sin_agentes_corre_un_dia(config_vacia):
    modelo = CiudadModel(config_vacia, semilla=3)
    modelo.correr()
    assert modelo.reloj.tick == 1440 and not modelo.running
    assert modelo.conteo["creados"] == 0
