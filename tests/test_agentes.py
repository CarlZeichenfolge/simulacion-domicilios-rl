"""Fases 2 y 4: agentes, ciclo completo de un pedido, políticas y reproducibilidad."""
import random

import pytest

from sim.agentes import EstadoRepartidor
from sim.config import fusionar
from sim.metricas import resumen_episodio
from sim.model import CiudadModel
from sim.pedidos import EstadoPedido
from sim.politicas.habitante_egreedy import PoliticaHabitanteEGreedy
from sim.politicas.repartidor_cercano import PoliticaRepartidorCercano
from sim.politicas.repartidor_qlearning import PoliticaRepartidorQLearning
from sim.politicas.repartidor_urgente import PoliticaRepartidorUrgente


class _P:  # pedido mínimo para probar políticas
    def __init__(self, id):
        self.id = id


def _info(id, distancia, pago=5000, edad=0, entrega=3, listo=False):
    return {"pedido": _P(id), "distancia": distancia, "pago": pago, "edad": edad,
            "distancia_entrega": entrega, "listo": listo}


# ------------------------------------------------------------- reproducibilidad
@pytest.mark.parametrize("politica", ["aleatoria", "cercano", "urgente", "qlearning"])
def test_misma_semilla_mismo_resultado(config, politica):
    config["repartidores"]["politica"] = politica
    a = CiudadModel(config, semilla=11)
    a.correr()
    b = CiudadModel(config, semilla=11)
    b.correr()
    assert resumen_episodio(a) == resumen_episodio(b)


def test_semillas_distintas_dan_dias_distintos(config):
    a = CiudadModel(config, semilla=1)
    a.correr()
    b = CiudadModel(config, semilla=2)
    b.correr()
    assert resumen_episodio(a) != resumen_episodio(b)


# --------------------------------------------------------- ciclo de un pedido
def test_pedido_de_punta_a_punta(config):
    modelo = CiudadModel(config, semilla=4)
    modelo.correr()
    entregados = [p for p in modelo.pedidos if p.estado == EstadoPedido.ENTREGADO]
    assert entregados, "ningún pedido llegó"
    for p in entregados:
        assert p.creado <= p.aceptado_rest <= p.listo <= p.recogido <= p.entregado
        assert p.listo - p.aceptado_rest >= p.tiempo_preparacion
        assert p.entregado - p.creado < config["pedidos"]["espera_maxima"]
    assert modelo.conteo["entregados"] == len(entregados)


def test_sin_repartidores_todo_se_cancela_a_tiempo(config):
    config = fusionar(config, {"repartidores": {"cantidad": 0}})
    modelo = CiudadModel(config, semilla=4)
    modelo.correr()
    espera = config["pedidos"]["espera_maxima"]
    vencidos = [p for p in modelo.pedidos if p.creado + espera < modelo.reloj.tick]
    assert vencidos
    assert all(p.estado == EstadoPedido.CANCELADO and p.cancelado - p.creado == espera
               for p in vencidos)


def test_repartidor_llega_al_destino(config):
    modelo = CiudadModel(config, semilla=4)
    for _ in range(1440):
        modelo.step()
        for r in modelo.repartidores:
            if r.estado == EstadoRepartidor.YENDO_A_CLIENTE and not r.ruta:
                assert r.nodo == r.pedido.destino


def test_con_lluvia_salen_menos_a_comer(config):
    def salidas(modo):
        cfg = fusionar(config, {"eventos": {"lluvia": {"modo": modo}}})
        total = 0
        for s in range(3):
            m = CiudadModel(cfg, semilla=s)
            m.correr()
            total += sum(1 for c in m.comidas if c["tipo"] == "salir")
        return total
    assert salidas("siempre") <= salidas("nunca")


# ----------------------------------------------------------------- políticas
def test_cercano_y_urgente():
    pedidos = [_info(1, 8, edad=50), _info(2, 2, edad=5), _info(3, 2, edad=10)]
    cercano = PoliticaRepartidorCercano().elegir({"pedidos": pedidos})
    assert cercano["pedido"].id == 2  # empate en distancia: gana el id menor
    urgente = PoliticaRepartidorUrgente().elegir({"pedidos": pedidos})
    assert urgente["pedido"].id == 1
    assert PoliticaRepartidorCercano().elegir({"pedidos": []})["tipo"] == "esperar"


def test_qlearning_actualiza_con_smdp(config):
    pol = PoliticaRepartidorQLearning(config["repartidores"]["qlearning"])
    pol.rng = random.Random(0)
    s = {"hora": 12, "lluvia": False, "pedidos": [_info(1, 2)]}
    s2 = {"hora": 12, "lluvia": False, "pedidos": []}
    accion = {"indice": pol.tipo_pedido(s["pedidos"][0])}
    pol.aprender(s, accion, recompensa=10.0, nueva_percepcion=s2, duracion=30)
    assert pol.q[pol.estado(s) + (accion["indice"],)] == pytest.approx(pol.alpha * 10.0)
    # Con epsilon 0 elige lo aprendido en vez de esperar.
    pol.epsilon = 0.0
    assert pol.elegir(s)["tipo"] == "aceptar"


def test_qlearning_no_aprende_en_evaluacion(config):
    pol = PoliticaRepartidorQLearning(config["repartidores"]["qlearning"])
    pol.explorar = False
    s = {"hora": 12, "lluvia": False, "pedidos": [_info(1, 2)]}
    pol.aprender(s, {"indice": 0}, 10.0, s, 5)
    assert not pol.q.any()


def test_qlearning_guarda_y_carga(config, tmp_path):
    pol = PoliticaRepartidorQLearning(config["repartidores"]["qlearning"])
    pol.q[0, 0, 0, 0] = 3.5
    pol.guardar(tmp_path / "q.npz")
    otra = PoliticaRepartidorQLearning(config["repartidores"]["qlearning"])
    otra.cargar(tmp_path / "q.npz")
    assert otra.q[0, 0, 0, 0] == 3.5


def test_bandit_se_aleja_del_restaurante_malo():
    pol = PoliticaHabitanteEGreedy({"epsilon": 0.0, "alpha": 0.5}, {"A": 2.0, "B": 1.0})
    pol.rng = random.Random(0)
    percepcion = {"restaurantes_asequibles": ["A", "B"]}
    assert pol.elegir(percepcion)["nombre"] == "A"
    for _ in range(5):
        pol.aprender(None, {"nombre": "A"}, -5.0, None)
    assert pol.elegir(percepcion)["nombre"] == "B"
