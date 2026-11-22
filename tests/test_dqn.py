"""Red neuronal (DQN): características, reproducibilidad, persistencia y aprendizaje."""
import random

import pytest

torch = pytest.importorskip("torch")

from experimentos.correr import correr_corrida  # noqa: E402
from sim.config import fusionar  # noqa: E402
from sim.politicas import crear_politica_repartidor  # noqa: E402
from sim.politicas.repartidor_dqn import ENTRADAS, PoliticaRepartidorDQN  # noqa: E402


class _P:
    def __init__(self, id):
        self.id = id


def _percepcion(n_pedidos):
    pedidos = [{"pedido": _P(i), "distancia": 2 + i, "distancia_entrega": 3, "pago": 5000,
                "edad": 10 * i, "listo": i % 2 == 0} for i in range(n_pedidos)]
    return {"nodo": (0, 0), "hora": 13, "lluvia": False, "pedidos": pedidos}


@pytest.fixture
def config_dqn(config):
    """DQN que empieza a entrenar casi de una: así unos pocos días bastan para probarla."""
    return fusionar(config, {"repartidores": {"politica": "dqn",
                                              "dqn": {"minimo_para_entrenar": 64,
                                                      "actualizar_objetivo_cada": 20}}})


def test_una_fila_por_pedido_mas_esperar(config_dqn):
    pol = PoliticaRepartidorDQN(config_dqn["repartidores"]["dqn"])
    x = pol.caracteristicas(_percepcion(4))
    assert x.shape == (5, len(ENTRADAS))
    assert x[-1, ENTRADAS.index("es_esperar")] == 1.0
    assert (x[:-1, ENTRADAS.index("es_esperar")] == 0.0).all()
    # Exactamente un pedido es "el más viejo" (edad 30) y uno "el más cercano" (distancia 2).
    assert x[:-1, ENTRADAS.index("es_el_mas_viejo")].tolist() == [0, 0, 0, 1]
    assert x[:-1, ENTRADAS.index("es_el_mas_cercano")].tolist() == [1, 0, 0, 0]


def test_sin_pedidos_solo_puede_esperar(config_dqn):
    pol = PoliticaRepartidorDQN(config_dqn["repartidores"]["dqn"])
    pol.vincular(type("Modelo", (), {"random": random.Random(0)})())
    assert pol.elegir(_percepcion(0))["tipo"] == "esperar"


def test_la_fabrica_crea_la_red(config_dqn):
    assert isinstance(crear_politica_repartidor(config_dqn["repartidores"]), PoliticaRepartidorDQN)


def _entrenar(config, dias=3):
    filas, _, pol = correr_corrida(config, semilla=1, episodios_entrenamiento=dias, episodios_evaluacion=1)
    return filas, pol


def test_entrena_y_es_reproducible(config_dqn):
    filas_a, pol_a = _entrenar(config_dqn)
    filas_b, pol_b = _entrenar(config_dqn)
    assert pol_a.pasos_gradiente > 0, "la red nunca dio un paso de gradiente"
    assert filas_a == filas_b
    for pa, pb in zip(pol_a.red.parameters(), pol_b.red.parameters()):
        assert torch.equal(pa, pb)
    assert len(pol_a.historial_perdida) == 3


def test_no_aprende_en_evaluacion(config_dqn):
    pol = PoliticaRepartidorDQN(config_dqn["repartidores"]["dqn"])
    pol.explorar = False
    accion = {"x": pol.caracteristicas(_percepcion(2))[0]}
    pol.aprender(_percepcion(2), accion, 1.0, _percepcion(1), 5)
    assert len(pol.memoria) == 0


def test_guarda_y_carga(config_dqn, tmp_path):
    _, pol = _entrenar(config_dqn, dias=2)
    pol.guardar(tmp_path / "red.pt")
    otra = PoliticaRepartidorDQN(config_dqn["repartidores"]["dqn"])
    otra.cargar(tmp_path / "red.pt")
    x = pol.caracteristicas(_percepcion(3))
    assert (pol.valores_q(x) == otra.valores_q(x)).all()
    assert otra.epsilon == pol.epsilon


def test_cargar_con_otra_arquitectura_falla(config_dqn, tmp_path):
    pol = PoliticaRepartidorDQN(config_dqn["repartidores"]["dqn"])
    pol.guardar(tmp_path / "red.pt")
    otra_cfg = fusionar(config_dqn, {"repartidores": {"dqn": {"capas_ocultas": [8]}}})
    with pytest.raises(ValueError):
        PoliticaRepartidorDQN(otra_cfg["repartidores"]["dqn"]).cargar(tmp_path / "red.pt")
