"""Entorno Gymnasium: cumple la interfaz oficial y es la MISMA simulación de siempre."""
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from sim.config import fusionar  # noqa: E402
from sim.entorno_gym import ID_ENTORNO, EntornoDomicilios, separar_observacion  # noqa: E402
from sim.metricas import resumen_episodio  # noqa: E402
from sim.model import CiudadModel  # noqa: E402


def _jugar(env, elegir, semilla_dia):
    """Juega un día completo; devuelve (recompensa sumada, decisiones, observaciones)."""
    obs, info = env.reset(options={"semilla_dia": semilla_dia})
    total, observaciones = 0.0, [obs]
    while not env.dia_terminado:
        obs, r, terminado, truncado, info = env.step(elegir(obs, info))
        total += r
        observaciones.append(obs)
        assert not terminado  # el día solo se acaba por tiempo
    assert truncado
    return total, len(observaciones) - 1, observaciones


def test_pasa_el_validador_oficial(config):
    check_env(EntornoDomicilios(config), skip_render_check=True)


def test_se_crea_con_gymnasium_make():
    env = gym.make(ID_ENTORNO)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_controlar_al_repartidor_desde_afuera_no_cambia_la_ciudad(config):
    """Si el agente de Gymnasium juega igual que la regla urgente (casilla 0 = el pedido
    más viejo), el día tiene que salir IDÉNTICO al de la simulación normal con urgente.
    Prueba que pausar el modelo en cada decisión no altera nada del azar ni del orden."""
    config = fusionar(config, {"gym": {"politica_otros": "urgente"}})
    env = EntornoDomicilios(config)
    total, decisiones, _ = _jugar(env, lambda obs, info: 0, semilla_dia=5)
    normal = CiudadModel(fusionar(config, {"repartidores": {"politica": "urgente"}}), semilla=5)
    normal.correr()
    assert decisiones > 0
    assert resumen_episodio(env.modelo) == resumen_episodio(normal)
    assert total == pytest.approx(normal.repartidores[0].recompensa_acumulada)


def test_misma_semilla_mismo_dia(config):
    env = EntornoDomicilios(config)
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    elegir = lambda rng: (lambda obs, info: int(rng.choice(np.flatnonzero(info["mascara_acciones"]))))  # noqa: E731
    a = _jugar(env, elegir(rng_a), semilla_dia=8)
    b = _jugar(env, elegir(rng_b), semilla_dia=8)
    assert a[0] == b[0] and a[1] == b[1]
    assert all((x == y).all() for x, y in zip(a[2], b[2]))


def test_casilla_vacia_cuenta_como_esperar(config):
    env = EntornoDomicilios(config)
    obs, info = env.reset(options={"semilla_dia": 5})
    vacia = int(np.flatnonzero(~info["mascara_acciones"])[0])  # primera casilla sin pedido
    _, r, _, _, info = env.step(vacia)
    assert info["accion_valida"] is False
    # step() corre hasta la siguiente decisión: la espera ya terminó y se cobró como rechazo.
    rep = config["repartidores"]
    assert env.externo.rechazos == 1
    assert info["minutos"] == rep["duracion_espera"]
    assert r == pytest.approx(-(rep["costo_espera_por_minuto"] * rep["duracion_espera"]
                                + rep["penalizacion_rechazo"]) * rep["escala_recompensa"])


def test_la_observacion_se_puede_desarmar(config):
    env = EntornoDomicilios(config)
    obs, info = env.reset(options={"semilla_dia": 5})
    contexto, pedidos = separar_observacion(obs, env.k)
    assert len(contexto) == 4
    assert len(pedidos) == int(info["mascara_acciones"][:env.k].sum()) > 0
