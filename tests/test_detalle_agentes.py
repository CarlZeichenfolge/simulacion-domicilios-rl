"""Lo que muestran los paneles del repartidor y del restaurante (vista 3D), probado sin ventana."""
import copy

import pytest

from sim.agentes.repartidor import EstadoRepartidor
from sim.config import fusionar
from sim.model import CiudadModel
from sim.politicas import crear_politica_repartidor
from vistas import detalle_habitante as dh
from vistas import detalle_repartidor as dr
from vistas import detalle_restaurante as dt


def _hasta(modelo, hora):
    while modelo.reloj.hora < hora:
        modelo.step()
    return modelo


def _modelo(config, politica="qlearning", congelada=False):
    config = fusionar(config, {"repartidores": {"politica": politica}})
    pol = crear_politica_repartidor(config["repartidores"])
    if congelada:
        pol.explorar = False
    return _hasta(CiudadModel(config, semilla=3, politica_repartidor=pol), 13)


def _mirar_todo(modelo):
    for r in modelo.repartidores:
        dr.estado(r), dr.trayecto(r), dr.que_sigue(r), dr.plata(r), dr.lista_pedidos(r)
        dr.opciones(r, 5), dr.exploro(r), dr.encabezado_decision(r), dr.frase_resultado(r), dr.balance(r)
    for rest in modelo.restaurantes.values():
        dt.ocupacion(rest), dt.en_cocina(rest), dt.en_cola(rest), dt.listos(rest), dt.plata(rest)
        dt.hoy(rest), dt.opinion(rest), dt.ficha(rest), dt.espera_nueva(rest), dt.lista_pedidos(rest)
    for h in modelo.habitantes:
        dh.que_sigue(h), dh.plata(modelo, h), dh.resumen_pedidos(modelo, h), dh.lista_pedidos(modelo, h)


def test_mirar_no_cambia_la_simulacion(config):
    """Los paneles valoran opciones con la política: no pueden gastar azar ni hacerla aprender."""
    modelo = _modelo(config)
    azar = modelo.random.getstate()
    q = copy.deepcopy(modelo.repartidores[0].politica.q)
    _mirar_todo(modelo)
    assert modelo.random.getstate() == azar
    assert (modelo.repartidores[0].politica.q == q).all()


def test_mirar_no_cambia_lo_que_pasa_despues(config):
    """Dos corridas iguales, una mirando los paneles cada minuto: terminan idénticas."""
    sin_mirar, mirando = _modelo(config), _modelo(config)
    for _ in range(120):
        _mirar_todo(mirando)
        sin_mirar.step()
        mirando.step()
    assert sin_mirar.conteo == mirando.conteo
    assert [r.recompensa_acumulada for r in sin_mirar.repartidores] == \
           [r.recompensa_acumulada for r in mirando.repartidores]


def test_el_repartidor_guarda_lo_que_decidio_y_como_le_fue(config):
    modelo = _modelo(config)
    decididos = [r for r in modelo.repartidores if r.ultima_decision]
    assert decididos, "a mediodía algún repartidor ya tuvo que escoger"
    for r in decididos:
        filas, sobran = dr.opciones(r, 50)
        assert sobran == 0
        assert len(filas) == len(r.ultima_decision["percepcion"]["pedidos"]) + 1  # + esperar
        assert sum(f["escogida"] for f in filas) == 1 and filas[0]["escogida"]
    terminados = [r for r in modelo.repartidores if r.ultimo_resultado]
    assert terminados
    for r in terminados:
        assert r.ultimo_resultado["motivo"] in ("entrego", "cancelado", "espero")
        assert dr.frase_resultado(r).startswith("(")


@pytest.mark.parametrize("politica", ["qlearning", "dqn"])
def test_congelada_escoge_la_opcion_de_mayor_valor(config, politica):
    """Sin explorar, lo que muestra el panel como escogido es lo de mayor Q: el panel dice la verdad."""
    modelo = _modelo(config, politica, congelada=True)
    for r in modelo.repartidores:
        if r.ultima_decision is None:
            continue
        filas, _ = dr.opciones(r, 50)
        assert filas[0]["valor"] == pytest.approx(max(f["valor"] for f in filas))
        assert not dr.exploro(r)


def test_reglas_fijas_no_tienen_valor(config):
    modelo = _modelo(config, "cercano")
    r = next(r for r in modelo.repartidores if r.ultima_decision)
    assert all(f["valor"] is None for f in dr.opciones(r, 50)[0])


def test_pedidos_y_plata_cuadran_con_el_modelo(config):
    """Cada pedido terminado aparece una vez en su repartidor, en su restaurante y en su habitante."""
    modelo = _modelo(config)
    terminados = [p for p in modelo.pedidos if p.terminado]
    con_moto = [p for p in terminados if p.repartidor is not None]
    assert sum(len(dr.lista_pedidos(r)) for r in modelo.repartidores) == len(con_moto)
    assert sum(len(dt.lista_pedidos(rest)) for rest in modelo.restaurantes.values()) == len(terminados)
    assert sum(len(dh.lista_pedidos(modelo, h)) for h in modelo.habitantes) == len(modelo.pedidos)
    for r in modelo.repartidores:
        entregados = [linea for linea in dr.lista_pedidos(r) if "cobró" in linea and "no cobró" not in linea]
        assert len(entregados) == r.entregas
        assert dh.pesos(r.ganancias - r.costos) in dr.plata(r)
    for h in modelo.habitantes:
        assert dh.plata(modelo, h).startswith(dh.pesos(h.dinero))


def test_trayecto_de_quien_lleva_un_pedido(config):
    modelo = _modelo(config)
    for _ in range(300):
        ocupados = [r for r in modelo.repartidores if r.estado != EstadoRepartidor.LIBRE and r.ruta]
        if ocupados:
            break
        modelo.step()
    r = ocupados[0]
    assert dr.minutos_de_ruta(r) > 0
    assert dr.cuadras_de_ruta(r) <= len(r.ruta)
    assert f"#{r.pedido.id}" in dr.estado(r)[0]
    assert any("de 60 min" in linea for linea in dr.trayecto(r))


def test_la_cocina_cuadra_con_el_restaurante(config):
    modelo = _modelo(config)
    for _ in range(60):
        modelo.step()
        for rest in modelo.restaurantes.values():
            ocupados, fogones = dt.ocupacion(rest)
            assert ocupados <= fogones
            assert len(dt.en_cocina(rest)) == ocupados
            assert len(dt.en_cola(rest)) == len(dt.esperando_fogon(rest))
    assert any(dt.hoy(rest) != "Hoy: ningún pedido todavía" for rest in modelo.restaurantes.values())


def test_envolver_no_cuenta_las_etiquetas_de_color():
    from vistas.vista_3d import envolver
    color = "<rgb(0.910,0.922,0.941)>"
    corta = f"{color}hola mundo"
    assert envolver(corta, 10) == [corta]  # 10 letras visibles: cabe aunque el texto tenga 34
    partes = envolver(f"{color}uno dos tres cuatro cinco seis", 12)
    assert len(partes) > 1
    assert all(len(p.replace(color, "")) <= 12 for p in partes)
    assert " ".join(p.strip() for p in partes) == f"{color}uno dos tres cuatro cinco seis"


def test_pesos():
    assert dh.pesos(80) == "$80"
    assert dh.pesos(6300) == "$6.3k"
    assert dh.pesos(-4060) == "−$4.1k"


def test_que_sigue_del_habitante(config):
    modelo = CiudadModel(config, semilla=3)
    while modelo.reloj.hora < 13:
        modelo.step()
    for h in modelo.habitantes:
        frase = dh.que_sigue(h)
        assert frase
        despierto_y_bien = (h.pedido is None and h.comida_en_curso is None and h.actividad != "dormir"
                            and h.hambre < h.cfg["umbral_hambre"])
        if despierto_y_bien:
            k = dh.minutos_hasta_hambre(h)
            assert k is None or k > 0
