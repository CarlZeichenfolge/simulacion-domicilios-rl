"""Lo que muestra el panel de un habitante (vista 3D), probado sin ventana."""
import copy

from sim.model import CiudadModel
from vistas import detalle_habitante as d


def _modelo_hasta_mediodia(config):
    modelo = CiudadModel(config, semilla=3)
    while modelo.reloj.hora < 13:
        modelo.step()
    return modelo


def test_cada_decision_queda_registrada_con_lo_que_comparo(config):
    modelo = _modelo_hasta_mediodia(config)
    decididos = [h for h in modelo.habitantes if h.ultima_decision]
    assert decididos, "a mediodía alguien ya tuvo que decidir qué comer"
    for h in decididos:
        dec = h.ultima_decision
        assert dec["hambre"] >= h.cfg["umbral_hambre"]
        if dec["opciones"]:
            # escogió la de mayor utilidad: la regla del agente basado en utilidad
            assert dec["eleccion"] == max(dec["opciones"], key=dec["opciones"].get)
        if dec["eleccion"] == "pedir":
            assert dec["restaurante"] in modelo.restaurantes
        frase = d.frase_decision(dec)
        assert frase.startswith("(") and "):" in frase


def test_mirar_al_habitante_no_cambia_la_simulacion(config):
    """El panel calcula 'si decidiera ya': no puede gastar azar ni tocar al agente."""
    modelo = _modelo_hasta_mediodia(config)
    azar = modelo.random.getstate()
    antes = [(h.hambre, h.energia, h.dinero, copy.deepcopy(h.politica.q)) for h in modelo.habitantes]
    for h in modelo.habitantes:
        d.estado(h)
        d.opciones_ahora(h)
        d.lo_que_aprendio(h)
        d.comidas_de_hoy(modelo, h)
    assert modelo.random.getstate() == azar
    assert antes == [(h.hambre, h.energia, h.dinero, copy.deepcopy(h.politica.q)) for h in modelo.habitantes]


def test_estado_dice_lo_que_hace(config):
    modelo = _modelo_hasta_mediodia(config)
    claves = {d.estado(h)[1] for h in modelo.habitantes}
    assert claves <= {"esperando", "cocinando", "afuera", "durmiendo", "hambre", "bien"}
    for h in modelo.habitantes:
        frase, clave = d.estado(h)
        if h.pedido is not None:
            assert clave == "esperando" and h.pedido.restaurante.nombre in frase


def test_comidas_de_hoy_sale_del_registro_del_modelo(config):
    modelo = _modelo_hasta_mediodia(config)
    for h in modelo.habitantes:
        propias = [c for c in modelo.comidas if c["habitante"] == h.unique_id and c["dia"] == modelo.reloj.dia]
        assert len(d.comidas_de_hoy(modelo, h)) == len(propias)


def test_frase_sin_decision():
    assert "no ha tenido que decidir" in d.frase_decision(None)


def test_frase_corta():
    dec = {"minuto_del_dia": 8 * 60 + 28, "lluvia": True, "hambre": 70, "opciones": {"cocinar": 6.1},
           "eleccion": "pedir", "restaurante": "Sushi", "exploro": True}
    assert d.frase_decision(dec) == "(08:28, lloviendo): Pedir a Sushi (explorando)"
