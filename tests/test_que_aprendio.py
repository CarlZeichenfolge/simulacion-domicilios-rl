"""Marcador de los duelos y análisis de decisiones (experimentos/que_aprendio.py)."""
import pytest

from experimentos.que_aprendio import registrar_decisiones, resumen
from sim.config import fusionar
from sim.metricas import resumen_episodio
from sim.model import CiudadModel
from sim.politicas import crear_politica_repartidor
from vistas.marcador import filas_marcador, texto_marcador


def _dia(config, politica, semilla=5, registrar=None):
    cfg = fusionar(config, {"repartidores": {"politica": politica}})
    pol = crear_politica_repartidor(cfg["repartidores"])
    if registrar is not None:
        registrar_decisiones(pol, registrar)
    modelo = CiudadModel(cfg, semilla=semilla, politica_repartidor=pol)
    modelo.correr()
    return modelo


@pytest.mark.parametrize("politica", ["aleatoria", "urgente"])
def test_registrar_decisiones_no_cambia_la_simulacion(config, politica):
    filas = []
    con = _dia(config, politica, registrar=filas)
    sin = _dia(config, politica)
    assert filas, "no se registró ninguna decisión"
    assert resumen_episodio(con) == resumen_episodio(sin)


def test_urgente_siempre_escoge_el_mas_viejo_y_nunca_espera(config):
    """Sirve de control del análisis: si esto fallara, el registro estaría mal."""
    filas = []
    _dia(config, "urgente", registrar=filas)
    import pandas as pd
    decisiones = pd.DataFrame(filas).assign(politica="urgente (regla fija)", dia=0)
    dias = pd.DataFrame([{"politica": "urgente (regla fija)", "tasa_cancelacion": 0.1, "recompensa_total": 1.0}])
    tabla = resumen(decisiones, dias.assign(dia=0))
    assert tabla.loc["urgente (regla fija)", "% de veces que espera"] == 0
    assert tabla.loc["urgente (regla fija)", "% acepta el más viejo"] == 100


def test_marcador_declara_ganador_segun_la_direccion_de_la_metrica(config):
    urgente, aleatoria = _dia(config, "urgente"), _dia(config, "aleatoria")
    filas = {f[0]: f for f in filas_marcador(aleatoria, urgente)}
    canc_a, canc_u = aleatoria.conteo["cancelados"], urgente.conteo["cancelados"]
    if canc_a != canc_u:  # en cancelaciones gana el MENOR
        esperado = "después" if canc_u / urgente.conteo["creados"] < canc_a / aleatoria.conteo["creados"] else "antes"
        assert filas["Pedidos cancelados"][3] == esperado
    texto = texto_marcador(aleatoria, urgente, "ANTES: aleatoria", "DESPUÉS: urgente")
    assert "FIN DEL DÍA" in texto and "Pedidos cancelados" in texto
