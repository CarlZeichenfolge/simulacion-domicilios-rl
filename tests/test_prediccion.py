"""Pruebas del predictor supervisado y de la separación de semillas (validación vs. prueba)."""
import pandas as pd

from sim.config import fusionar
from sim.model import CiudadModel
from experimentos.correr import DESPLAZAMIENTO_FASE, semilla_episodio
from experimentos.prediccion import CONJUNTOS, evaluar_tarea, generar_datos, semillas_por_conjunto


def test_cada_pedido_guarda_solo_lo_que_se_sabia_al_crearlo(config):
    modelo = CiudadModel(config, semilla=11)
    modelo.correr()
    columnas = set(config["prediccion"]["numericas"] + config["prediccion"]["categoricas"])
    assert modelo.pedidos, "la prueba necesita pedidos"
    for pedido in modelo.pedidos:
        foto = pedido.datos["foto"]
        assert set(foto) == columnas
        # La hora de la foto es la de creación, no la de entrega (sin fuga de datos).
        assert foto["hora"] == (pedido.creado % 1440) / 60


def test_tomar_la_foto_no_cambia_la_simulacion(config):
    """La foto no usa self.random: dos corridas con la misma semilla son idénticas."""
    a, b = CiudadModel(config, semilla=5), CiudadModel(config, semilla=5)
    a.correr()
    b.correr()
    assert a.conteo == b.conteo
    # pandas compara NaN == NaN como iguales (el faltante es parte de la foto).
    fotos_a = pd.DataFrame([p.datos["foto"] for p in a.pedidos])
    fotos_b = pd.DataFrame([p.datos["foto"] for p in b.pedidos])
    assert fotos_a.equals(fotos_b)


def test_conjuntos_del_predictor_no_se_cruzan_con_nada(config):
    cfg = config["prediccion"]
    bloques = semillas_por_conjunto(cfg)
    vistos = set()
    for conjunto in CONJUNTOS:
        semillas = set(bloques[conjunto])
        assert len(semillas) == cfg["dias"][conjunto]
        assert not semillas & vistos
        vistos |= semillas
    # Tampoco con los días de los experimentos 1-4 (semillas 1..5, todas sus fases).
    experimentos = {semilla_episodio(s, fase, d) for s in range(1, 6)
                    for fase in DESPLAZAMIENTO_FASE for d in range(400)}
    assert not vistos & experimentos


def test_prueba_final_usa_dias_distintos_a_validacion():
    validacion = {semilla_episodio(s, "evaluacion", d) for s in range(1, 6) for d in range(30)}
    prueba = {semilla_episodio(s, "prueba", d) for s in range(1, 6) for d in range(30)}
    entrenamiento = {semilla_episodio(s, "entrenamiento", d) for s in range(1, 6) for d in range(300)}
    assert not validacion & prueba
    assert not prueba & entrenamiento


def test_predictor_le_gana_a_la_linea_base(config):
    """Con pocos días y modelos chicos, igual debe superar a la media / 'siempre no'."""
    config = fusionar(config, {"prediccion": {
        "dias": {"entrenamiento": 6, "validacion": 2, "prueba": 2}, "arboles": 20,
        "candidatos": {"ridge_alpha": [1], "logistica_c": [1], "knn_k": [5], "bosque_profundidad": [5]}}})
    datos = generar_datos(config, procesos=1)
    assert set(datos["conjunto"]) == set(CONJUNTOS)
    reg = evaluar_tarea(datos, "regresion", config["prediccion"])["prueba"].set_index("familia")
    assert reg.loc["Random Forest", "MAE"] < reg.loc["Línea base (media)", "MAE"]
    clas = evaluar_tarea(datos, "clasificacion", config["prediccion"])["prueba"].set_index("familia")
    assert clas.loc["Línea base (siempre 'no')", "Recall"] == 0
    assert clas.loc["Regresión logística", "Recall"] > 0
