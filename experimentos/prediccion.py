"""Predictor supervisado: lo que la app le diría al cliente cuando hace el pedido.

Uso:
    python -m experimentos.prediccion
    python -m experimentos.prediccion --procesos 4

Dos preguntas sobre el MISMO conjunto de datos (Unidad 4 del curso: el tipo de
problema lo define la pregunta, no el dataset):
  - Regresión:      ¿cuántos minutos tardará en llegar?        (y = tiempo_entrega)
  - Clasificación:  ¿se va a cancelar?                          (y = cancelado, 0/1)

De dónde salen los datos: la propia simulación. Cada pedido guarda una "foto"
de lo que se sabía cuando nació (CiudadModel._foto_pedido); la respuesta y se
conoce después, cuando el pedido se entrega o se cancela. Por eso aquí SÍ es
aprendizaje supervisado: hay ejemplos con la respuesta correcta, cosa que no
existe para la decisión del repartidor (por eso él usa Q-learning).

Decisiones que siguen la metodología del curso (y por qué):
  - Partición POR DÍA (entrenamiento / validación / prueba), no por pedido.
    Los pedidos de un mismo día comparten lluvia, trancones y repartidores; si
    se mezclaran entre conjuntos, la prueba "vería" días que ya conoce
    (una forma sutil de fuga de datos).
  - Solo variables conocidas al crear el pedido (Frente 11, data leakage).
  - Todo el preprocesamiento va dentro de un Pipeline que se ajusta solo con
    entrenamiento: imputación por mediana + indicador de faltante, escalado
    estándar (KNN mide distancias) y one-hot para el restaurante.
  - Hiperparámetros (alpha, C, K, max_depth) se eligen en VALIDACIÓN; después
    se reentrena con entrenamiento + validación y se mide UNA sola vez en PRUEBA.
  - Siempre contra una línea base (DummyRegressor / DummyClassifier).
  - Clasificación con clases desbalanceadas (~10 % cancela): se elige por F1 y
    se reportan precisión y recall, porque la exactitud engaña (decir siempre
    "no se cancela" ya acierta ~90 %).

Limitación conocida (para la sustentación): el tiempo de entrega solo se observa
en pedidos entregados; los que habrían tardado más de 60 min se cancelaron y no
aparecen en la regresión (dato censurado). Por eso las dos preguntas van juntas:
primero "¿se cancelará?", y si no, "¿cuánto tardará?".
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from sim.config import cargar_config, fusionar  # noqa: E402
from sim.model import CiudadModel  # noqa: E402
from sim.pedidos import EstadoPedido  # noqa: E402

CONJUNTOS = ("entrenamiento", "validacion", "prueba")
RESULTADOS = RAIZ / "resultados"
OBJETIVO = {"regresion": "tiempo_entrega", "clasificacion": "cancelado"}
# Métrica con la que se elige el hiperparámetro en validación, y si más alto es mejor.
CRITERIO = {"regresion": ("MAE", False), "clasificacion": ("F1", True)}


# ---------------------------------------------------------------------- datos
def semillas_por_conjunto(cfg_pred: dict) -> dict[str, list[int]]:
    """Días consecutivos de un bloque de semillas propio, repartidos por conjunto."""
    base, inicio, salida = cfg_pred["semilla_base"] * 100_000, 0, {}
    for conjunto in CONJUNTOS:
        cantidad = cfg_pred["dias"][conjunto]
        salida[conjunto] = [base + inicio + i for i in range(cantidad)]
        inicio += cantidad
    return salida


def pedidos_de_un_dia(args) -> list[dict]:
    config, semilla, conjunto = args
    modelo = CiudadModel(config, semilla=semilla)
    modelo.correr()
    filas = []
    for pedido in modelo.pedidos:
        if not pedido.terminado:
            continue  # el día acabó antes de saber si llegaba: no tiene respuesta y
        fila = dict(pedido.datos["foto"])
        fila.update(dia=semilla, conjunto=conjunto,
                    cancelado=int(pedido.estado == EstadoPedido.CANCELADO),
                    tiempo_entrega=pedido.tiempo_entrega)
        filas.append(fila)
    return filas


def generar_datos(config: dict, procesos: int | None = None) -> pd.DataFrame:
    """Simula los días de los tres conjuntos con la política "desplegada"."""
    cfg_pred = config["prediccion"]
    config = fusionar(config, {"repartidores": {"politica": cfg_pred["politica"]},
                               "metricas": {"recolectar_cada": 0}})
    trabajos = [(config, s, c) for c, semillas in semillas_por_conjunto(cfg_pred).items() for s in semillas]
    procesos = procesos or max(1, (os.cpu_count() or 2) - 1)
    if procesos == 1:
        lotes = map(pedidos_de_un_dia, trabajos)
    else:  # map conserva el orden: el resultado no depende de qué proceso termina primero
        with ProcessPoolExecutor(max_workers=procesos) as pool:
            lotes = list(pool.map(pedidos_de_un_dia, trabajos))
    return pd.DataFrame([fila for lote in lotes for fila in lote])


# --------------------------------------------------------------------- modelos
def preprocesador(cfg_pred: dict) -> ColumnTransformer:
    numericas = Pipeline([
        # Faltante = "no había repartidor libre": se imputa y además se marca,
        # porque que falte el dato ya dice algo (Frente 2, faltante informativo).
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("escalar", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numericas, cfg_pred["numericas"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cfg_pred["categoricas"]),
    ])


def candidatos(tarea: str, cfg_pred: dict) -> dict[str, list[tuple[str, object]]]:
    """Por familia, la lista (hiperparámetro, estimador) que se compara en validación."""
    c = cfg_pred["candidatos"]
    semilla, arboles, peso = cfg_pred["semilla_modelos"], cfg_pred["arboles"], cfg_pred["peso_clases"]
    if tarea == "regresion":
        return {
            "Línea base (media)": [("-", DummyRegressor(strategy="mean"))],
            "Regresión lineal": [("-", LinearRegression())],
            "Ridge": [(f"alpha={a}", Ridge(alpha=a)) for a in c["ridge_alpha"]],
            "KNN": [(f"K={k}", KNeighborsRegressor(n_neighbors=k)) for k in c["knn_k"]],
            "Random Forest": [(f"max_depth={d}", RandomForestRegressor(
                n_estimators=arboles, max_depth=d, random_state=semilla, n_jobs=-1))
                for d in c["bosque_profundidad"]],
        }
    return {
        "Línea base (siempre 'no')": [("-", DummyClassifier(strategy="most_frequent"))],
        "Regresión logística": [(f"C={v}", LogisticRegression(C=v, class_weight=peso, max_iter=2000))
                                for v in c["logistica_c"]],
        "KNN": [(f"K={k}", KNeighborsClassifier(n_neighbors=k)) for k in c["knn_k"]],
        "Random Forest": [(f"max_depth={d}", RandomForestClassifier(
            n_estimators=arboles, max_depth=d, class_weight=peso, random_state=semilla, n_jobs=-1))
            for d in c["bosque_profundidad"]],
    }


def metricas(tarea: str, modelo, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = modelo.predict(X)
    if tarea == "regresion":
        return {"MAE": mean_absolute_error(y, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
                "R2": r2_score(y, pred)}
    prob = modelo.predict_proba(X)[:, 1]
    return {"Exactitud": accuracy_score(y, pred),
            "Precisión": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y, prob)}


def evaluar_tarea(datos: pd.DataFrame, tarea: str, cfg_pred: dict) -> dict:
    """Elige hiperparámetros en validación y mide una sola vez en prueba."""
    objetivo = OBJETIVO[tarea]
    df = datos if tarea == "clasificacion" else datos[datos["cancelado"] == 0]
    columnas = cfg_pred["numericas"] + cfg_pred["categoricas"]
    X, y = df[columnas], df[objetivo]
    en = {c: (df["conjunto"] == c).to_numpy() for c in CONJUNTOS}
    ajuste = en["entrenamiento"] | en["validacion"]
    nombre_criterio, mayor_es_mejor = CRITERIO[tarea]

    filas_validacion, filas_prueba, modelos = [], [], {}
    for familia, opciones in candidatos(tarea, cfg_pred).items():
        mejor = None
        for etiqueta, estimador in opciones:
            modelo = Pipeline([("prep", preprocesador(cfg_pred)), ("modelo", clone(estimador))])
            modelo.fit(X[en["entrenamiento"]], y[en["entrenamiento"]])
            val = metricas(tarea, modelo, X[en["validacion"]], y[en["validacion"]])
            ent = metricas(tarea, modelo, X[en["entrenamiento"]], y[en["entrenamiento"]])
            filas_validacion.append({"familia": familia, "hiperparametro": etiqueta, **val,
                                     f"{nombre_criterio}_entrenamiento": ent[nombre_criterio]})
            puntaje = val[nombre_criterio] if mayor_es_mejor else -val[nombre_criterio]
            if mejor is None or puntaje > mejor[0]:
                mejor = (puntaje, etiqueta, estimador, val[nombre_criterio])
        _, etiqueta, estimador, criterio_val = mejor
        # Modelo final de la familia: el mejor hiperparámetro reentrenado con más datos.
        final = Pipeline([("prep", preprocesador(cfg_pred)), ("modelo", clone(estimador))])
        final.fit(X[ajuste], y[ajuste])
        filas_prueba.append({"familia": familia, "hiperparametro": etiqueta,
                             f"{nombre_criterio}_validacion": criterio_val,
                             **metricas(tarea, final, X[en["prueba"]], y[en["prueba"]])})
        modelos[familia] = final

    validacion, prueba = pd.DataFrame(filas_validacion), pd.DataFrame(filas_prueba)
    # El "ganador" se escoge por validación, no por prueba (si no, la prueba deja de ser prueba).
    columna = f"{nombre_criterio}_validacion"
    ganador = prueba.sort_values(columna, ascending=not mayor_es_mejor).iloc[0]["familia"]
    return {"tarea": tarea, "validacion": validacion, "prueba": prueba, "modelos": modelos,
            "ganador": ganador, "X_prueba": X[en["prueba"]], "y_prueba": y[en["prueba"]]}


def importancias(modelo: Pipeline, cantidad: int) -> pd.Series:
    """Importancia de cada variable según el Random Forest (nombres legibles)."""
    nombres = [n.split("__", 1)[1].replace("missingindicator_", "falta_")
               for n in modelo["prep"].get_feature_names_out()]
    serie = pd.Series(modelo["modelo"].feature_importances_, index=nombres)
    return serie.sort_values(ascending=False).head(cantidad)


# --------------------------------------------------------------------- gráficas
def graficar(reg: dict, clas: dict, guardar: bool = True) -> None:
    """Con guardar=False solo muestra las figuras (notebook): no pisa los PNG de la corrida completa."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from experimentos.graficas import SERIES, SUPERFICIE, TENUE, TINTA, TINTA_2, _guardar, estilo

    estilo()
    RESULTADOS.mkdir(exist_ok=True)

    def terminar(fig, nombre: str) -> None:
        if guardar:
            _guardar(fig, nombre)
        else:
            plt.show()

    # 1) Regresión: error en prueba, curva de K y qué variables usa el bosque.
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    p = reg["prueba"].sort_values("MAE", ascending=False)
    colores = [TENUE if f.startswith("Línea base") else SERIES[0] for f in p["familia"]]
    barras = ejes[0].barh(p["familia"], p["MAE"], color=colores, height=0.6)
    ejes[0].bar_label(barras, fmt="%.1f", padding=3, color=TINTA_2, fontsize=9)
    ejes[0].set_title("Error absoluto medio en PRUEBA (min)")
    ejes[0].grid(axis="y", visible=False)
    ejes[0].grid(axis="x", visible=True)

    knn = reg["validacion"][reg["validacion"]["familia"] == "KNN"].copy()
    knn["K"] = knn["hiperparametro"].str.slice(2).astype(int)
    ejes[1].plot(knn["K"], knn["MAE_entrenamiento"], marker="o", color=SERIES[1], label="Entrenamiento")
    ejes[1].plot(knn["K"], knn["MAE"], marker="o", color=SERIES[2], label="Validación")
    mejor_k = knn.loc[knn["MAE"].idxmin()]
    ejes[1].annotate(f"K elegido = {int(mejor_k['K'])}", (mejor_k["K"], mejor_k["MAE"]),
                     xytext=(0, 16), textcoords="offset points", ha="center", color=TINTA_2, fontsize=9,
                     arrowprops={"arrowstyle": "-", "color": TENUE})
    ejes[1].set_xscale("log")
    ejes[1].set_xticks(knn["K"], [str(k) for k in knn["K"]])
    ejes[1].set_xlabel("K (vecinos), escala log")
    ejes[1].set_ylabel("MAE (min)")
    ejes[1].set_title("KNN: K pequeño memoriza, K grande generaliza de más")
    ejes[1].legend(loc="lower right")

    imp = importancias(reg["modelos"]["Random Forest"], 8).iloc[::-1]
    ejes[2].barh(imp.index, imp.values, color=SERIES[3], height=0.6)
    ejes[2].set_title("Variables que más usa el Random Forest")
    ejes[2].set_xlabel("Importancia (reducción de impureza)")
    ejes[2].grid(axis="y", visible=False)
    ejes[2].grid(axis="x", visible=True)
    fig.suptitle("¿Cuánto tardará el pedido? (regresión)", x=0.01, ha="left", fontweight="bold", color=TINTA)
    fig.tight_layout()
    terminar(fig, "05_prediccion_tiempo")

    # 2) Clasificación: precisión / recall / F1 en prueba y matriz de confusión del ganador.
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    p = clas["prueba"]
    x = np.arange(len(p))
    ancho = 0.26
    for i, (metrica, color) in enumerate(zip(["Precisión", "Recall", "F1"], SERIES[:3])):
        ejes[0].bar(x + (i - 1) * ancho, p[metrica], width=ancho - 0.02, color=color, label=metrica)
    ejes[0].set_xticks(x, [f.replace(" (", "\n(") for f in p["familia"]])
    ejes[0].set_ylim(0, 1)
    ejes[0].set_title("Detectar cancelaciones, en PRUEBA")
    ejes[0].legend(loc="upper left", ncols=3)

    ganador = clas["ganador"]
    matriz = confusion_matrix(clas["y_prueba"], clas["modelos"][ganador].predict(clas["X_prueba"]))
    mapa = LinearSegmentedColormap.from_list("azul", [SUPERFICIE, SERIES[0]])
    ejes[1].imshow(matriz, cmap=mapa)
    for (i, j), valor in np.ndenumerate(matriz):
        oscuro = valor > matriz.max() / 2
        ejes[1].text(j, i, f"{valor}", ha="center", va="center", fontsize=13, fontweight="bold",
                     color="white" if oscuro else TINTA)
    ejes[1].set_xticks([0, 1], ["dice: llega", "dice: se cancela"])
    ejes[1].set_yticks([0, 1], ["llegó", "se canceló"])
    ejes[1].grid(False)
    ejes[1].set_title(f"Matriz de confusión: {ganador}")
    fig.suptitle("¿Se cancelará el pedido? (clasificación)", x=0.01, ha="left", fontweight="bold", color=TINTA)
    fig.tight_layout()
    terminar(fig, "05_prediccion_cancelacion")


# ------------------------------------------------------------------------ main
def correr(config: dict | None = None, procesos: int | None = None, guardar: bool = True) -> dict:
    config = config or cargar_config()
    cfg_pred = config["prediccion"]
    datos = generar_datos(config, procesos)
    resultados = {tarea: evaluar_tarea(datos, tarea, cfg_pred) for tarea in OBJETIVO}
    resultados["datos"] = datos
    if guardar:
        RESULTADOS.mkdir(exist_ok=True)
        datos.to_csv(RESULTADOS / "05_prediccion_datos.csv", index=False)
        for tarea in OBJETIVO:
            for parte in ("validacion", "prueba"):
                resultados[tarea][parte].to_csv(RESULTADOS / f"05_prediccion_{tarea}_{parte}.csv", index=False)
    graficar(resultados["regresion"], resultados["clasificacion"], guardar)
    return resultados


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--procesos", type=int, default=None)
    args = parser.parse_args()
    r = correr(procesos=args.procesos)
    datos = r["datos"]
    print("\nPedidos por conjunto:", datos["conjunto"].value_counts().to_dict(),
          f"| cancelados: {datos['cancelado'].mean():.1%}")
    for tarea in OBJETIVO:
        print(f"\n== {tarea} (ganador por validación: {r[tarea]['ganador']}) ==")
        print(r[tarea]["prueba"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
