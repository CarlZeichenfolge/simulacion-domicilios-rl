"""Genera las gráficas comparativas a partir de los CSV de resultados/.

Uso:
    python -m experimentos.graficas            # todas las que tengan CSV
    python -m experimentos.graficas 01_politicas

Decisiones de diseño (por si el profe pregunta):
- Barras de error = desviación estándar ENTRE SEMILLAS (no entre episodios):
  cada semilla es una réplica independiente del experimento.
- Cada política / restaurante tiene siempre el mismo color en todas las
  gráficas, para que el lector no tenga que reaprender la leyenda.
- Un solo eje por gráfica: métricas con escalas distintas van en paneles separados.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # sin ventana en terminal; en un notebook (Colab) se deja el backend inline
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "resultados"

# Paleta categórica en orden fijo (validada para daltonismo en pares adyacentes).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TINTA, TINTA_2, TENUE = "#0b0b0b", "#52514e", "#898781"
GRILLA, EJE, SUPERFICIE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

COLOR = {  # el color sigue a la entidad, no a su posición en la gráfica
    "aleatoria": SERIES[0], "cercano": SERIES[1], "urgente": SERIES[2],
    "qlearning": SERIES[3], "qlearning_sin_rechazo": SERIES[4], "dqn": SERIES[6],
    "Pizza": SERIES[0], "Hamburguesas": SERIES[1], "Sushi": SERIES[2],
    # Agentes del entorno Gymnasium: mismo color que su equivalente en la simulación.
    "aleatorio": SERIES[0],
}

ETIQUETAS = {
    "entregados": "Pedidos entregados por día",
    "tasa_cancelacion": "Tasa de cancelación",
    "tiempo_entrega_promedio": "Tiempo de entrega (min)",
    "ganancia_neta_por_repartidor": "Ganancia neta por repartidor (miles $)",
    "satisfaccion_promedio": "Satisfacción promedio",
    "pedidos_creados": "Pedidos creados por día",
    "comidas_salir": "Veces que salieron a comer",
    "recompensa_total": "Recompensa total del episodio",
}


def estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": TINTA, "axes.labelcolor": TINTA_2, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.edgecolor": EJE, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRILLA, "grid.linewidth": 0.8,
        "xtick.color": TENUE, "ytick.color": TENUE, "legend.frameon": False,
        "lines.linewidth": 2,
    })


def _guardar(fig, nombre: str) -> None:
    ruta = RESULTADOS / f"{nombre}.png"
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  {ruta.relative_to(RAIZ)}")


def _por_semilla(df: pd.DataFrame, columnas: list[str], fase: str = "evaluacion") -> pd.DataFrame:
    """Promedio de los episodios de evaluación (o de prueba) de cada (variante, semilla)."""
    evaluacion = df[df["fase"] == fase].copy()
    evaluacion["ganancia_neta_por_repartidor"] /= 1000
    return evaluacion.groupby(["variante", "semilla_corrida"])[columnas].mean().reset_index()


def barras(df: pd.DataFrame, nombre: str, metricas: list[str], orden: list[str], titulo: str,
           fase: str = "evaluacion") -> None:
    """Paneles pequeños: una métrica por panel, una barra por variante, error = std entre semillas."""
    datos = _por_semilla(df, metricas, fase)
    orden = [v for v in orden if v in set(datos["variante"])]
    resumen = datos.groupby("variante")[metricas].agg(["mean", "std"]).loc[orden]
    fig, ejes = plt.subplots(1, len(metricas), figsize=(3.3 * len(metricas), 3.6))
    for eje, metrica in zip(ejes, metricas):
        medias, desv = resumen[(metrica, "mean")], resumen[(metrica, "std")].fillna(0)
        colores = [COLOR.get(v, SERIES[i % len(SERIES)]) for i, v in enumerate(orden)]
        x = range(len(orden))
        eje.bar(x, medias, width=0.7, color=colores, edgecolor=SUPERFICIE, linewidth=2,
                yerr=desv, error_kw={"ecolor": TINTA_2, "elinewidth": 1, "capsize": 3})
        for xi, m, d in zip(x, medias, desv):
            texto = f"{m:.1%}" if metrica == "tasa_cancelacion" else f"{m:.1f}"
            arriba = m >= 0
            eje.annotate(texto, (xi, m + d if arriba else m - d), xytext=(0, 3 if arriba else -3),
                         textcoords="offset points", ha="center", va="bottom" if arriba else "top",
                         fontsize=8, color=TINTA_2)
        eje.axhline(0, color=EJE, linewidth=1)
        eje.set_xticks(list(x), [v.replace("_", "\n") for v in orden], fontsize=8)
        eje.set_title(ETIQUETAS.get(metrica, metrica), fontsize=10)
        if metrica == "tasa_cancelacion":
            eje.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _guardar(fig, nombre)


def curva_aprendizaje(df: pd.DataFrame, nombre: str, ventana: int = 20,
                      titulo: str = "Curva de aprendizaje del repartidor con Q-learning") -> None:
    """Media móvil por episodio, promedio entre semillas, banda = ±1 std entre semillas.
    Las heurísticas aparecen como líneas punteadas de referencia (su nivel en evaluación)."""
    entrenamiento = df[df["fase"] == "entrenamiento"]
    if entrenamiento.empty:
        return
    referencias = _por_semilla(df, ["recompensa_total", "tasa_cancelacion"]).groupby("variante").mean()
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
    for eje, metrica in zip(ejes, ["recompensa_total", "tasa_cancelacion"]):
        for variante, grupo in entrenamiento.groupby("variante"):
            tabla = grupo.pivot_table(index="episodio", columns="semilla_corrida", values=metrica)
            suave = tabla.rolling(ventana, min_periods=1).mean()
            media, desv = suave.mean(axis=1), suave.std(axis=1).fillna(0)
            color = COLOR.get(variante, SERIES[3])
            eje.fill_between(media.index, media - desv, media + desv, color=color, alpha=0.15, linewidth=0)
            eje.plot(media.index, media, color=color, label=f"{variante} (entrenando)")
        for variante in referencias.index:
            if variante in set(entrenamiento["variante"]):
                continue
            eje.axhline(referencias.loc[variante, metrica], color=COLOR.get(variante, TENUE),
                        linestyle="--", linewidth=1.2, label=f"{variante} (fija)")
        eje.set_xlabel("Episodio (1 día simulado)")
        eje.set_title(ETIQUETAS[metrica] + f" (media móvil {ventana})", fontsize=10)
        if metrica == "tasa_cancelacion":
            eje.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ejes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _guardar(fig, nombre)


def perdida_red(df: pd.DataFrame, nombre: str, ventana: int = 10) -> None:
    """Pérdida de Huber por episodio de la red.

    Ojo: en DQN la pérdida NO mide qué tan buena es la política (eso lo dice la
    recompensa). Al inicio los valores Q son casi cero y los objetivos también,
    así que el error es pequeño; a medida que la red aprende valores más grandes
    el error absoluto crece y luego se queda estable. No llega a cero porque el
    objetivo r + γ^k·max Q' cambia con cada copia de la red objetivo y la
    recompensa es ruidosa (lluvia, propinas). Lo que se verifica aquí es que
    NO DIVERJA (que no crezca sin control), el riesgo clásico de DQN."""
    datos = df[(df["fase"] == "entrenamiento") & df.get("perdida_red", pd.Series(dtype=float)).notna()]
    if datos.empty:
        return
    tabla = datos.pivot_table(index="episodio", columns="semilla_corrida", values="perdida_red")
    suave = tabla.rolling(ventana, min_periods=1).mean()
    media, desv = suave.mean(axis=1), suave.std(axis=1).fillna(0)
    fig, eje = plt.subplots(figsize=(6.5, 3.6))
    eje.fill_between(media.index, media - desv, media + desv, color=COLOR["dqn"], alpha=0.15, linewidth=0)
    eje.plot(media.index, media, color=COLOR["dqn"])
    eje.set_xlabel("Episodio (1 día simulado)")
    eje.set_ylabel("Pérdida de Huber")
    eje.set_title(f"Media de 5 semillas, media móvil {ventana}; banda = ±1 desviación", fontsize=9,
                  color=TINTA_2, fontweight="normal")
    fig.suptitle("La pérdida de la red no diverge: se estabiliza tras el arranque", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _guardar(fig, nombre)


def gymnasium(nombre: str = "06_gymnasium", ventana: int = 20) -> None:
    """Izquierda: la red entrenada SOLO con env.reset()/env.step(). Derecha: prueba vs otros agentes."""
    ruta_ent, ruta_prueba = RESULTADOS / "gym_entrenamiento.csv", RESULTADOS / "gym_prueba.csv"
    if not (ruta_ent.exists() and ruta_prueba.exists()):
        print("  (no existe gym_entrenamiento.csv / gym_prueba.csv, se omite)")
        return
    ent, prueba = pd.read_csv(ruta_ent), pd.read_csv(ruta_prueba)
    fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    suave = ent.set_index("dia")["recompensa_repartidor0"].rolling(ventana, min_periods=1).mean()
    izq.plot(suave.index, suave, color=COLOR["dqn"], label="dqn (entrenando)")
    resumen = prueba.groupby("agente")["recompensa_repartidor0"].agg(["mean", "std"])
    for agente in ["aleatorio", "urgente"]:
        izq.axhline(resumen.loc[agente, "mean"], color=COLOR[agente], linestyle="--", linewidth=1.2,
                    label=f"{agente} (fijo)")
    izq.set_xlabel("Día de entrenamiento")
    izq.set_title(f"Recompensa del repartidor 0 (media móvil {ventana})", fontsize=10)
    izq.legend(fontsize=8, loc="lower right")
    orden = ["aleatorio", "urgente", "dqn"]
    x = range(len(orden))
    der.bar(x, resumen.loc[orden, "mean"], width=0.7, color=[COLOR[a] for a in orden],
            edgecolor=SUPERFICIE, linewidth=2, yerr=resumen.loc[orden, "std"],
            error_kw={"ecolor": TINTA_2, "elinewidth": 1, "capsize": 3})
    for xi, a in zip(x, orden):
        m, d = resumen.loc[a, "mean"], resumen.loc[a, "std"]
        der.annotate(f"{m:.1f}", (xi, m + d), xytext=(0, 3), textcoords="offset points", ha="center",
                     fontsize=8, color=TINTA_2)
    der.set_xticks(list(x), orden, fontsize=9)
    der.set_title("Prueba: 30 días nunca vistos (error = desv. entre días)", fontsize=10)
    fig.suptitle("La red entrenada solo con la interfaz de Gymnasium (reset / step)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _guardar(fig, nombre)


def lineas_cantidad(df: pd.DataFrame, nombre: str) -> None:
    datos = _por_semilla(df, ["tiempo_entrega_promedio", "tasa_cancelacion", "satisfaccion_promedio"])
    datos["cantidad"] = datos["variante"].str.extract(r"(\d+)").astype(int)
    resumen = datos.drop(columns=["variante", "semilla_corrida"]).groupby("cantidad").agg(["mean", "std"])
    fig, ejes = plt.subplots(1, 3, figsize=(11, 3.4))
    for eje, metrica in zip(ejes, ["tiempo_entrega_promedio", "tasa_cancelacion", "satisfaccion_promedio"]):
        media, desv = resumen[(metrica, "mean")], resumen[(metrica, "std")].fillna(0)
        eje.errorbar(media.index, media, yerr=desv, color=SERIES[0], marker="o", markersize=7,
                     capsize=3, markeredgecolor=SUPERFICIE, markeredgewidth=2)
        eje.set_xticks(list(media.index))
        eje.set_xlabel("Repartidores")
        eje.set_title(ETIQUETAS[metrica], fontsize=10)
        if metrica == "tasa_cancelacion":
            eje.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle("Más repartidores: menos espera y menos cancelaciones", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _guardar(fig, nombre)


def preferencias(nombre: str) -> None:
    """Fracción de domicilios a cada restaurante por día, con y sin aprendizaje."""
    ruta = RESULTADOS / f"{nombre}_comidas.csv"
    if not ruta.exists():
        return
    comidas = pd.read_csv(ruta)
    pedidos = comidas[comidas["tipo"] == "pedir"]
    variantes = sorted(pedidos["variante"].unique())
    fig, ejes = plt.subplots(1, len(variantes), figsize=(5.5 * len(variantes), 3.8), sharey=True)
    for eje, variante in zip(ejes, variantes):
        sub = pedidos[pedidos["variante"] == variante]
        conteo = sub.groupby(["semilla_corrida", "dia", "restaurante"]).size().unstack(fill_value=0)
        fraccion = conteo.div(conteo.sum(axis=1), axis=0).groupby("dia").mean()
        for restaurante in ["Pizza", "Hamburguesas", "Sushi"]:
            if restaurante in fraccion:
                serie = fraccion[restaurante]
                eje.plot(serie.index + 1, serie, color=COLOR[restaurante], marker="o", markersize=5,
                         markeredgecolor=SUPERFICIE, markeredgewidth=1.5, label=restaurante)
                eje.annotate(restaurante, (serie.index[-1] + 1, serie.iloc[-1]), xytext=(6, 0),
                             textcoords="offset points", va="center", fontsize=8, color=TINTA_2)
        eje.set_title(f"Habitantes: {variante.replace('_', ' ')}", fontsize=10)
        eje.set_xlabel("Día")
        eje.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ejes[0].set_ylabel("Fracción de domicilios")
    ejes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Evolución de las preferencias de restaurante (bandit ε-greedy)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _guardar(fig, f"{nombre}_evolucion")


def graficar(nombre: str) -> None:
    ruta = RESULTADOS / f"{nombre}.csv"
    if not ruta.exists():
        print(f"  (no existe {ruta.name}, se omite)")
        return
    df = pd.read_csv(ruta)
    metricas = ["tasa_cancelacion", "tiempo_entrega_promedio", "ganancia_neta_por_repartidor",
                "satisfaccion_promedio"]
    if nombre.startswith("01"):
        orden = ["aleatoria", "cercano", "urgente", "qlearning", "qlearning_sin_rechazo"]
        barras(df, nombre, metricas, orden, "Políticas del repartidor (evaluación, 5 semillas)")
        curva_aprendizaje(df, f"{nombre}_aprendizaje")
    elif nombre.startswith("02"):
        lineas_cantidad(df, nombre)
    elif nombre.startswith("03"):
        barras(df, nombre, ["pedidos_creados", "comidas_salir"] + metricas[:2],
               ["sin_lluvia", "estocastica", "con_lluvia"], "Efecto de la lluvia")
    elif nombre.startswith("04"):
        barras(df, nombre, ["satisfaccion_promedio", "tasa_cancelacion"], ["aprende", "no_aprende"],
               "Habitantes que aprenden vs. que no")
        preferencias(nombre)
    elif nombre.startswith("05"):
        ruta_prueba = RESULTADOS / f"{nombre}_prueba.csv"
        if ruta_prueba.exists():  # la comparación final va con los días de PRUEBA, no de validación
            barras(pd.read_csv(ruta_prueba), nombre, metricas, ["urgente", "qlearning", "dqn"],
                   "Tabla Q vs. red neuronal (DQN): prueba final, 30 días nunca vistos, 5 semillas",
                   fase="prueba")
        curva_aprendizaje(df, f"{nombre}_aprendizaje",
                          titulo="Curva de aprendizaje: tabla Q vs. red neuronal (mismo presupuesto)")
        perdida_red(df, f"{nombre}_perdida")


def main():
    estilo()
    nombres = sys.argv[1:] or ["01_politicas", "02_cantidad_repartidores", "03_lluvia", "04_preferencias",
                               "05_dqn", "06_gymnasium"]
    for nombre in nombres:
        print(f"[{nombre}]")
        if nombre == "06_gymnasium":
            gymnasium(nombre)
        else:
            graficar(nombre)


if __name__ == "__main__":
    main()
