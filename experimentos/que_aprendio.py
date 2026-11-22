"""¿Qué aprendió la red? Cómo decide cada política, decisión por decisión.

Uso:
    python -m experimentos.que_aprendio

Los números de cancelación dicen QUE la red entrenada es mejor; este script dice
POR QUÉ: registra cada vez que un repartidor tuvo que escoger (si esperó o qué
pedido aceptó) y compara cuatro políticas en los mismos días de prueba:

    red sin entrenar  ·  red entrenada  ·  tabla Q entrenada  ·  regla urgente

Registrar no cambia la simulación: solo se envuelve elegir() para anotar lo que
devuelve, sin tocar el azar.

Salidas: resultados/07_que_aprendio.csv (una fila por decisión) y
resultados/07_que_aprendio.png.
"""
from __future__ import annotations

import argparse

import pandas as pd

from experimentos.correr import CARPETA_RESULTADOS, RAIZ, semilla_episodio
from sim.config import cargar_config, fusionar
from sim.metricas import resumen_episodio
from sim.model import CiudadModel
from sim.politicas import crear_politica_repartidor

# nombre en las tablas -> (política del YAML, ¿cargar el modelo entrenado?)
POLITICAS = {
    "red sin entrenar": ("dqn", False),
    "red entrenada": ("dqn", True),
    "tabla Q entrenada": ("qlearning", True),
    "urgente (regla fija)": ("urgente", False),
}


def registrar_decisiones(politica, destino: list) -> None:
    """Envuelve politica.elegir para anotar cada decisión en `destino` (no usa azar)."""
    elegir_original = politica.elegir

    def elegir(percepcion):
        accion = elegir_original(percepcion)
        pedidos = percepcion["pedidos"]
        fila = {"hora": percepcion["hora"], "lluvia": percepcion["lluvia"], "disponibles": len(pedidos),
                "espera": accion["tipo"] == "esperar"}
        if not fila["espera"]:
            elegido = next(p for p in pedidos if p["pedido"] is accion["pedido"])
            fila.update(edad=elegido["edad"], cuadras_restaurante=elegido["distancia"],
                        cuadras_entrega=elegido["distancia_entrega"], pago=elegido["pago"],
                        listo=elegido["listo"],
                        es_el_mas_viejo=elegido["edad"] == max(p["edad"] for p in pedidos),
                        es_el_mas_cercano=elegido["distancia"] == min(p["distancia"] for p in pedidos))
        destino.append(fila)
        return accion

    politica.elegir = elegir


def crear(config: dict, nombre: str):
    politica_yaml, entrenada = POLITICAS[nombre]
    cfg = fusionar(config, {"repartidores": {"politica": politica_yaml}})
    politica = crear_politica_repartidor(cfg["repartidores"])
    if entrenada:
        politica.cargar(RAIZ / config["que_aprendio"]["modelos"][politica_yaml])
    if hasattr(politica, "explorar"):
        politica.explorar = False  # se mide lo que hace, no lo que prueba al azar
    return cfg, politica


def correr(config: dict | None = None, guardar: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or cargar_config(cambios={"metricas": {"recolectar_cada": 0}})
    q = config["que_aprendio"]
    decisiones, dias = [], []
    for nombre in POLITICAS:
        cfg, politica = crear(config, nombre)
        filas: list = []
        registrar_decisiones(politica, filas)
        for dia in range(q["dias"]):
            inicio = len(filas)
            modelo = CiudadModel(cfg, semilla=semilla_episodio(q["semilla"], "prueba", dia), politica_repartidor=politica)
            modelo.correr()
            for fila in filas[inicio:]:
                fila.update(politica=nombre, dia=dia)
            dias.append(dict(resumen_episodio(modelo), politica=nombre, dia=dia))
        decisiones.extend(filas)
    decisiones, dias = pd.DataFrame(decisiones), pd.DataFrame(dias)
    if guardar:
        CARPETA_RESULTADOS.mkdir(exist_ok=True)
        decisiones.to_csv(CARPETA_RESULTADOS / "07_que_aprendio.csv", index=False)
        graficar(resumen(decisiones, dias))
    return decisiones, dias


def resumen(decisiones: pd.DataFrame, dias: pd.DataFrame) -> pd.DataFrame:
    """Una fila por política: cómo decide (arriba) y qué consigue (abajo)."""
    aceptadas = decisiones[~decisiones["espera"]]
    # "El más viejo" y "el más cercano" solo dicen algo si había de dónde escoger: con un
    # único pedido, ese pedido es las dos cosas a la vez e inflaría los porcentajes.
    con_opciones = aceptadas[aceptadas["disponibles"] >= 2]
    tabla = pd.DataFrame({
        "decisiones por día": decisiones.groupby("politica").size() / decisiones.groupby("politica")["dia"].nunique(),
        "% de veces que espera": decisiones.groupby("politica")["espera"].mean() * 100,
        "% acepta el más viejo": con_opciones.groupby("politica")["es_el_mas_viejo"].mean() * 100,
        "% acepta el más cercano": con_opciones.groupby("politica")["es_el_mas_cercano"].mean() * 100,
        "edad del pedido aceptado (min)": aceptadas.groupby("politica")["edad"].mean(),
        "cuadras al restaurante": aceptadas.groupby("politica")["cuadras_restaurante"].mean(),
        "cancelación": dias.groupby("politica")["tasa_cancelacion"].mean() * 100,
        "recompensa por día": dias.groupby("politica")["recompensa_total"].mean(),
    })
    return tabla.loc[[p for p in POLITICAS if p in tabla.index]]  # orden fijo, aunque falte alguna


def graficar(tabla: pd.DataFrame, guardar: bool = True) -> None:
    """Paneles pequeños, una barra por política, valor escrito encima (identidad no solo por color)."""
    import matplotlib.pyplot as plt

    from experimentos.graficas import COLOR, SERIES, SUPERFICIE, TINTA_2, _guardar, estilo

    estilo()
    colores = {"red sin entrenar": SERIES[0], "red entrenada": COLOR["dqn"],
               "tabla Q entrenada": COLOR["qlearning"], "urgente (regla fija)": COLOR["urgente"]}
    paneles = [("% de veces que espera", "Espera habiendo pedidos (%)", "{:.0f}%"),
               ("% acepta el más viejo", "Escoge el más viejo* (%)", "{:.0f}%"),
               ("% acepta el más cercano", "Escoge el más cercano* (%)", "{:.0f}%"),
               ("edad del pedido aceptado (min)", "Edad del pedido que acepta (min)", "{:.1f}"),
               ("cancelación", "Cancelación de la ciudad (%)", "{:.1f}%")]
    fig, ejes = plt.subplots(1, len(paneles), figsize=(15, 3.9))
    etiquetas = [n.replace(" (regla fija)", "").replace(" ", "\n", 1) for n in tabla.index]
    for eje, (columna, titulo, formato) in zip(ejes, paneles):
        valores = tabla[columna]
        eje.bar(range(len(valores)), valores, width=0.7, color=[colores[n] for n in tabla.index],
                edgecolor=SUPERFICIE, linewidth=2)
        for x, v in enumerate(valores):
            eje.annotate(formato.format(v), (x, v), xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=8, color=TINTA_2)
        eje.set_xticks(range(len(valores)), etiquetas, fontsize=8)
        eje.set_title(titulo, fontsize=10)
    fig.suptitle("Qué aprendió la red: cómo decide cada política (30 días de prueba)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, -0.02, "* solo decisiones con 2 o más pedidos disponibles (con uno solo, es el más viejo y el más "
             "cercano a la vez).", fontsize=8, color=TINTA_2)
    fig.tight_layout()
    if guardar:
        _guardar(fig, "07_que_aprendio")
    else:
        plt.show()


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    decisiones, dias = correr()
    print(resumen(decisiones, dias).round(1).T.to_string())


if __name__ == "__main__":
    main()
