"""Prueba final del experimento 1 con días que nunca se usaron para decidir nada.

Uso (después de correr el experimento 1, que deja las tablas Q en resultados/):
    python -m experimentos.prueba_final
    python -m experimentos.prueba_final config/experimentos/01_politicas.yaml --procesos 4

Por qué existe: los 30 días de "evaluacion" del experimento 1 se miraron muchas
veces mientras se diseñaban la recompensa, el estado y la calibración, así que
en la práctica funcionaron como conjunto de VALIDACIÓN. Para tener una medida
honesta de generalización hace falta un conjunto de PRUEBA que se toque una
sola vez: las mismas tablas Q ya congeladas (sin entrenar de nuevo) evaluadas
en un bloque de semillas propio (ver DESPLAZAMIENTO_FASE en correr.py).
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from experimentos.correr import CARPETA_RESULTADOS, RAIZ, correr_corrida
from sim.config import cargar_config, fusionar, leer_yaml
from sim.politicas import crear_politica_repartidor

YAML_POR_DEFECTO = RAIZ / "config" / "experimentos" / "01_politicas.yaml"
COLUMNAS = ["tasa_cancelacion", "tiempo_entrega_promedio", "recompensa_total", "satisfaccion_promedio"]


def ruta_tabla(nombre_exp: str, variante: str, semilla: int):
    return CARPETA_RESULTADOS / f"{nombre_exp}_{variante}_s{semilla}_q.npz"


def _trabajo(args):
    nombre_exp, variante, config, semilla, dias = args
    politica = crear_politica_repartidor(config["repartidores"])
    if politica.aprende:
        tabla = ruta_tabla(nombre_exp, variante, semilla)
        if not tabla.exists():
            raise FileNotFoundError(f"Falta {tabla}: corre antes el experimento 1")
        politica.cargar(tabla)
    # 0 días de entrenamiento: la tabla queda congelada tal como salió del experimento.
    filas, _, _ = correr_corrida(config, semilla, 0, dias, politica=politica, fase_evaluacion="prueba")
    for fila in filas:
        fila["variante"] = variante
    return filas


def correr_prueba(ruta_yaml=YAML_POR_DEFECTO, procesos: int | None = None) -> pd.DataFrame:
    exp = leer_yaml(ruta_yaml)
    base = cargar_config(cambios=exp.get("comun", {}))
    trabajos = [(exp["nombre"], variante, fusionar(base, cambios or {}), semilla, exp["episodios_prueba"])
                for variante, cambios in exp["variantes"].items() for semilla in exp["semillas"]]
    procesos = procesos or max(1, (os.cpu_count() or 2) - 1)
    print(f"[prueba final] {len(trabajos)} corridas de {exp['episodios_prueba']} días en {procesos} procesos")
    with ProcessPoolExecutor(max_workers=procesos) as pool:
        filas = [f for lote in pool.map(_trabajo, trabajos) for f in lote]
    df = pd.DataFrame(filas)
    df.insert(0, "experimento", exp["nombre"])
    salida = CARPETA_RESULTADOS / f"{exp['nombre']}_prueba.csv"
    df.to_csv(salida, index=False)
    print(f"[prueba final] guardado en {salida.relative_to(RAIZ)}")
    return df


def resumen(df: pd.DataFrame) -> pd.DataFrame:
    """Media por política y, para la cancelación, la desviación entre semillas."""
    tabla = df.groupby("variante")[COLUMNAS].mean()
    tabla["desv_cancelacion_entre_semillas"] = (
        df.groupby(["variante", "semilla_corrida"])["tasa_cancelacion"].mean().groupby("variante").std())
    return tabla.sort_values("tasa_cancelacion")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experimento", nargs="?", default=str(YAML_POR_DEFECTO))
    parser.add_argument("--procesos", type=int, default=None)
    args = parser.parse_args()
    print(resumen(correr_prueba(args.experimento, args.procesos)).round(4).to_string())


if __name__ == "__main__":
    main()
