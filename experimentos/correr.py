"""Corre episodios sin gráficos y guarda los resultados en CSV.

Uso:
    python -m experimentos.correr config/experimentos/01_politicas.yaml
    python -m experimentos.correr config/experimentos/01_politicas.yaml --procesos 4

Cada corrida (variante, semilla) tiene dos fases:
  1. entrenamiento: la política aprende (solo si es una política que aprende).
  2. evaluacion: episodios con semillas comunes a TODAS las variantes, sin
     exploración ni aprendizaje. Por qué semillas comunes: así la lluvia y
     los pedidos son los mismos para todas las políticas (comparación pareada)
     y la diferencia que se ve es de la política, no de la suerte.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from sim.config import cargar_config, fusionar, leer_yaml  # noqa: E402
from sim.metricas import resumen_episodio  # noqa: E402
from sim.model import CiudadModel  # noqa: E402
from sim.politicas import crear_politica_repartidor  # noqa: E402

CARPETA_RESULTADOS = RAIZ / "resultados"
# Tres bloques de semillas que nunca se cruzan (cada bloque admite hasta 30.000 días):
#   entrenamiento -> la política aprende.
#   evaluacion    -> los resultados que miramos MIENTRAS diseñábamos (cumple el papel
#                    del conjunto de validación: con ellos se decidieron recompensas,
#                    estados y calibración).
#   prueba        -> se corre UNA sola vez al final, con las tablas ya congeladas.
#                    Por qué: si se elige el diseño mirando un conjunto, ese conjunto
#                    deja de medir la generalización real (Géron, cap. 2).
DESPLAZAMIENTO_FASE = {"entrenamiento": 0, "evaluacion": 50_000, "prueba": 80_000}


def semilla_episodio(semilla: int, fase: str, episodio: int) -> int:
    return semilla * 100_000 + DESPLAZAMIENTO_FASE[fase] + episodio


def correr_corrida(config: dict, semilla: int, episodios_entrenamiento: int,
                   episodios_evaluacion: int, politica=None, guardar_comidas: bool = False,
                   al_terminar_episodio=None, fase_evaluacion: str = "evaluacion"):
    """Entrena y evalúa una política. Devuelve (filas, comidas, política).

    `fase_evaluacion="prueba"` usa el bloque de semillas reservado para la
    prueba final, que no se usó para tomar ninguna decisión de diseño.
    """
    politica = politica or crear_politica_repartidor(config["repartidores"])
    if not politica.aprende:
        episodios_entrenamiento = 0  # no tiene nada que aprender: ahorra tiempo
    filas, comidas = [], []
    for fase, cantidad in (("entrenamiento", episodios_entrenamiento),
                           (fase_evaluacion, episodios_evaluacion)):
        if fase == fase_evaluacion and hasattr(politica, "explorar"):
            politica.explorar = False
        for episodio in range(cantidad):
            modelo = CiudadModel(config, semilla=semilla_episodio(semilla, fase, episodio),
                                 politica_repartidor=politica)
            modelo.correr()
            fila = resumen_episodio(modelo)
            fila.update(fase=fase, episodio=episodio, semilla_corrida=semilla,
                        epsilon=getattr(politica, "epsilon", float("nan")))
            filas.append(fila)
            if guardar_comidas:
                comidas.extend(dict(c, fase=fase, episodio=episodio, semilla_corrida=semilla)
                               for c in modelo.comidas)
            if fase == "entrenamiento":
                politica.fin_episodio()
            if al_terminar_episodio:
                al_terminar_episodio(fase, episodio, fila)
    return filas, comidas, politica


def _trabajo(args):
    """Una corrida en un proceso aparte (debe ser función de módulo para Windows)."""
    nombre_exp, variante, config, semilla, n_ent, n_eval, guardar_comidas = args
    inicio = time.time()
    filas, comidas, politica = correr_corrida(config, semilla, n_ent, n_eval,
                                              guardar_comidas=guardar_comidas)
    for fila in filas:
        fila["variante"] = variante
    for comida in comidas:
        comida["variante"] = variante
    if hasattr(politica, "guardar"):
        politica.guardar(CARPETA_RESULTADOS / f"{nombre_exp}_{variante}_s{semilla}_q.npz")
    return variante, semilla, filas, comidas, time.time() - inicio


def correr_experimento(ruta_yaml: str | Path, procesos: int | None = None) -> pd.DataFrame:
    exp = leer_yaml(ruta_yaml)
    nombre = exp["nombre"]
    base = cargar_config(cambios=exp.get("comun", {}))
    trabajos = []
    for variante, cambios in exp["variantes"].items():
        config = fusionar(base, cambios or {})
        for semilla in exp["semillas"]:
            trabajos.append((nombre, variante, config, semilla, exp.get("episodios_entrenamiento", 0),
                             exp.get("episodios_evaluacion", 1), exp.get("guardar_comidas", False)))

    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    procesos = procesos or max(1, (os.cpu_count() or 2) - 1)
    print(f"[{nombre}] {len(trabajos)} corridas en {procesos} procesos")
    todas_filas, todas_comidas = [], []
    with ProcessPoolExecutor(max_workers=procesos) as pool:
        futuros = [pool.submit(_trabajo, t) for t in trabajos]
        for futuro in as_completed(futuros):
            variante, semilla, filas, comidas, segundos = futuro.result()
            todas_filas.extend(filas)
            todas_comidas.extend(comidas)
            print(f"  {variante:<20} semilla {semilla:<4} {len(filas):>4} episodios  {segundos:6.1f} s")

    df = pd.DataFrame(todas_filas).sort_values(["variante", "semilla_corrida", "fase", "episodio"])
    df.insert(0, "experimento", nombre)
    df.to_csv(CARPETA_RESULTADOS / f"{nombre}.csv", index=False)
    if todas_comidas:
        pd.DataFrame(todas_comidas).to_csv(CARPETA_RESULTADOS / f"{nombre}_comidas.csv", index=False)
    print(f"[{nombre}] guardado en resultados/{nombre}.csv")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experimentos", nargs="+", help="uno o más YAML de config/experimentos/")
    parser.add_argument("--procesos", type=int, default=None)
    args = parser.parse_args()
    for ruta in args.experimentos:
        df = correr_experimento(ruta, args.procesos)
        evaluacion = df[df["fase"] == "evaluacion"]
        columnas = ["entregados", "tasa_cancelacion", "tiempo_entrega_promedio",
                    "ganancia_neta_por_repartidor", "satisfaccion_promedio"]
        print(evaluacion.groupby("variante")[columnas].mean().round(3).to_string(), "\n")


if __name__ == "__main__":
    main()
