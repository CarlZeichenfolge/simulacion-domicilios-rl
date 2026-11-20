"""Punto de entrada.

    python main.py --vista ninguna --episodios 300 --semilla 42   # entrena sin ventana
    python main.py --vista 2d                                      # demo con Pygame
    python main.py --vista 3d                                      # demo con Ursina

Con --vista ninguna se entrena la política del YAML y se guardan
resultados/entrenamiento.csv y resultados/tabla_q.npz. Las vistas cargan esa
tabla (si existe) para mostrar al repartidor ya entrenado.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experimentos.correr import CARPETA_RESULTADOS, correr_corrida
from sim.config import cargar_config
from sim.model import CiudadModel
from sim.politicas import crear_politica_repartidor

TABLA_Q = CARPETA_RESULTADOS / "tabla_q.npz"


def entrenar(config: dict, episodios: int, semilla: int, ruta_q: Path) -> None:
    def progreso(fase, episodio, fila):
        if episodio % 25 == 0 or episodio == episodios - 1:
            print(f"  {fase:<13} ep {episodio:>4}  entregados {fila['entregados']:>3}  "
                  f"cancelación {fila['tasa_cancelacion']:.2f}  recompensa {fila['recompensa_total']:7.1f}  "
                  f"epsilon {fila['epsilon']:.3f}")

    print(f"Entrenando política '{config['repartidores']['politica']}' por {episodios} episodios")
    filas, _, politica = correr_corrida(config, semilla, episodios, episodios_evaluacion=10,
                                        al_terminar_episodio=progreso)
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    pd.DataFrame(filas).to_csv(CARPETA_RESULTADOS / "entrenamiento.csv", index=False)
    if hasattr(politica, "guardar"):
        politica.guardar(ruta_q)
        print(f"Tabla Q guardada en {ruta_q}")


def crear_modelo_demo(config: dict, semilla: int, ruta_q: Path) -> CiudadModel:
    politica = crear_politica_repartidor(config["repartidores"])
    if hasattr(politica, "cargar"):
        if ruta_q.exists():
            politica.cargar(ruta_q)
            politica.explorar = False
            print(f"Usando tabla Q entrenada: {ruta_q}")
        else:
            print("Aviso: no hay tabla Q entrenada; el repartidor arranca sin saber nada. "
                  "Corre primero: python main.py --vista ninguna")
    return CiudadModel(config, semilla=semilla, politica_repartidor=politica)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--vista", choices=["ninguna", "2d", "3d"], default="ninguna")
    parser.add_argument("--episodios", type=int, default=300)
    parser.add_argument("--semilla", type=int, default=None)
    parser.add_argument("--politica", default=None, help="sobrescribe repartidores.politica del YAML")
    parser.add_argument("--tabla-q", default=str(TABLA_Q))
    args = parser.parse_args()

    cambios = {"repartidores": {"politica": args.politica}} if args.politica else {}
    config = cargar_config(args.config, cambios)
    semilla = config["semilla"] if args.semilla is None else args.semilla
    ruta_q = Path(args.tabla_q)

    if args.vista == "ninguna":
        config["metricas"]["recolectar_cada"] = 0
        entrenar(config, args.episodios, semilla, ruta_q)
        return

    modelo = crear_modelo_demo(config, semilla, ruta_q)
    if args.vista == "2d":
        from vistas.vista_2d import Vista2D
        Vista2D(modelo).correr()
    else:
        from vistas.vista_3d import Vista3D
        Vista3D(modelo).correr()


if __name__ == "__main__":
    main()
