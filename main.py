"""Punto de entrada.

    python main.py --vista ninguna --episodios 300 --semilla 42   # entrena sin ventana
    python main.py --vista 2d                                      # demo con Pygame
    python main.py --vista 3d                                      # demo con Ursina
    python main.py --vista duelo                                   # antes vs después de la red, en 2D
    python main.py --vista duelo3d                                 # antes vs después de la red, en 3D
    python main.py --vista 3d --politica dqn --hora 7              # la red en 3D, arrancando a las 7 a. m.
    python main.py --vista ninguna --politica dqn                  # entrena la red neuronal

Las demos arrancan a las 11:00 (vistas.hora_inicio) a 2 minutos simulados por
segundo real; con ↑/↓ se acelera o se frena.

Con --vista ninguna se entrena la política del YAML y se guardan
resultados/entrenamiento.csv y el modelo (resultados/tabla_q.npz para la tabla,
resultados/red_dqn.pt para la red). Las vistas lo cargan (si existe) para
mostrar al repartidor ya entrenado.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experimentos.correr import CARPETA_RESULTADOS, correr_corrida
from sim.config import cargar_config, fusionar
from sim.model import CiudadModel
from sim.politicas import crear_politica_repartidor

TABLA_Q = CARPETA_RESULTADOS / "tabla_q.npz"
RED_DQN = CARPETA_RESULTADOS / "red_dqn.pt"
# Copias versionadas en git (resultados/ no se sube): las usan las demos si no hay un modelo entrenado local.
CARPETA_MODELOS = Path(__file__).resolve().parent / "modelos"
VERSIONADO = {".npz": CARPETA_MODELOS / "tabla_q_300dias.npz", ".pt": CARPETA_MODELOS / "red_dqn_300dias.pt"}


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
        print(f"Modelo guardado en {ruta_q}")


def crear_modelo_demo(config: dict, semilla: int, ruta_q: Path) -> CiudadModel:
    politica = crear_politica_repartidor(config["repartidores"])
    if hasattr(politica, "cargar"):
        if not ruta_q.exists():
            respaldo = VERSIONADO.get(getattr(politica, "extension", ".npz"))
            if respaldo is not None and respaldo.exists():
                ruta_q = respaldo  # entrenado 300 días, guardado en el repositorio
        if ruta_q.exists():
            politica.cargar(ruta_q)
            politica.explorar = False
            print(f"Usando modelo entrenado: {ruta_q}")
        else:
            print(f"Aviso: no existe {ruta_q}; el repartidor arranca sin saber nada. "
                  f"Corre primero: python main.py --vista ninguna --politica {config['repartidores']['politica']}")
    return CiudadModel(config, semilla=semilla, politica_repartidor=politica)


def crear_duelo(config: dict, semilla: int, entrenado: CiudadModel):
    """(modelo antes, modelo después, título antes, título después) para la vista duelo.

    El "antes" es la MISMA clase de política recién creada y sin explorar: así
    lo único que cambia entre las dos ciudades es lo aprendido. Si la política
    no aprende (una regla fija), no hay "antes": se compara con la aleatoria.
    """
    nombre = config["repartidores"]["politica"]
    politica = crear_politica_repartidor(config["repartidores"])
    if politica.aprende:
        politica.explorar = False
        antes = CiudadModel(config, semilla=semilla, politica_repartidor=politica)
        titulo = f"ANTES: {nombre} sin entrenar"
    else:
        antes = CiudadModel(fusionar(config, {"repartidores": {"politica": "aleatoria"}}), semilla=semilla)
        titulo = "ANTES: aleatoria"
    return antes, entrenado, titulo, f"DESPUÉS: {nombre} entrenado"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--vista", choices=["ninguna", "2d", "3d", "duelo", "duelo3d"], default="ninguna")
    parser.add_argument("--episodios", type=int, default=300)
    parser.add_argument("--semilla", type=int, default=None)
    parser.add_argument("--politica", default=None,
                        help="sobrescribe repartidores.politica del YAML (en los duelos, por defecto dqn)")
    parser.add_argument("--tabla-q", default=None,
                        help="modelo a guardar/cargar (por defecto tabla_q.npz o red_dqn.pt según la política)")
    parser.add_argument("--hora", type=float, default=None,
                        help="hora a la que arranca la demo (por defecto vistas.hora_inicio del YAML)")
    args = parser.parse_args()

    politica = args.politica or ("dqn" if args.vista in ("duelo", "duelo3d") else None)
    cambios = {"repartidores": {"politica": politica}} if politica else {}
    config = cargar_config(args.config, cambios)
    semilla = config["semilla"] if args.semilla is None else args.semilla
    ruta_q = Path(args.tabla_q) if args.tabla_q else (
        RED_DQN if config["repartidores"]["politica"] == "dqn" else TABLA_Q)

    if args.vista == "ninguna":
        config["metricas"]["recolectar_cada"] = 0
        entrenar(config, args.episodios, semilla, ruta_q)
        return

    from vistas.velocidad import adelantar
    hora = config["vistas"]["hora_inicio"] if args.hora is None else args.hora
    if args.vista in ("duelo", "duelo3d"):
        # Mismo clima en las dos ciudades: si no, una puede tener lluvia y la otra no, y la
        # diferencia ya no sería solo de lo aprendido.
        config = fusionar(config, {"eventos": {"azar_propio": True}})
    modelo = crear_modelo_demo(config, semilla, ruta_q)
    if args.vista in ("duelo", "duelo3d"):
        antes, despues, titulo_antes, titulo_despues = crear_duelo(config, semilla, modelo)
        adelantar(antes, hora)
        adelantar(despues, hora)
        if args.vista == "duelo":
            from vistas.vista_duelo import VistaDuelo
            VistaDuelo(antes, despues, titulo_antes, titulo_despues).correr()
        else:
            from vistas.vista_3d import VistaDuelo3D
            VistaDuelo3D(antes, despues, titulo_antes, titulo_despues).correr()
        return
    adelantar(modelo, hora)
    if args.vista == "2d":
        from vistas.vista_2d import Vista2D
        Vista2D(modelo).correr()
    else:
        from vistas.vista_3d import Vista3D
        Vista3D(modelo).correr()


if __name__ == "__main__":
    main()
