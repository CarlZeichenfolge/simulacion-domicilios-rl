"""La red DQN entrenada SOLO con la interfaz de Gymnasium (reset / step).

Uso:
    python -m experimentos.gym_dqn

Qué demuestra: el mismo bucle que se usa con CartPole o MuJoCo,
    obs, info = env.reset(); accion = agente(obs); obs, r, ... = env.step(accion)
funciona con nuestra ciudad. El agente no toca Mesa: solo ve vectores de
números. Luego se compara en los días de PRUEBA contra un agente aleatorio
y contra la regla urgente, todos jugando como repartidor 0.

Salidas: resultados/gym_entrenamiento.csv, resultados/gym_prueba.csv y
resultados/gym_dqn.pt (la red).
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from experimentos.correr import CARPETA_RESULTADOS, semilla_episodio
from sim.config import cargar_config
from sim.entorno_gym import EntornoDomicilios, separar_observacion
from sim.politicas.repartidor_dqn import PoliticaRepartidorDQN, filas_candidatas

EDAD = 2  # posición de "edad del pedido" dentro de los 7 números de cada casilla


class AgenteAleatorio:
    """Escoge al azar entre las acciones legales (la línea base de "no sabe nada")."""
    nombre = "aleatorio"

    def __init__(self, k: int, rng: random.Random):
        self.k, self.rng = k, rng

    def actuar(self, obs, mascara):
        return self.rng.choice(np.flatnonzero(mascara).tolist())

    def observar(self, *args):
        pass


class AgenteUrgente:
    """La mejor regla fija del proyecto, leída desde el vector: la casilla con más edad."""
    nombre = "urgente"

    def __init__(self, k: int):
        self.k = k

    def actuar(self, obs, mascara):
        _, pedidos = separar_observacion(obs, self.k)
        if not pedidos:
            return self.k
        return int(np.argmax([p[EDAD] for p in pedidos]))  # empate: la casilla más baja = el más antiguo

    def observar(self, *args):
        pass


class AgenteDQN:
    """Traduce el vector de Gymnasium a las filas (situación, pedido) que entiende la red."""
    nombre = "dqn"

    def __init__(self, k: int, politica: PoliticaRepartidorDQN, rng: random.Random):
        self.k, self.pol, self.rng = k, politica, rng
        self._x = None

    def actuar(self, obs, mascara):
        contexto, pedidos = separar_observacion(obs, self.k)
        x = filas_candidatas(contexto, pedidos)
        if self.pol.explorar and self.rng.random() < self.pol.epsilon:
            i = self.rng.randrange(len(x))
        else:
            i = int(np.argmax(self.pol.valores_q(x)))
        self._x = x[i]
        return self.k if i == len(pedidos) else i

    def observar(self, obs, recompensa, obs_siguiente, minutos, truncado):
        # Al cerrar el día la última acción queda abierta: no hay s' con qué
        # calcular el objetivo, así que esa transición no se guarda.
        if not truncado:
            siguientes = filas_candidatas(*separar_observacion(obs_siguiente, self.k))
            self.pol.recordar(self._x, recompensa, max(1, minutos), siguientes)


def jugar_dia(env: EntornoDomicilios, agente, semilla_dia: int) -> dict:
    obs, info = env.reset(options={"semilla_dia": semilla_dia})
    total, decisiones = 0.0, 0
    while not env.dia_terminado:  # casi nunca, pero un día puede acabar sin que le toque decidir
        accion = agente.actuar(obs, info["mascara_acciones"])
        obs_sig, r, _, truncado, info = env.step(accion)
        agente.observar(obs, r, obs_sig, info["minutos"], truncado)
        obs, total, decisiones = obs_sig, total + r, decisiones + 1
    m, rep = env.modelo, env.externo
    return {"recompensa_repartidor0": total, "decisiones": decisiones, "entregas_repartidor0": rep.entregas,
            "cancelaciones_repartidor0": rep.cancelaciones, "rechazos_repartidor0": rep.rechazos,
            "tasa_cancelacion_ciudad": m.conteo["cancelados"] / max(1, m.conteo["creados"])}


def correr(config: dict | None = None, guardar: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or cargar_config(cambios={"metricas": {"recolectar_cada": 0}})
    g = config["gym"]
    env = EntornoDomicilios(config)
    check_env(env, skip_render_check=True)  # el validador oficial: si la interfaz está mal, falla aquí
    k = env.k

    politica = PoliticaRepartidorDQN(config["repartidores"]["dqn"])
    dqn = AgenteDQN(k, politica, random.Random(g["semilla"]))
    filas_ent, inicio = [], time.time()
    for dia in range(g["dias_entrenamiento"]):
        fila = jugar_dia(env, dqn, semilla_episodio(g["semilla"], "entrenamiento", dia))
        politica.fin_episodio()
        filas_ent.append(dict(fila, dia=dia, epsilon=politica.epsilon, perdida_red=politica.historial_perdida[-1]))
        if dia % 50 == 0 or dia == g["dias_entrenamiento"] - 1:
            print(f"  día {dia:3d}  recompensa {fila['recompensa_repartidor0']:7.1f}  "
                  f"ε={politica.epsilon:.2f}  {time.time() - inicio:5.0f} s", flush=True)
    politica.explorar = False

    filas_prueba = []
    for agente in (AgenteAleatorio(k, random.Random(g["semilla"])), AgenteUrgente(k), dqn):
        for dia in range(g["dias_prueba"]):
            fila = jugar_dia(env, agente, semilla_episodio(g["semilla"], "prueba", dia))
            filas_prueba.append(dict(fila, agente=agente.nombre, dia=dia))

    entrenamiento, prueba = pd.DataFrame(filas_ent), pd.DataFrame(filas_prueba)
    if guardar:
        CARPETA_RESULTADOS.mkdir(exist_ok=True)
        entrenamiento.to_csv(CARPETA_RESULTADOS / "gym_entrenamiento.csv", index=False)
        prueba.to_csv(CARPETA_RESULTADOS / "gym_prueba.csv", index=False)
        politica.guardar(CARPETA_RESULTADOS / "gym_dqn.pt")
    return entrenamiento, prueba


def resumen(prueba: pd.DataFrame) -> pd.DataFrame:
    columnas = ["recompensa_repartidor0", "entregas_repartidor0", "cancelaciones_repartidor0",
                "rechazos_repartidor0", "tasa_cancelacion_ciudad"]
    return prueba.groupby("agente")[columnas].mean().sort_values("recompensa_repartidor0", ascending=False)


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    _, prueba = correr()
    print(resumen(prueba).round(3).to_string())


if __name__ == "__main__":
    main()
