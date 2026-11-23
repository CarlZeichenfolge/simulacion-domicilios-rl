"""Q-learning tabular: aprende cuánto vale aceptar cada TIPO de pedido según el contexto."""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import numpy as np

from sim.politicas.base import Politica, accion_aceptar, accion_esperar


class PoliticaRepartidorQLearning(Politica):
    """Estado (contexto) = (franja horaria, lluvia, carga de pedidos disponibles).
    Acción = el tipo de pedido que acepta, descrito por (distancia al
    restaurante, distancia de entrega, edad del pedido, ¿ya está listo?), o esperar.

    Por qué la acción es un "tipo de pedido" y no "el más cercano / el mejor
    pagado": con reglas fijas como acciones el agente no puede distinguir un
    pedido salvable de uno que igual se va a cancelar, y en las pruebas
    terminaba esperando sin hacer nada. Describir el pedido con pocas
    características discretas le permite aprender, por ejemplo, que un pedido
    viejo pero cercano y ya listo vale la pena, y uno viejo y lejano no.
    Sigue siendo tabular: Q[contexto, tipo] es una celda que se puede leer.

    Por qué SMDP (gamma ** duracion): aceptar dura 20-40 minutos y esperar 3.
    Descontar por minutos reales hace comparables acciones de distinta duración.

    Por qué una sola tabla para todos: los repartidores son idénticos; compartir
    la tabla junta la experiencia de todos y aprende más rápido.
    """

    aprende = True

    def __init__(self, cfg: dict):
        super().__init__()
        self.alpha = cfg["alpha"]
        self.gamma = cfg["gamma"]
        self.epsilon = cfg["epsilon_inicial"]
        self.epsilon_min = cfg["epsilon_min"]
        self.decaimiento = cfg["decaimiento_epsilon"]
        self.cortes_hora = list(cfg["cortes_hora"])
        self.cortes_carga = list(cfg["cortes_carga"])
        self.cortes_distancia = list(cfg["cortes_distancia"])
        self.cortes_entrega = list(cfg["cortes_entrega"])
        self.cortes_edad = list(cfg["cortes_edad"])
        self.forma_tipo = (len(self.cortes_distancia) + 1, len(self.cortes_entrega) + 1,
                           len(self.cortes_edad) + 1, 2)
        self.n_tipos = int(np.prod(self.forma_tipo))
        self.ESPERAR = self.n_tipos  # la última columna es "esperar"
        forma = (len(self.cortes_hora) + 1, 2, len(self.cortes_carga) + 1, self.n_tipos + 1)
        self.q = np.zeros(forma)
        self.visitas = np.zeros(forma, dtype=np.int64)
        self.explorar = True  # False en evaluación: solo explota lo aprendido, no actualiza

    # --------------------------------------------------------- discretización
    def estado(self, percepcion: dict) -> tuple[int, int, int]:
        return (bisect_right(self.cortes_hora, percepcion["hora"]),
                int(percepcion["lluvia"]),
                bisect_right(self.cortes_carga, len(percepcion["pedidos"])))

    def tipo_pedido(self, info: dict) -> int:
        celdas = (bisect_right(self.cortes_distancia, info["distancia"]),
                  bisect_right(self.cortes_entrega, info["distancia_entrega"]),
                  bisect_right(self.cortes_edad, info["edad"]),
                  int(info["listo"]))
        return int(np.ravel_multi_index(celdas, self.forma_tipo))

    def describir_tipo(self, indice: int) -> str:
        """Texto legible de una columna de la tabla, para el informe."""
        if indice == self.ESPERAR:
            return "esperar"
        d, e, a, listo = np.unravel_index(indice, self.forma_tipo)
        return f"dist_rest={d} dist_entrega={e} edad={a} listo={bool(listo)}"

    # ----------------------------------------------------------------- decidir
    def elegir(self, percepcion: dict) -> dict:
        pedidos = percepcion["pedidos"]
        opciones = [(self.tipo_pedido(p), p) for p in pedidos] + [(self.ESPERAR, None)]
        if self.explorar and self.rng.random() < self.epsilon:
            indice, info = self.rng.choice(opciones)
        else:
            valores = self.q[self.estado(percepcion)]
            mejor = max(valores[i] for i, _ in opciones)
            # Desempate al azar con el rng del modelo para no sesgar hacia un tipo.
            indice, info = self.rng.choice([(i, p) for i, p in opciones if valores[i] == mejor])
        if info is None:
            accion = accion_esperar()
            accion["indice"] = self.ESPERAR
            return accion
        return accion_aceptar(info, indice)

    def valorar(self, percepcion: dict) -> list[float]:
        valores = self.q[self.estado(percepcion)]
        return [float(valores[self.tipo_pedido(p)]) for p in percepcion["pedidos"]] + [float(valores[self.ESPERAR])]

    # ---------------------------------------------------------------- aprender
    def aprender(self, percepcion, accion, recompensa, nueva_percepcion, duracion=1):
        """Q(s,a) <- Q(s,a) + alpha [r + gamma^k max_a' Q(s',a') - Q(s,a)].

        El max del siguiente estado es solo sobre los pedidos que de verdad
        hay en s' (más esperar): no se puede aceptar un tipo de pedido que no existe.
        """
        if not self.explorar:
            return
        s = self.estado(percepcion)
        a = accion["indice"]
        s2 = self.estado(nueva_percepcion)
        legales = {self.tipo_pedido(p) for p in nueva_percepcion["pedidos"]} | {self.ESPERAR}
        futuro = max(self.q[s2][b] for b in legales)
        objetivo = recompensa + (self.gamma ** duracion) * futuro
        self.q[s + (a,)] += self.alpha * (objetivo - self.q[s + (a,)])
        self.visitas[s + (a,)] += 1

    def fin_episodio(self) -> None:
        """Decaimiento exponencial: explora mucho al inicio, explota al final."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decaimiento)

    # -------------------------------------------------------------- persistencia
    def guardar(self, ruta: str | Path) -> None:
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        np.savez(ruta, q=self.q, visitas=self.visitas, epsilon=self.epsilon)

    def cargar(self, ruta: str | Path) -> None:
        datos = np.load(ruta)
        if datos["q"].shape != self.q.shape:
            raise ValueError("La tabla Q guardada no coincide con los cortes del YAML actual")
        self.q = datos["q"]
        self.visitas = datos["visitas"]
        self.epsilon = float(datos["epsilon"])
