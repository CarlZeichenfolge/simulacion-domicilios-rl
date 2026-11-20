"""La ciudad como grafo de calles con edificios en las intersecciones."""
from __future__ import annotations

import math
import random

import networkx as nx

TIPOS_EDIFICIO = ("restaurante", "universidad", "oficina", "casa")


class Ciudad:
    """Cuadrícula de N x N manzanas convertida en grafo de NetworkX.

    Cada nodo es una intersección (x, y). Cada arista es una calle con
    `longitud` (m), `tiempo_base` (min) y `factor_trancon` (1.0 = normal).

    Por qué un grafo y no una matriz de celdas: así A* trabaja sobre calles
    reales y el trancón es solo un atributo de la arista que cambia el peso.
    """

    def __init__(self, cfg: dict, rng: random.Random):
        self.tamano = cfg["tamano"]
        self.longitud_calle = cfg["longitud_calle_m"]
        self.velocidad_base = cfg["velocidad_base_m_min"]
        self.minutos_por_cuadra = self.longitud_calle / self.velocidad_base

        self.grafo = nx.grid_2d_graph(self.tamano + 1, self.tamano + 1)
        for nodo in self.grafo.nodes:
            self.grafo.nodes[nodo]["pos"] = (nodo[0] * self.longitud_calle, nodo[1] * self.longitud_calle)
        for u, v in self.grafo.edges:
            datos = self.grafo.edges[u, v]
            datos["longitud"] = self.longitud_calle
            datos["tiempo_base"] = self.minutos_por_cuadra
            datos["factor_trancon"] = 1.0

        self.edificios: dict[tuple, str] = {}
        self.restaurantes: dict[str, tuple] = {}
        self._ubicar_edificios(cfg, rng)

    # ------------------------------------------------------------------ edificios
    def _ubicar_edificios(self, cfg: dict, rng: random.Random) -> None:
        """Restaurantes primero (pueden venir fijos del YAML), luego el resto al azar.

        Por qué con el rng recibido: la misma semilla produce el mismo mapa.
        """
        libres = sorted(self.grafo.nodes)
        for rest in cfg["restaurantes"]:
            if rest.get("nodo") is not None:
                nodo = tuple(rest["nodo"])
                if nodo not in self.grafo:
                    raise ValueError(f"Nodo {nodo} de {rest['nombre']} fuera de la ciudad")
                libres.remove(nodo)
                self._poner(nodo, "restaurante")
                self.restaurantes[rest["nombre"]] = nodo
        rng.shuffle(libres)
        for rest in cfg["restaurantes"]:
            if rest["nombre"] not in self.restaurantes:
                nodo = libres.pop()
                self._poner(nodo, "restaurante")
                self.restaurantes[rest["nombre"]] = nodo

        cantidades = cfg["edificios"]
        pedidos = [("universidad", cantidades["universidades"]),
                   ("oficina", cantidades["oficinas"]),
                   ("casa", cantidades["casas"])]
        if sum(n for _, n in pedidos) > len(libres):
            raise ValueError("No caben tantos edificios en la ciudad")
        for tipo, cantidad in pedidos:
            for _ in range(cantidad):
                self._poner(libres.pop(), tipo)

    def _poner(self, nodo: tuple, tipo: str) -> None:
        self.edificios[nodo] = tipo
        self.grafo.nodes[nodo]["edificio"] = tipo

    def nodos_de_tipo(self, tipo: str) -> list[tuple]:
        return sorted(n for n, t in self.edificios.items() if t == tipo)

    # ------------------------------------------------------------------- rutas
    def tiempo_arista(self, u: tuple, v: tuple, factor_velocidad: float = 1.0) -> float:
        """Minutos para recorrer la calle u-v ahora mismo.

        tiempo_base * factor_trancon / factor_velocidad: el trancón alarga
        la calle y la lluvia baja la velocidad (factor < 1 alarga el tiempo).
        """
        datos = self.grafo.edges[u, v]
        return datos["tiempo_base"] * datos["factor_trancon"] / factor_velocidad

    def ruta(self, origen: tuple, destino: tuple, factor_velocidad: float = 1.0) -> list[tuple]:
        """Camino más rápido con A* (incluye origen y destino).

        Heurística: distancia euclidiana / velocidad base. Es admisible porque
        trancón (>= 1) y lluvia (<= 1) solo pueden alargar el tiempo real, así
        que A* sigue garantizando la ruta óptima y explora menos que Dijkstra.
        """
        if origen == destino:
            return [origen]

        def peso(u, v, datos):
            return datos["tiempo_base"] * datos["factor_trancon"] / factor_velocidad

        return nx.astar_path(self.grafo, origen, destino, heuristic=self.heuristica, weight=peso)

    def heuristica(self, a: tuple, b: tuple) -> float:
        (xa, ya), (xb, yb) = self.grafo.nodes[a]["pos"], self.grafo.nodes[b]["pos"]
        return math.hypot(xa - xb, ya - yb) / self.velocidad_base

    def tiempo_ruta(self, ruta: list[tuple], factor_velocidad: float = 1.0) -> float:
        return sum(self.tiempo_arista(u, v, factor_velocidad) for u, v in zip(ruta, ruta[1:]))

    @staticmethod
    def distancia_cuadras(a: tuple, b: tuple) -> int:
        """Distancia Manhattan en cuadras.

        Por qué Manhattan y no A*: en una cuadrícula es exactamente la ruta
        más corta sin trancón, y cuesta O(1). Los repartidores la usan para
        comparar muchos pedidos por tick sin correr A* para cada uno.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def aristas(self) -> list[tuple]:
        return list(self.grafo.edges)
