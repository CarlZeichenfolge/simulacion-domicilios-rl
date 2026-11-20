"""Pedido y su ciclo de vida.

CREADO -> EN_PREPARACION -> LISTO -> EN_CAMINO -> ENTREGADO
Cualquier estado no terminal -> CANCELADO (si se pasa la espera máxima).
La asignación del repartidor es independiente del estado de cocina: un
repartidor puede aceptar un pedido que todavía se está preparando.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EstadoPedido(Enum):
    CREADO = "creado"
    EN_PREPARACION = "en_preparacion"
    LISTO = "listo"
    EN_CAMINO = "en_camino"
    ENTREGADO = "entregado"
    CANCELADO = "cancelado"


TERMINALES = (EstadoPedido.ENTREGADO, EstadoPedido.CANCELADO)

# Transiciones válidas. Por qué explícitas: si un agente intenta un salto
# ilegal (entregar algo que no se recogió) queremos un error, no un dato raro.
TRANSICIONES = {
    EstadoPedido.CREADO: {EstadoPedido.EN_PREPARACION, EstadoPedido.CANCELADO},
    EstadoPedido.EN_PREPARACION: {EstadoPedido.LISTO, EstadoPedido.CANCELADO},
    EstadoPedido.LISTO: {EstadoPedido.EN_CAMINO, EstadoPedido.CANCELADO},
    EstadoPedido.EN_CAMINO: {EstadoPedido.ENTREGADO, EstadoPedido.CANCELADO},
    EstadoPedido.ENTREGADO: set(),
    EstadoPedido.CANCELADO: set(),
}


class TransicionInvalida(Exception):
    pass


@dataclass(eq=False)
class Pedido:
    id: int
    habitante: Any
    restaurante: Any
    destino: tuple
    precio: float
    pago_repartidor: float
    creado: int
    estado: EstadoPedido = EstadoPedido.CREADO
    repartidor: Any = None
    aceptado_rest: int | None = None
    listo: int | None = None
    recogido: int | None = None
    entregado: int | None = None
    cancelado: int | None = None
    tiempo_preparacion: int | None = None  # lo sabe el restaurante, no el habitante
    datos: dict = field(default_factory=dict)

    # ------------------------------------------------------------- transiciones
    def _pasar_a(self, nuevo: EstadoPedido) -> None:
        if nuevo not in TRANSICIONES[self.estado]:
            raise TransicionInvalida(f"Pedido {self.id}: {self.estado.name} -> {nuevo.name}")
        self.estado = nuevo

    def iniciar_preparacion(self, t: int, tiempo_preparacion: int) -> None:
        self._pasar_a(EstadoPedido.EN_PREPARACION)
        self.aceptado_rest = t
        self.tiempo_preparacion = tiempo_preparacion

    def marcar_listo(self, t: int) -> None:
        self._pasar_a(EstadoPedido.LISTO)
        self.listo = t

    def asignar(self, repartidor: Any) -> None:
        if not self.disponible:
            raise TransicionInvalida(f"Pedido {self.id} no está disponible para asignar")
        self.repartidor = repartidor

    def recoger(self, t: int) -> None:
        if self.repartidor is None:
            raise TransicionInvalida(f"Pedido {self.id} se recoge sin repartidor")
        self._pasar_a(EstadoPedido.EN_CAMINO)
        self.recogido = t

    def entregar(self, t: int) -> None:
        self._pasar_a(EstadoPedido.ENTREGADO)
        self.entregado = t

    def cancelar(self, t: int) -> None:
        self._pasar_a(EstadoPedido.CANCELADO)
        self.cancelado = t

    # ------------------------------------------------------------------ consultas
    @property
    def terminado(self) -> bool:
        return self.estado in TERMINALES

    @property
    def disponible(self) -> bool:
        """Sin repartidor y aún no recogido: lo pueden ver los repartidores libres."""
        return self.repartidor is None and self.estado in (
            EstadoPedido.CREADO, EstadoPedido.EN_PREPARACION, EstadoPedido.LISTO)

    def edad(self, t: int) -> int:
        return t - self.creado

    @property
    def tiempo_entrega(self) -> int | None:
        return None if self.entregado is None else self.entregado - self.creado
