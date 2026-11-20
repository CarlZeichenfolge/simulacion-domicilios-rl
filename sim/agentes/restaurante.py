"""Restaurante: agente basado en modelo."""
from __future__ import annotations

from sim.agentes.base import AgenteBase
from sim.pedidos import EstadoPedido, Pedido


class Restaurante(AgenteBase):
    """Mantiene un modelo interno de su cocina (cola, fogones ocupados y
    demanda por hora) y actúa según ese modelo: no reacciona solo al pedido
    que llega, sino al estado acumulado de su cocina.

    Por qué basado en modelo y no reactivo puro: su decisión (qué empezar a
    cocinar) depende de cosas que no se ven en la percepción del instante,
    como cuántos fogones siguen ocupados por pedidos de minutos atrás.
    """

    def __init__(self, model, cfg: dict, nodo: tuple):
        super().__init__(model)
        self.nombre = cfg["nombre"]
        self.nodo = nodo
        self.precio_medio = cfg["precio_medio"]
        self.precio_desv = cfg["precio_desv"]
        self.prep_media = cfg["prep_media"]
        self.prep_desv = cfg["prep_desv"]
        self.capacidad = cfg["capacidad"]
        self.cola: list[Pedido] = []      # esperando fogón (FIFO)
        self.cocina: list[Pedido] = []    # en preparación
        self.demanda_por_hora = [0] * 24  # acumulado del episodio
        self.historial_demanda: dict[tuple[int, int], int] = {}  # (día, hora) -> pedidos
        self.preparados = 0

    # ------------------------------------------------------------ interfaz
    def cotizar(self) -> float:
        """Precio con algo de variación (plato, adiciones), redondeado a 100."""
        precio = self.model.random.gauss(self.precio_medio, self.precio_desv)
        return max(round(precio, -2), 0.5 * self.precio_medio)

    def recibir_pedido(self, pedido: Pedido) -> None:
        self.cola.append(pedido)
        reloj = self.model.reloj
        self.demanda_por_hora[reloj.hora] += 1
        clave = (reloj.dia, reloj.hora)
        self.historial_demanda[clave] = self.historial_demanda.get(clave, 0) + 1

    def estimar_espera(self) -> float:
        """Su propio modelo de cuánto tardaría un pedido nuevo (no lo ve el habitante)."""
        tandas = (len(self.cola) + len(self.cocina)) / self.capacidad
        return (tandas + 1) * self.prep_media

    # --------------------------------------------------------------- ciclo
    def percibir(self) -> dict:
        # Los cancelados se descartan: liberan fogón y no se cocinan.
        return {
            "cola": [p for p in self.cola if p.estado == EstadoPedido.CREADO],
            "cocina": [p for p in self.cocina if p.estado == EstadoPedido.EN_PREPARACION],
        }

    def decidir(self, percepcion: dict) -> dict:
        """Regla FIFO con capacidad: primero termina lo que está listo, luego
        llena los fogones libres con los pedidos más antiguos.

        Por qué FIFO: es lo justo con el cliente y minimiza cancelaciones,
        porque el pedido más viejo es el más cerca de los 60 minutos.
        """
        terminados = [p for p in percepcion["cocina"]
                      if self.t - p.aceptado_rest >= p.tiempo_preparacion]
        cupos = self.capacidad - (len(percepcion["cocina"]) - len(terminados))
        return {"terminar": terminados, "iniciar": percepcion["cola"][:max(0, cupos)]}

    def actuar(self, accion: dict) -> None:
        for pedido in accion["terminar"]:
            pedido.marcar_listo(self.t)
            self.preparados += 1
        for pedido in accion["iniciar"]:
            tiempo = max(1, round(self.model.random.gauss(self.prep_media, self.prep_desv)))
            pedido.iniciar_preparacion(self.t, tiempo)
        self.cocina = [p for p in self.cocina + accion["iniciar"]
                       if p.estado == EstadoPedido.EN_PREPARACION]
        self.cola = [p for p in self.cola if p.estado == EstadoPedido.CREADO]
