"""Repartidor: agente basado en metas, con aprendizaje."""
from __future__ import annotations

from enum import Enum

from sim.agentes.base import AgenteBase
from sim.pedidos import EstadoPedido, Pedido


class EstadoRepartidor(Enum):
    LIBRE = "libre"
    YENDO_A_RESTAURANTE = "yendo_a_restaurante"
    YENDO_A_CLIENTE = "yendo_a_cliente"


class Repartidor(AgenteBase):
    """Su meta es entregar el pedido aceptado; para lograrla planea la ruta con
    A* y la sigue. La política solo decide QUÉ pedido aceptar (o esperar).

    La recompensa de una decisión se conoce cuando la acción termina:
    pago - costo_por_minuto * minutos, o una penalización si se canceló.
    Por eso la decisión pendiente se guarda y se aprende al cerrarla.
    """

    def __init__(self, model, nodo: tuple, politica, cfg: dict):
        super().__init__(model, politica)
        self.nodo = nodo
        self.estado = EstadoRepartidor.LIBRE
        self.pedido: Pedido | None = None
        self.ruta: list[tuple] = []   # próximos nodos, sin incluir el actual
        self.progreso = 0.0           # fracción recorrida de la arista nodo -> ruta[0]
        self.costo_por_minuto = cfg["costo_por_minuto"]
        self.costo_espera_por_minuto = cfg["costo_espera_por_minuto"]
        self.penalizacion_cancelado = cfg["penalizacion_cancelado"]
        # Como la "tasa de aceptación" de las apps reales: sin esto el agente
        # aprende a escoger solo los pedidos cómodos y deja vencer el resto.
        self.penalizacion_rechazo = cfg["penalizacion_rechazo"]
        self.rechazos = 0
        self.duracion_espera = cfg["duracion_espera"]
        self.escala = cfg["escala_recompensa"]
        self.ganancias = 0.0
        self.costos = 0.0
        self.recompensa_acumulada = 0.0
        self.entregas = 0
        self.cancelaciones = 0
        self._decision: tuple[dict, dict, int] | None = None  # (percepción, acción, tick)
        self._espera_hasta: int | None = None
        self._percepcion: dict | None = None
        # Lo que vio y escogió la última vez que la política decidió, y cómo le fue a la
        # última acción cerrada. Solo para explicarlo en la vista 3D: no cambia ninguna decisión.
        self.ultima_decision: dict | None = None
        self.ultimo_resultado: dict | None = None

    @property
    def espera_hasta(self) -> int | None:
        """Tick en que termina la espera (rechazo) en curso, o None si no está esperando."""
        return self._espera_hasta

    # ------------------------------------------------------------------ ciclo
    def step(self) -> None:
        # Antes de percibir se cierran las acciones que ya terminaron solas
        # (pedido cancelado, espera cumplida): de ahí sale la recompensa.
        self._cerrar_acciones_terminadas()
        super().step()

    def necesita_decision(self) -> bool:
        """¿En su próximo step la política tendrá que escoger? (lo usa el entorno Gymnasium).

        Cierra primero lo que ya terminó, igual que step(): así la recompensa
        de la acción anterior ya está contada cuando se pide la siguiente.
        Llamarlo antes de step() no cambia nada, porque cerrar dos veces no
        hace nada la segunda vez.
        """
        self._cerrar_acciones_terminadas()
        return (self.estado == EstadoRepartidor.LIBRE and self._espera_hasta is None
                and bool(self.model.pedidos_disponibles()))

    def percibir(self) -> dict:
        """Solo ve pedidos sin repartidor, su distancia en cuadras al
        restaurante, el pago ofrecido, la hora y si llueve. No ve pedidos
        futuros ni el tiempo real de cocina: entorno parcialmente observable.
        """
        modelo = self.model
        pedidos = []
        if self.estado == EstadoRepartidor.LIBRE:
            ciudad = modelo.ciudad
            pedidos = [{"pedido": p,
                        "distancia": ciudad.distancia_cuadras(self.nodo, p.restaurante.nodo),
                        "distancia_entrega": ciudad.distancia_cuadras(p.restaurante.nodo, p.destino),
                        "pago": modelo.pago_actual(p),
                        "edad": p.edad(self.t),
                        "listo": p.estado == EstadoPedido.LISTO}
                       for p in modelo.pedidos_disponibles()]
        return {"nodo": self.nodo, "hora": modelo.reloj.hora,
                "lluvia": modelo.eventos.lloviendo, "pedidos": pedidos}

    def decidir(self, percepcion: dict) -> dict:
        self._percepcion = percepcion
        if self.estado != EstadoRepartidor.LIBRE or self._espera_hasta is not None:
            return {"tipo": "continuar"}
        if not percepcion["pedidos"]:
            # Sin pedidos la única acción legal es quedarse: no hay nada que decidir ni aprender.
            return {"tipo": "inactivo"}
        accion = self.politica.elegir(percepcion)
        self.ultima_decision = {"tick": self.t, "minuto_del_dia": self.model.reloj.minuto_del_dia,
                                "percepcion": percepcion, "accion": accion}
        return accion

    def actuar(self, accion: dict) -> None:
        tipo = accion["tipo"]
        if tipo == "aceptar":
            self._aceptar(accion)
        elif tipo == "esperar":
            self._abrir_decision(accion)
            self._espera_hasta = self.t + self.duracion_espera
        elif tipo == "continuar":
            self._avanzar()

    # ---------------------------------------------------------------- acciones
    def _aceptar(self, accion: dict) -> None:
        pedido = accion["pedido"]
        pedido.pago_repartidor = self.model.pago_actual(pedido)  # el incentivo queda fijo al aceptar
        pedido.asignar(self)
        self.pedido = pedido
        self._abrir_decision(accion)
        self.estado = EstadoRepartidor.YENDO_A_RESTAURANTE
        self._planear(pedido.restaurante.nodo)

    def _planear(self, destino: tuple) -> None:
        """A* desde donde está; si va a mitad de una calle, termina esa calle primero."""
        modelo = self.model
        fv = modelo.eventos.factor_velocidad
        if self.ruta and self.progreso > 0:
            siguiente = self.ruta[0]
            self.ruta = [siguiente] + modelo.ciudad.ruta(siguiente, destino, fv)[1:]
        else:
            self.progreso = 0.0
            self.ruta = modelo.ciudad.ruta(self.nodo, destino, fv)[1:]

    def _destino_actual(self) -> tuple | None:
        if self.estado == EstadoRepartidor.YENDO_A_RESTAURANTE:
            return self.pedido.restaurante.nodo
        if self.estado == EstadoRepartidor.YENDO_A_CLIENTE:
            return self.pedido.destino
        return None

    def _avanzar(self) -> None:
        if self.estado == EstadoRepartidor.LIBRE:
            return
        if self.model.eventos.cambio_trafico and self.ruta:
            self._planear(self._destino_actual())
        self._mover(1.0)
        if self.ruta:
            return
        if self.estado == EstadoRepartidor.YENDO_A_RESTAURANTE:
            if self.pedido.estado == EstadoPedido.LISTO:
                self.pedido.recoger(self.t)
                self.estado = EstadoRepartidor.YENDO_A_CLIENTE
                self._planear(self.pedido.destino)
            # si no está listo, espera en la puerta del restaurante
        elif self.estado == EstadoRepartidor.YENDO_A_CLIENTE:
            self._entregar()

    def _mover(self, minutos: float) -> None:
        """Avanza por la ruta. El progreso se guarda como fracción de la calle
        para que, si cambia el trancón a mitad de camino, no se pierda avance."""
        ciudad, fv = self.model.ciudad, self.model.eventos.factor_velocidad
        while self.ruta and minutos > 1e-9:
            costo = ciudad.tiempo_arista(self.nodo, self.ruta[0], fv)
            restante = (1.0 - self.progreso) * costo
            if minutos >= restante:
                minutos -= restante
                self.nodo = self.ruta.pop(0)
                self.progreso = 0.0
            else:
                self.progreso += minutos / costo
                minutos = 0.0

    def _entregar(self) -> None:
        pedido = self.pedido
        pedido.entregar(self.t)
        self.model.notificar_entrega(pedido)
        self.entregas += 1
        self.ganancias += pedido.pago_repartidor
        minutos = self.t - self._decision[2]
        costo = self.costo_por_minuto * minutos
        self.costos += costo
        self._liberar()
        self._cerrar_decision(pedido.pago_repartidor - costo, minutos, "entrego", pedido)

    def _cerrar_acciones_terminadas(self) -> None:
        if self.pedido is not None and self.pedido.estado == EstadoPedido.CANCELADO:
            pedido = self.pedido
            minutos = self.t - self._decision[2]
            costo = self.costo_por_minuto * minutos
            self.costos += costo
            self.cancelaciones += 1
            self._liberar()
            self._cerrar_decision(-(self.penalizacion_cancelado + costo), minutos, "cancelado", pedido)
        elif self._espera_hasta is not None and self.t >= self._espera_hasta:
            # Solo se elige esperar si había pedidos, así que siempre es un rechazo.
            self._espera_hasta = None
            costo = self.costo_espera_por_minuto * self.duracion_espera + self.penalizacion_rechazo
            self.costos += costo
            self.rechazos += 1
            self._cerrar_decision(-costo, self.duracion_espera, "espero", None)

    def _liberar(self) -> None:
        self.pedido = None
        self.estado = EstadoRepartidor.LIBRE
        self.ruta = []
        self.progreso = 0.0  # si se canceló a mitad de calle, se queda en el último nodo

    def _abrir_decision(self, accion: dict) -> None:
        self._decision = (self._percepcion, accion, self.t)

    def _cerrar_decision(self, recompensa_pesos: float, minutos: int, motivo: str, pedido: Pedido | None) -> None:
        """Convierte el resultado de la acción en recompensa y se la pasa a la política."""
        percepcion, accion, _ = self._decision
        self._decision = None
        recompensa = recompensa_pesos * self.escala
        self.recompensa_acumulada += recompensa
        self.ultimo_resultado = {"tick": self.t, "motivo": motivo, "pedido": pedido, "pesos": recompensa_pesos,
                                 "recompensa": recompensa, "minutos": minutos}
        self.politica.aprender(percepcion, accion, recompensa, self.percibir(), max(1, minutos))

    # ------------------------------------------------------------------ vista
    def posicion_continua(self) -> tuple[float, float]:
        """Coordenadas en cuadras (interpoladas) para que las vistas lo dibujen."""
        if self.ruta and self.progreso > 0:
            (x0, y0), (x1, y1) = self.nodo, self.ruta[0]
            return x0 + (x1 - x0) * self.progreso, y0 + (y1 - y0) * self.progreso
        return float(self.nodo[0]), float(self.nodo[1])
