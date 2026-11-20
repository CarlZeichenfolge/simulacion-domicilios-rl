"""Habitante: agente basado en utilidad que aprende qué restaurante le conviene."""
from __future__ import annotations

from sim.agentes.base import AgenteBase
from sim.pedidos import EstadoPedido, Pedido
from sim.reloj import MINUTOS_POR_DIA

ACTIVIDADES = ("dormir", "trabajar", "estudiar", "ejercicio", "libre")
ACTIVIDADES_FUERA = ("trabajar", "estudiar")  # en esas está en la oficina / universidad


def expandir_horario(tramos: list) -> list[str]:
    """Convierte [[0, 6, 'dormir'], ...] en una lista de 1440 actividades.

    Por qué precalcular: consultar la actividad es O(1) y se hace 40 veces
    por tick; recorrer los tramos cada vez sería lo más caro del habitante.
    """
    por_minuto: list[str | None] = [None] * MINUTOS_POR_DIA
    for inicio, fin, actividad in tramos:
        if actividad not in ACTIVIDADES:
            raise ValueError(f"Actividad desconocida en el horario: {actividad}")
        for minuto in range(inicio * 60, fin * 60):
            por_minuto[minuto] = actividad
    if None in por_minuto:
        raise ValueError("El horario no cubre las 24 horas")
    return por_minuto


class Habitante(AgenteBase):
    """Cuando el hambre pasa el umbral compara la utilidad de cocinar, salir a
    comer o pedir domicilio, y escoge la mayor.

    Utilidad = gusto - peso_precio * precio(miles) - peso_tiempo * minutos.
    El gusto y los pesos son propios de cada persona (heterogeneidad). Lo que
    NO sabe es cuánto tarda de verdad cada restaurante: eso lo aprende con
    el bandit epsilon-greedy de su política.
    """

    def __init__(self, model, cfg: dict, casa: tuple, lugar_fuera: tuple, tipo: str,
                 horario: list[str], desfase: int, gusto: dict[str, float],
                 peso_precio: float, peso_tiempo: float, politica):
        super().__init__(model, politica)
        self.cfg = cfg
        self.casa = casa
        self.lugar_fuera = lugar_fuera   # oficina o universidad
        self.tipo = tipo                 # trabajador | estudiante
        self.horario = horario
        self.desfase = desfase
        self.gusto = gusto
        self.peso_precio = peso_precio
        self.peso_tiempo = peso_tiempo
        rng = model.random
        self.hambre = rng.uniform(*cfg["hambre_inicial"])
        self.energia = float(cfg["energia_inicial"])
        self.dinero = float(cfg["dinero_inicial"])
        self.pedido: Pedido | None = None
        self.comida_en_curso: str | None = None  # "cocinar" | "salir"
        self.ocupado_hasta: int | None = None
        self.ultima_satisfaccion: float | None = None

    # -------------------------------------------------------------- estado
    @property
    def actividad(self) -> str:
        minuto = (self.model.reloj.minuto_del_dia - self.desfase) % MINUTOS_POR_DIA
        return self.horario[minuto]

    @property
    def ubicacion(self) -> tuple:
        return self.lugar_fuera if self.actividad in ACTIVIDADES_FUERA else self.casa

    def utilidad(self, gusto: float, costo_pesos: float, minutos: float) -> float:
        return gusto - self.peso_precio * costo_pesos / 1000 - self.peso_tiempo * minutos

    # --------------------------------------------------------------- ciclo
    def percibir(self) -> dict:
        """Ve su propio cuerpo, su agenda, si llueve y el MENÚ (precio publicado)
        de cada restaurante. No ve colas de cocina ni tiempos reales."""
        domicilio = self.model.cfg["pedidos"]["tarifa_domicilio"]
        asequibles = [r.nombre for r in self.model.restaurantes.values()
                      if self.dinero >= r.precio_medio + domicilio]
        return {"actividad": self.actividad, "hambre": self.hambre, "energia": self.energia,
                "dinero": self.dinero, "lluvia": self.model.eventos.lloviendo,
                "pedido": self.pedido, "restaurantes_asequibles": asequibles}

    def decidir(self, percepcion: dict) -> dict:
        if self.pedido is not None:
            return {"tipo": "recibir"} if self.pedido.terminado else {"tipo": "nada"}
        if (self.comida_en_curso is not None or percepcion["actividad"] == "dormir"
                or percepcion["hambre"] < self.cfg["umbral_hambre"]):
            return {"tipo": "nada"}

        opciones = self.opciones_comida(percepcion)
        if not opciones:
            return {"tipo": "nada"}  # sin plata ni energía: aguanta hambre
        mejor = max(opciones, key=opciones.get)
        if mejor != "pedir":
            return {"tipo": mejor}
        eleccion = self.politica.elegir(percepcion)
        return {"tipo": "pedir", "nombre": eleccion["nombre"]}

    def opciones_comida(self, percepcion: dict) -> dict[str, float]:
        """Utilidad de cada alternativa factible.

        Por qué la lluvia entra como penalización a "salir" y no como un
        multiplicador de pedidos: así el aumento de domicilios con lluvia
        SALE de la decisión del agente en vez de estar impuesto desde afuera.
        """
        c, s = self.cfg["cocinar"], self.cfg["salir"]
        opciones = {}
        if (percepcion["actividad"] not in ACTIVIDADES_FUERA
                and percepcion["energia"] >= c["energia_minima"]
                and percepcion["dinero"] >= c["costo"]):
            opciones["cocinar"] = self.utilidad(self.gusto["cocinar"], c["costo"], c["tiempo"])
        if percepcion["dinero"] >= s["costo"]:
            u = self.utilidad(s["gusto"], s["costo"], s["tiempo"])
            opciones["salir"] = u - (s["penalizacion_lluvia"] if percepcion["lluvia"] else 0.0)
        if percepcion["restaurantes_asequibles"]:
            opciones["pedir"] = self.politica.mejor_valor(percepcion["restaurantes_asequibles"])
        return opciones

    def actuar(self, accion: dict) -> None:
        self._vivir()
        tipo = accion["tipo"]
        if tipo == "recibir":
            self._recibir()
        elif tipo == "pedir":
            restaurante = self.model.restaurantes[accion["nombre"]]
            self.pedido = self.model.crear_pedido(self, restaurante, self.ubicacion)
            self.model.registrar_comida(self, "pedir", accion["nombre"])
        elif tipo in ("cocinar", "salir"):
            datos = self.cfg[tipo]
            self.comida_en_curso = tipo
            self.ocupado_hasta = self.t + datos["tiempo"]
            self.dinero -= datos["costo"]
            if tipo == "cocinar":
                self.energia = max(0.0, self.energia - datos["gasto_energia"])
            self.model.registrar_comida(self, tipo, None)

    # ------------------------------------------------------------- internos
    def _vivir(self) -> None:
        """Dinámica pasiva: el hambre y la energía cambian según la actividad."""
        actividad = self.actividad
        self.hambre = min(100.0, self.hambre + self.cfg["tasa_hambre"][actividad])
        self.energia = min(100.0, max(0.0, self.energia + self.cfg["tasa_energia"][actividad]))
        if self.t > 0 and self.model.reloj.minuto_del_dia == 0:
            self.dinero += self.cfg["ingreso_diario"]
        if self.ocupado_hasta is not None and self.t >= self.ocupado_hasta:
            self._comer()
            self.comida_en_curso = None
            self.ocupado_hasta = None

    def _comer(self) -> None:
        self.hambre = max(0.0, self.hambre - self.cfg["saciedad_comida"])

    def _recibir(self) -> None:
        """Cierra el pedido y aprende de él (recompensa = satisfacción real)."""
        pedido = self.pedido
        self.pedido = None
        accion = {"tipo": "restaurante", "nombre": pedido.restaurante.nombre}
        if pedido.estado == EstadoPedido.ENTREGADO:
            total = pedido.precio + self.model.cfg["pedidos"]["tarifa_domicilio"]
            self.dinero -= total
            self._comer()
            satisfaccion = self.utilidad(self.gusto[pedido.restaurante.nombre], total,
                                         pedido.tiempo_entrega)
        else:
            satisfaccion = self.cfg["aprendizaje"]["recompensa_cancelado"]
        self.ultima_satisfaccion = satisfaccion
        self.model.registrar_satisfaccion(self, pedido, satisfaccion)
        self.politica.aprender(None, accion, satisfaccion, None)
