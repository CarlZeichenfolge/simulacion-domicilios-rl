"""CiudadModel: el entorno que contiene ciudad, reloj, eventos, pedidos y agentes."""
from __future__ import annotations

import random

import mesa

from sim.agentes import Habitante, Repartidor, Restaurante
from sim.agentes.habitante import expandir_horario
from sim.agentes.repartidor import EstadoRepartidor
from sim.ciudad import Ciudad
from sim.eventos import GestorEventos
from sim.metricas import crear_datacollector
from sim.pedidos import Pedido
from sim.politicas import crear_politica_repartidor
from sim.politicas.habitante_egreedy import PoliticaHabitanteEGreedy
from sim.reloj import MINUTOS_POR_DIA, Reloj

# Orden de activación por tipo dentro de un minuto. Por qué fijo: primero
# se piden cosas, luego la cocina las procesa y al final los repartidores
# ven qué hay. Dentro de cada tipo el orden se baraja con la semilla para
# que ningún repartidor tenga ventaja sistemática por ir siempre primero.
ORDEN_ACTIVACION = (Habitante, Restaurante, Repartidor)


class CiudadModel(mesa.Model):
    """Entorno multiagente, estocástico, dinámico, parcialmente observable,
    discreto en el tiempo y secuencial.

    `politica_repartidor` permite pasar una política ya entrenada (o en
    entrenamiento) para que la tabla Q sobreviva entre episodios.
    """

    def __init__(self, config: dict, semilla: int | None = None, politica_repartidor=None):
        semilla = config["semilla"] if semilla is None else semilla
        super().__init__(rng=semilla)
        self.cfg = config
        self.semilla = semilla
        self.reloj = Reloj(config["tiempo"]["hora_inicio"])
        self.duracion = MINUTOS_POR_DIA * config["tiempo"]["dias_por_episodio"]

        # El mapa puede tener su propia semilla: es el escenario fijo del
        # experimento; lo que varía entre episodios son eventos y decisiones.
        semilla_mapa = config["ciudad"].get("semilla_mapa")
        rng_mapa = random.Random(semilla_mapa) if semilla_mapa is not None else self.random
        self.ciudad = Ciudad(config["ciudad"], rng_mapa)
        # azar_propio: la lluvia y los trancones usan su propio generador (sembrado con la
        # misma semilla), así dos ciudades con políticas distintas viven el MISMO clima. Lo
        # usan los duelos antes/después; apagado, todo sale de model.random como siempre.
        rng_eventos = (random.Random(f"eventos-{semilla}") if config["eventos"]["azar_propio"]
                       else self.random)
        self.eventos = GestorEventos(config["eventos"], self.ciudad, rng_eventos)

        self.pedidos: list[Pedido] = []
        self.pedidos_activos: list[Pedido] = []
        self.conteo = {"creados": 0, "entregados": 0, "cancelados": 0}
        self.satisfacciones: list[float] = []
        self.comidas: list[dict] = []       # cada decisión de comer (para preferencias)
        self.minutos_lluvia = 0

        self.restaurantes: dict[str, Restaurante] = {}
        self._crear_restaurantes()
        self.politica_repartidor = politica_repartidor or crear_politica_repartidor(config["repartidores"])
        self._crear_repartidores()
        self._crear_habitantes()

        self.recolectar_cada = config["metricas"]["recolectar_cada"]
        self.datacollector = crear_datacollector()

    # ---------------------------------------------------------------- creación
    def _crear_restaurantes(self) -> None:
        for cfg in self.cfg["ciudad"]["restaurantes"]:
            nodo = self.ciudad.restaurantes[cfg["nombre"]]
            self.restaurantes[cfg["nombre"]] = Restaurante(self, cfg, nodo)

    def _crear_repartidores(self) -> None:
        cfg = self.cfg["repartidores"]
        nodos = sorted(self.ciudad.grafo.nodes)
        for _ in range(cfg["cantidad"]):
            nodo = tuple(cfg["nodo_inicial"]) if cfg["nodo_inicial"] is not None else self.random.choice(nodos)
            # Todos comparten la misma política (misma tabla Q).
            Repartidor(self, nodo, self.politica_repartidor, cfg)

    def _crear_habitantes(self) -> None:
        cfg = self.cfg["habitantes"]
        if cfg["cantidad"] == 0:
            return
        rng = self.random
        casas = self.ciudad.nodos_de_tipo("casa")
        oficinas = self.ciudad.nodos_de_tipo("oficina")
        universidades = self.ciudad.nodos_de_tipo("universidad")
        horarios = {tipo: expandir_horario(tramos) for tipo, tramos in cfg["horarios"].items()}
        pref = cfg["preferencias"]
        domicilio = self.cfg["pedidos"]["tarifa_domicilio"]
        for _ in range(cfg["cantidad"]):
            estudiante = rng.random() < cfg["proporcion_estudiantes"]
            tipo = "estudiante" if estudiante else "trabajador"
            lugar_fuera = rng.choice(universidades if estudiante else oficinas)
            gusto = {nombre: min(10.0, max(0.0, rng.gauss(base, pref["gusto_desv"])))
                     for nombre, base in pref["gusto_base"].items()}
            for opcion in ("cocinar", "salir"):
                gusto[opcion] = min(10.0, max(0.0, rng.gauss(cfg[opcion]["gusto"], pref["gusto_desv"])))
            peso_precio = rng.uniform(*pref["peso_precio"])
            peso_tiempo = rng.uniform(*pref["peso_tiempo"])
            # Creencia inicial: conoce gusto y menú, pero supone que todos
            # tardan lo mismo (no sabe aún que el sushi es lento).
            t0 = cfg["aprendizaje"]["tiempo_esperado_inicial"]
            iniciales = {nombre: gusto[nombre] - peso_precio * (r.precio_medio + domicilio) / 1000
                         - peso_tiempo * t0 for nombre, r in self.restaurantes.items()}
            politica = PoliticaHabitanteEGreedy(cfg["aprendizaje"], iniciales)
            Habitante(self, cfg, casa=rng.choice(casas), lugar_fuera=lugar_fuera, tipo=tipo,
                      horario=horarios[tipo],
                      desfase=rng.randint(-cfg["desfase_horario_max"], cfg["desfase_horario_max"]),
                      gusto=gusto, peso_precio=peso_precio, peso_tiempo=peso_tiempo,
                      politica=politica)

    # ------------------------------------------------------------------ pedidos
    def crear_pedido(self, habitante, restaurante: Restaurante, destino: tuple) -> Pedido:
        """El pago al repartidor crece con la distancia y a veces trae propina.

        Por qué propina aleatoria: hace que "mejor pagado" y "más cercano" no
        sean siempre el mismo pedido, si no, Q-learning no tendría qué escoger.
        """
        cfg = self.cfg["pedidos"]
        cuadras = self.ciudad.distancia_cuadras(restaurante.nodo, destino)
        propina = 0.0
        if self.random.random() < cfg["prob_propina"]:
            propina = round(self.random.uniform(cfg["propina_min"], cfg["propina_max"]), -2)
        pedido = Pedido(id=len(self.pedidos), habitante=habitante, restaurante=restaurante,
                        destino=destino, precio=restaurante.cotizar(),
                        pago_repartidor=cfg["pago_base"] + cfg["pago_por_cuadra"] * cuadras + propina,
                        creado=self.reloj.tick)
        pedido.datos["foto"] = self._foto_pedido(restaurante, cuadras)
        self.pedidos.append(pedido)
        self.pedidos_activos.append(pedido)
        self.conteo["creados"] += 1
        restaurante.recibir_pedido(pedido)
        return pedido

    def _foto_pedido(self, restaurante: Restaurante, cuadras_entrega: int) -> dict:
        """Lo que la plataforma sabe en el instante en que nace el pedido.

        Por qué solo el instante de creación: es lo único disponible cuando
        hay que estimarle al cliente cuánto tardará. Guardar algo posterior
        (quién lo aceptó, si llovió después) sería fuga de datos para el
        predictor supervisado de experimentos/prediccion.py.
        No usa self.random: tomar la foto no altera la simulación.
        """
        libres = [r for r in self.repartidores if r.estado == EstadoRepartidor.LIBRE]
        return {
            "hora": self.reloj.minuto_del_dia / 60,
            "lluvia": int(self.eventos.lloviendo),
            "trancon": int(self.eventos.trancon_activo),
            "restaurante": restaurante.nombre,
            "cuadras_entrega": cuadras_entrega,
            "pedidos_en_cocina": len(restaurante.cola) + len(restaurante.cocina),
            "pedidos_sin_repartidor": len(self.pedidos_disponibles()),
            "repartidores_libres": len(libres),
            # Sin repartidores libres el dato no existe: queda faltante (NaN) a
            # propósito, y el predictor lo imputa y marca (faltante informativo).
            "cuadras_repartidor_libre": min((self.ciudad.distancia_cuadras(r.nodo, restaurante.nodo)
                                             for r in libres), default=float("nan")),
        }

    def pago_actual(self, pedido: Pedido) -> float:
        """Pago ofrecido ahora: crece con cada minuto que el pedido espera sin dueño.

        Por qué: es el incentivo de las apps reales para que alguien rescate los
        pedidos viejos. Sin esto, al repartidor que aprende le conviene más un
        pedido nuevo que uno a punto de cancelarse (paga igual y arriesga menos).
        """
        return pedido.pago_repartidor + self.cfg["pedidos"]["incentivo_por_minuto"] * pedido.edad(self.reloj.tick)

    def pedidos_disponibles(self) -> list[Pedido]:
        return [p for p in self.pedidos_activos if p.disponible]

    def notificar_entrega(self, pedido: Pedido) -> None:
        self.conteo["entregados"] += 1

    def _revisar_cancelaciones(self) -> None:
        espera_maxima = self.cfg["pedidos"]["espera_maxima"]
        t = self.reloj.tick
        for pedido in self.pedidos_activos:
            if not pedido.terminado and pedido.edad(t) >= espera_maxima:
                pedido.cancelar(t)
                self.conteo["cancelados"] += 1

    # ------------------------------------------------------------------ registros
    def registrar_comida(self, habitante, tipo: str, restaurante: str | None) -> None:
        self.comidas.append({"tick": self.reloj.tick, "dia": self.reloj.dia,
                             "habitante": habitante.unique_id, "tipo": tipo,
                             "restaurante": restaurante, "lluvia": self.eventos.lloviendo})

    def registrar_satisfaccion(self, habitante, pedido: Pedido, satisfaccion: float) -> None:
        self.satisfacciones.append(satisfaccion)

    # --------------------------------------------------------------------- paso
    def step(self) -> None:
        for _ in self.minuto():
            pass  # sin agente externo el minuto nunca se pausa

    def minuto(self, externo=None):
        """Un minuto de simulación escrito como generador.

        Por qué generador: el entorno Gymnasium (sim/entorno_gym.py) necesita
        que el modelo se DETENGA justo cuando el repartidor `externo` tiene que
        decidir, le devuelva el control a quien llama a env.step(accion) y
        luego siga desde ese mismo punto. Un generador hace exactamente eso
        sin hilos. Con externo=None nunca se pausa y el resultado es idéntico
        al de antes: shuffle() baraja con el mismo azar que shuffle_do().
        """
        self.eventos.actualizar(self.reloj)
        if self.eventos.lloviendo:
            self.minutos_lluvia += 1
        self._revisar_cancelaciones()
        for clase in ORDEN_ACTIVACION:
            grupo = self.agents_by_type.get(clase)
            if not grupo:
                continue
            if externo is None or clase is not Repartidor:
                grupo.shuffle_do("step")
                continue
            for agente in grupo.shuffle():
                if agente is externo and agente.necesita_decision():
                    yield agente
                agente.step()
        self.pedidos_activos = [p for p in self.pedidos_activos if not p.terminado]
        if self.recolectar_cada and self.reloj.tick % self.recolectar_cada == 0:
            self.datacollector.collect(self)
        self.reloj.avanzar()
        if self.reloj.tick >= self.duracion:
            self.running = False

    def correr(self) -> None:
        """Corre el episodio completo sin vista."""
        while self.running:
            self.step()

    # ------------------------------------------------------------------ atajos
    @property
    def repartidores(self) -> list[Repartidor]:
        return list(self.agents_by_type.get(Repartidor, []))

    @property
    def habitantes(self) -> list[Habitante]:
        return list(self.agents_by_type.get(Habitante, []))
