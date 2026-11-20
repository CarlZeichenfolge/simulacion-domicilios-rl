"""Vista 3D con Ursina: maqueta low poly de la ciudad para la demo.

Igual que la vista 2D, solo LEE el modelo y lo dibuja; nunca cambia su estado.

Controles:
    clic derecho + arrastrar   girar la cámara
    rueda                      zoom
    ESPACIO                    pausa / sigue
    ↑ / ↓                      más o menos minutos simulados por cuadro
    ESC                        salir
"""
from __future__ import annotations

import math
import random

from sim.agentes import EstadoRepartidor
from sim.model import CiudadModel
from sim.pedidos import EstadoPedido

ESCALA = 3.0  # unidades de mundo por cuadra

HEX_EDIFICIO = {"casa": "#c9a27e", "oficina": "#7d93b2", "universidad": "#9a74c4"}
ALTURA = {"casa": 0.8, "oficina": 2.8, "universidad": 1.6}
HEX_RESTAURANTE = {"Pizza": "#e34948", "Hamburguesas": "#eda100", "Sushi": "#1baf7a"}
HEX_REPARTIDOR = {EstadoRepartidor.LIBRE: "#e6e6e6",
                  EstadoRepartidor.YENDO_A_RESTAURANTE: "#ffd400",
                  EstadoRepartidor.YENDO_A_CLIENTE: "#3fb6ff"}
HEX_PEDIDO = {EstadoPedido.CREADO: "#b4b4b4", EstadoPedido.EN_PREPARACION: "#eda100",
              EstadoPedido.LISTO: "#78dc78", EstadoPedido.EN_CAMINO: "#3fb6ff"}


class Vista3D:
    def __init__(self, modelo: CiudadModel):
        self.modelo = modelo
        cfg = modelo.cfg["vista_2d"]  # la velocidad inicial se comparte con la 2D
        self.ticks_por_cuadro = cfg["ticks_por_cuadro"]
        self.pausa = False
        self.calles: dict[tuple, object] = {}
        self.motos: dict[int, object] = {}
        self.marcas_pedido: dict[int, object] = {}
        self.gotas: list = []
        self.ultimo_trafico = None

    # ------------------------------------------------------------ coordenadas
    def mundo(self, x: float, y: float, altura: float = 0.0):
        from ursina import Vec3
        return Vec3(x * ESCALA, altura, y * ESCALA)

    # ----------------------------------------------------------------- inicio
    def correr(self) -> None:
        try:
            from ursina import Ursina
        except ImportError:
            raise SystemExit("La vista 3D necesita Ursina: pip install ursina") from None
        app = Ursina(title="Ciudad de domicilios 3D", borderless=False, development_mode=False)
        self.construir()
        app.run()

    def construir(self) -> None:
        """Crea todas las entidades. Separado de correr() para poder probarlo sin ventana."""
        from ursina import (AmbientLight, DirectionalLight, EditorCamera, Entity, Text, Vec3,
                            camera, color, scene, window)

        from ursina.shaders import lit_with_shadows_shader
        Entity.default_shader = lit_with_shadows_shader  # sin esto Ursina no sombrea las caras
        window.color = color.hex("#1b1e24")
        ciudad = self.modelo.ciudad
        n = ciudad.tamano
        centro = self.mundo(n / 2, n / 2)

        # Suelo
        Entity(model="plane", scale=(n * ESCALA + 6, 1, n * ESCALA + 6), position=centro + Vec3(0, -0.02, 0),
               color=color.hex("#2e3a2f"))

        # Calles
        for u, v in ciudad.grafo.edges:
            a, b = self.mundo(*u), self.mundo(*v)
            horizontal = u[1] == v[1]
            self.calles[(u, v)] = Entity(
                model="cube", position=(a + b) / 2,
                # alturas distintas para que en los cruces no se peleen el mismo plano (z-fighting)
                scale=(ESCALA + 0.6, 0.10, 0.6) if horizontal else (0.6, 0.04, ESCALA + 0.6),
                color=color.hex("#4a4e57"))

        # Edificios: en el centro de la manzana vecina a su intersección
        for nodo, tipo in ciudad.edificios.items():
            dx = 0.5 if nodo[0] < n else -0.5
            dy = 0.5 if nodo[1] < n else -0.5
            base = self.mundo(nodo[0] + dx * 0.55, nodo[1] + dy * 0.55)
            if tipo == "restaurante":
                nombre = next(r for r, nd in ciudad.restaurantes.items() if nd == nodo)
                Entity(model="cube", position=base + Vec3(0, 0.5, 0), scale=(1.4, 1.0, 1.4),
                       color=color.hex("#ece6da"))
                Entity(model="cube", position=base + Vec3(0, 1.1, 0), scale=(1.6, 0.2, 1.6),
                       color=color.hex(HEX_RESTAURANTE[nombre]))
                Text(text=nombre, parent=scene, position=base + Vec3(0, 2.6, 0), billboard=True,
                     scale=45, origin=(0, 0), color=color.white)
            else:
                alto = ALTURA[tipo]
                Entity(model="cube", position=base + Vec3(0, alto / 2, 0), scale=(1.1, alto, 1.1),
                       color=color.hex(HEX_EDIFICIO[tipo]))

        # Luz
        sol = DirectionalLight(shadows=True)
        sol.look_at(Vec3(1, -1.5, 0.8))
        AmbientLight(color=color.rgba(0.55, 0.55, 0.6, 1))

        # Lluvia: un grupo fijo de gotas que se reciclan (no crea entidades por cuadro)
        rng = random.Random(0)  # solo decorativo: no toca el rng del modelo
        from ursina.shaders import unlit_shader
        for _ in range(300):
            gota = Entity(model="cube", scale=(0.06, 1.0, 0.06), color=color.rgba(0.8, 0.87, 1, 0.6),
                          shader=unlit_shader,
                          position=Vec3(rng.uniform(0, n * ESCALA), rng.uniform(0, 12), rng.uniform(0, n * ESCALA)),
                          enabled=False)
            self.gotas.append(gota)

        # Textos de interfaz
        self.hud = Text(text="", position=window.top_left + Vec3(0.02, -0.02, 0), origin=(-0.5, 0.5),
                        scale=1.1, background=True)
        self.hud_reps = Text(text="", position=window.top_right + Vec3(-0.02, -0.02, 0), origin=(0.5, 0.5),
                             scale=1.0, background=True)
        Text(text="clic der: girar   rueda: zoom   ESPACIO: pausa   flechas arriba/abajo: velocidad",
             position=window.bottom_left + Vec3(0.02, 0.03, 0), origin=(-0.5, -0.5), scale=0.8,
             color=color.hex("#9aa0aa"))

        # Cámara orbital sobre el centro de la ciudad
        self.camara = EditorCamera(position=centro, rotation=(50, 20, 0))
        self.camara.target_z = -n * ESCALA * 1.6  # EditorCamera interpola el zoom hacia target_z
        camera.z = self.camara.target_z
        camera.fov = 50

        # Controlador: Ursina llama a update() e input() de cada Entity
        vista = self

        class Controlador(Entity):
            def update(self):
                vista.actualizar()

            def input(self, tecla):
                vista.tecla(tecla)

        self.controlador = Controlador()

    # ---------------------------------------------------------------- entrada
    def tecla(self, tecla: str) -> None:
        if tecla == "space":
            self.pausa = not self.pausa
        elif tecla == "up arrow":
            self.ticks_por_cuadro = min(60, self.ticks_por_cuadro * 2)
        elif tecla == "down arrow":
            self.ticks_por_cuadro = max(1, self.ticks_por_cuadro // 2)
        elif tecla == "escape":
            from ursina import application
            application.quit()

    # ----------------------------------------------------------------- cuadro
    def actualizar(self) -> None:
        if not self.pausa:
            for _ in range(self.ticks_por_cuadro):
                if self.modelo.running:
                    self.modelo.step()
        self._trafico()
        self._repartidores()
        self._pedidos()
        self._lluvia()
        self._textos()

    def _trafico(self) -> None:
        from ursina import color
        estado = (self.modelo.eventos.trancon_activo, self.modelo.reloj.dia)
        if estado == self.ultimo_trafico:
            return  # solo se repinta cuando cambia: 220 calles por cuadro sería caro
        self.ultimo_trafico = estado
        for (u, v), calle in self.calles.items():
            factor = self.modelo.ciudad.grafo.edges[u, v]["factor_trancon"]
            calle.color = color.hex("#c0392b") if factor > 1 else color.hex("#4a4e57")

    def _repartidores(self) -> None:
        from ursina import Entity, Vec3, color
        for r in self.modelo.repartidores:
            moto = self.motos.get(r.unique_id)
            if moto is None:
                moto = Entity(model="cube", scale=(0.7, 0.5, 1.2))
                Entity(parent=moto, model="sphere", scale=(0.7, 0.8, 0.4), position=(0, 1.1, -0.1),
                       color=color.hex("#2b2b2b"))  # casco
                Entity(parent=moto, model="cube", scale=(0.9, 0.9, 0.5), position=(0, 1.0, -0.6),
                       color=color.hex("#ff6a3d"))  # caja de domicilios
                self.motos[r.unique_id] = moto
            destino = self.mundo(*r.posicion_continua(), altura=0.3)
            delta = destino - moto.position
            if delta.length() > 0.01:
                moto.rotation_y = math.degrees(math.atan2(delta.x, delta.z))
            moto.position = destino
            moto.color = color.hex(HEX_REPARTIDOR[r.estado])

    def _pedidos(self) -> None:
        from ursina import Entity, color, destroy, time
        activos = {p.id: p for p in self.modelo.pedidos_activos if not p.terminado}
        for pid in list(self.marcas_pedido):
            if pid not in activos:
                destroy(self.marcas_pedido.pop(pid))
        espera_maxima = self.modelo.cfg["pedidos"]["espera_maxima"]
        for pid, pedido in activos.items():
            marca = self.marcas_pedido.get(pid)
            if marca is None:
                marca = Entity(model="sphere", scale=0.45)
                self.marcas_pedido[pid] = marca
            vieja = pedido.edad(self.modelo.reloj.tick) / espera_maxima > 0.66
            marca.color = color.hex("#ff3b3b") if vieja else color.hex(HEX_PEDIDO[pedido.estado])
            marca.position = self.mundo(*pedido.destino, altura=1.6 + 0.2 * math.sin(time.time() * 3 + pid))

    def _lluvia(self) -> None:
        from ursina import time
        lloviendo = self.modelo.eventos.lloviendo
        alto = 12
        for gota in self.gotas:
            gota.enabled = lloviendo
            if lloviendo:
                # módulo y no "+= alto": el primer cuadro trae un dt enorme y la gota
                # caería tan abajo que nunca volvería a aparecer
                gota.y = (gota.y - 20 * min(time.dt, 0.1)) % alto

    def _textos(self) -> None:
        m = self.modelo
        entregados = [p.tiempo_entrega for p in m.pedidos if p.tiempo_entrega is not None]
        promedio = f"{sum(entregados) / len(entregados):.1f} min" if entregados else "-"
        clima = "Lluvia" if m.eventos.lloviendo else "Seco"
        trafico = "Hora pico" if m.eventos.trancon_activo else "Tráfico normal"
        self.hud.text = (f"{m.reloj.texto()}   x{self.ticks_por_cuadro}{'  PAUSA' if self.pausa else ''}\n"
                         f"{clima}  |  {trafico}\n"
                         f"Política: {m.cfg['repartidores']['politica']}\n"
                         f"Creados {m.conteo['creados']}  Entregados {m.conteo['entregados']}  "
                         f"Cancelados {m.conteo['cancelados']}\n"
                         f"Tiempo promedio de entrega: {promedio}")
        self.hud_reps.text = "\n".join(
            f"Repartidor {r.unique_id}: {r.estado.value.replace('_', ' ')}  ${(r.ganancias - r.costos) / 1000:.1f}k"
            for r in m.repartidores)
