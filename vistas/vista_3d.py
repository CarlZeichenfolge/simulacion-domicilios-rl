"""Vista 3D con Ursina: maqueta low poly de la ciudad para la demo.

Igual que la vista 2D, solo LEE el modelo y lo dibuja; nunca cambia su estado.

    python main.py --vista 3d --politica dqn       # una ciudad, manejada por la red entrenada
    python main.py --vista duelo3d                 # ANTES vs DESPUÉS: dos ciudades 3D lado a lado

Controles:
    clic derecho + arrastrar   girar la cámara
    rueda                      zoom
    ESPACIO                    pausa / sigue
    ↑ / ↓                      más o menos minutos simulados por segundo
    ESC                        salir
"""
from __future__ import annotations

import math
import random

from sim.agentes import EstadoRepartidor
from sim.model import CiudadModel
from sim.pedidos import EstadoPedido
from vistas.marcador import texto_marcador
from vistas.velocidad import Velocidad

ESCALA = 3.0  # unidades de mundo por cuadra
ALTO_LLUVIA = 7.0     # altura de la cortina de lluvia (más alta tapa los textos)
VELOCIDAD_GOTA = 18.0  # unidades por segundo real (decorativo, no depende de la simulación)

HEX_EDIFICIO = {"casa": "#c9a27e", "oficina": "#7d93b2", "universidad": "#9a74c4"}
ALTURA = {"casa": 0.8, "oficina": 2.8, "universidad": 1.6}
HEX_RESTAURANTE = {"Pizza": "#e34948", "Hamburguesas": "#eda100", "Sushi": "#1baf7a"}
HEX_REPARTIDOR = {EstadoRepartidor.LIBRE: "#e6e6e6",
                  EstadoRepartidor.YENDO_A_RESTAURANTE: "#ffd400",
                  EstadoRepartidor.YENDO_A_CLIENTE: "#3fb6ff"}
HEX_PEDIDO = {EstadoPedido.CREADO: "#b4b4b4", EstadoPedido.EN_PREPARACION: "#eda100",
              EstadoPedido.LISTO: "#78dc78", EstadoPedido.EN_CAMINO: "#3fb6ff"}
# Duelo: azul = sin entrenar, ámbar = entrenada (los mismos de la vista duelo 2D).
HEX_ANTES, HEX_DESPUES = "#6ea0f0", "#f0b428"
AYUDA = "clic der: girar   rueda: zoom   ESPACIO: pausa   flechas arriba/abajo: velocidad   ESC: salir"


# ------------------------------------------------------------------ escena común
def preparar_escena(centro, distancia: float, giro: float = 20):
    """Luz, sombras, fondo y cámara orbital. Una sola vez por ventana, aunque haya dos ciudades.
    `giro`: cuánto se gira la cámara alrededor del centro (0 = de frente)."""
    from ursina import AmbientLight, DirectionalLight, EditorCamera, Entity, Vec3, camera, color, window
    from ursina.shaders import lit_with_shadows_shader

    Entity.default_shader = lit_with_shadows_shader  # sin esto Ursina no sombrea las caras
    window.color = color.hex("#1b1e24")
    sol = DirectionalLight(shadows=True)
    sol.look_at(Vec3(1, -1.5, 0.8))
    AmbientLight(color=color.rgba(0.55, 0.55, 0.6, 1))
    camara = EditorCamera(position=centro, rotation=(50, giro, 0))
    camara.target_z = -distancia  # EditorCamera interpola el zoom hacia target_z
    camera.z = camara.target_z
    camera.fov = 50
    return camara


class Encuadre:
    """Pone la cámara a la distancia justa para que quepa un ancho dado.

    Por qué no basta con fijar la distancia una vez: Ursina aplica el fov a veces en
    horizontal y a veces en vertical según cuándo termina de crearse la ventana, y con
    la misma distancia las dos ciudades salían recortadas. Aquí se lee el ángulo REAL de
    la lente y se recalcula cuando la ventana cambia de forma (no en cada cuadro, para
    que el zoom con la rueda del mouse se respete).
    """

    def __init__(self, camara, ancho_visible: float):
        self.camara = camara
        self.ancho_visible = ancho_visible
        self.ultima_forma = None

    def revisar(self) -> None:
        from ursina import camera, window
        horizontal = camera.perspective_lens.getFov()[0]
        forma = (tuple(window.size), round(horizontal, 2))
        if forma == self.ultima_forma:
            return
        self.ultima_forma = forma
        distancia = (self.ancho_visible / 2) / math.tan(math.radians(horizontal / 2))
        self.camara.target_z = -distancia
        camera.z = -distancia


def crear_controlador(al_actualizar, al_tecla):
    """Ursina llama a update() e input() de cada Entity: este objeto conecta la vista con ese ciclo."""
    from ursina import Entity

    class Controlador(Entity):
        def update(self):
            al_actualizar()

        def input(self, tecla):
            al_tecla(tecla)

    return Controlador()


def poner_texto(texto_ursina, contenido: str) -> None:
    """Ursina rehace la geometría de las letras cada vez que se asigna .text: solo si cambió."""
    if texto_ursina.text != contenido:
        texto_ursina.text = contenido


def tecla_comun(tecla: str, velocidad: Velocidad, alternar_pausa) -> None:
    if tecla == "space":
        alternar_pausa()
    elif tecla == "up arrow":
        velocidad.subir()
    elif tecla == "down arrow":
        velocidad.bajar()
    elif tecla == "escape":
        from ursina import application
        application.quit()


# --------------------------------------------------------------------- una ciudad
class Vista3D:
    """Una ciudad 3D. `desplazamiento` la corre en el eje x para poner dos en la misma escena."""

    def __init__(self, modelo: CiudadModel, desplazamiento: float = 0.0, titulo: str | None = None,
                 color_titulo: str = "#ffffff", velocidad: Velocidad | None = None):
        self.modelo = modelo
        self.desplazamiento = desplazamiento
        self.titulo = titulo
        self.color_titulo = color_titulo
        self.velocidad = velocidad or Velocidad(modelo.cfg["vistas"])
        self.pausa = False
        self.calles: dict[tuple, tuple] = {}  # (u, v) -> (posición, escala) de cada calle
        self.motos: dict[int, object] = {}
        self.marcas_pedido: dict[int, object] = {}
        self.ultimo_trafico = None
        self._anterior: dict[int, tuple] = {}  # posición de cada moto un minuto antes (para suavizar)
        self.pedido_de_marca: dict[int, object] = {}  # pid -> Pedido, para saber si se canceló al desaparecer
        self.rechazos_vistos: dict[int, int] = {r.unique_id: r.rechazos for r in modelo.repartidores}
        self.cfg_avisos = modelo.cfg["vistas"]["avisos"]
        self.avisos: dict = {}  # clave -> [letrero, momento en que nació, veces]

    # ------------------------------------------------------------ avisos
    def avisar(self, clave, texto: str, posicion, hex_color: str) -> None:
        """Letrero que sube y se desvanece: hace VISIBLES los errores (rechazos y cancelaciones)
        que de otra forma solo se verían como un número en el marcador.

        Una clave por moto: si la misma moto vuelve a rechazar mientras su letrero sigue
        vivo, no se apila otro encima; el mismo letrero se renueva y cuenta ("×3").
        """
        from ursina import Text, color, scene, time
        if clave in self.avisos:
            aviso = self.avisos[clave]
            aviso[2] += 1
            aviso[0].text = f"{texto} ×{aviso[2]}"
            aviso[0].position = posicion
            aviso[1] = time.time()
            return
        if len(self.avisos) >= self.cfg_avisos["max_activos"]:
            return
        letrero = Text(text=texto, parent=scene, position=posicion, billboard=True, scale=40,
                       origin=(0, 0), color=color.hex(hex_color))
        self.avisos[clave] = [letrero, time.time(), 1]

    def _mover_avisos(self) -> None:
        from ursina import destroy, time
        ahora = time.time()
        duracion = self.cfg_avisos["duracion"]
        for clave, (letrero, nacio, _) in list(self.avisos.items()):
            edad = ahora - nacio
            if edad >= duracion:
                destroy(letrero)
                del self.avisos[clave]
                continue
            letrero.y += self.cfg_avisos["subida"] * min(time.dt, self.velocidad.dt_maximo)
            letrero.alpha = 1 - edad / duracion

    # ------------------------------------------------------------ coordenadas
    def mundo(self, x: float, y: float, altura: float = 0.0):
        from ursina import Vec3
        return Vec3(self.desplazamiento + x * ESCALA, altura, y * ESCALA)

    @property
    def lado(self) -> float:
        return self.modelo.ciudad.tamano * ESCALA

    def centro(self):
        n = self.modelo.ciudad.tamano
        return self.mundo(n / 2, n / 2)

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
        """Crea todo sin llamar a app.run(): así se puede probar o capturar sin ventana."""
        cfg = self.modelo.cfg["vista_3d"]
        camara = preparar_escena(self.centro(), self.lado * cfg["encuadre"])
        self.encuadre = Encuadre(camara, self.lado * cfg["encuadre"])
        self.construir_ciudad()
        self.construir_hud()
        self.controlador = crear_controlador(self.actualizar_cuadro, self.tecla)

    def construir_ciudad(self) -> None:
        """Lo que no cambia (suelo, calles, edificios) se FUSIONA en una sola malla.

        Por qué: Ursina dibuja cada Entity por separado. Con ~300 piezas por ciudad
        (y el doble en el duelo) el computador se pasaba el cuadro dibujando cubos
        sueltos; fusionadas son una sola orden de dibujo. Las calles con trancón se
        pintan aparte, encima, cuando cambia el tráfico (4 veces al día).
        """
        from ursina import Entity, Text, Vec3, color, scene
        from ursina.shaders import unlit_shader

        ciudad = self.modelo.ciudad
        n = ciudad.tamano
        estatico = Entity()
        Entity(parent=estatico, model="plane", scale=(self.lado + 6, 1, self.lado + 6),
               position=self.centro() + Vec3(0, -0.02, 0), color=color.hex("#2e3a2f"))
        for u, v in ciudad.grafo.edges:
            a, b = self.mundo(*u), self.mundo(*v)
            horizontal = u[1] == v[1]
            # alturas distintas para que en los cruces no se peleen el mismo plano (z-fighting)
            escala = (ESCALA + 0.6, 0.10, 0.6) if horizontal else (0.6, 0.04, ESCALA + 0.6)
            self.calles[(u, v)] = ((a + b) / 2, escala)
            Entity(parent=estatico, model="cube", position=(a + b) / 2, scale=escala, color=color.hex("#4a4e57"))
        # Edificios: en el centro de la manzana vecina a su intersección
        for nodo, tipo in ciudad.edificios.items():
            dx = 0.5 if nodo[0] < n else -0.5
            dy = 0.5 if nodo[1] < n else -0.5
            base = self.mundo(nodo[0] + dx * 0.55, nodo[1] + dy * 0.55)
            if tipo == "restaurante":
                nombre = next(r for r, nd in ciudad.restaurantes.items() if nd == nodo)
                Entity(parent=estatico, model="cube", position=base + Vec3(0, 0.5, 0), scale=(1.4, 1.0, 1.4),
                       color=color.hex("#ece6da"))
                Entity(parent=estatico, model="cube", position=base + Vec3(0, 1.1, 0), scale=(1.6, 0.2, 1.6),
                       color=color.hex(HEX_RESTAURANTE[nombre]))
                Text(text=nombre, parent=scene, position=base + Vec3(0, 2.6, 0), billboard=True,
                     scale=45, origin=(0, 0), color=color.white)
            else:
                alto = ALTURA[tipo]
                Entity(parent=estatico, model="cube", position=base + Vec3(0, alto / 2, 0), scale=(1.1, alto, 1.1),
                       color=color.hex(HEX_EDIFICIO[tipo]))
        estatico.combine()
        self.trancones = None  # malla de las calles trancadas; se rehace en _trafico()
        if self.titulo:  # letrero en el borde delantero de la ciudad (en el duelo), lejos de los textos de arriba
            Text(text=self.titulo, parent=scene, position=self.mundo(n / 2, -1.6, 0.6), billboard=True,
                 scale=60, origin=(0, 0), color=color.hex(self.color_titulo))
        # Lluvia: 300 gotas fusionadas en una malla, repetidas una altura más arriba. Se mueve la
        # malla entera hacia abajo y se "enrolla" cada ALTO_LLUVIA: parece que llueve sin parar.
        rng = random.Random(0)  # solo decorativo: no toca el rng del modelo
        self.lluvia = Entity(shader=unlit_shader)
        for _ in range(300):
            x, y, alto = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(0, ALTO_LLUVIA)
            for extra in (0, ALTO_LLUVIA):
                Entity(parent=self.lluvia, model="cube", scale=(0.06, 1.0, 0.06), color=color.hex("#cddcff"),
                       position=self.mundo(x, y, alto + extra))
        self.lluvia.combine()
        self.lluvia.enabled = False

    def construir_hud(self) -> None:
        from ursina import Text, Vec3, color, window
        self.hud = Text(text="", position=window.top_left + Vec3(0.02, -0.02, 0), origin=(-0.5, 0.5),
                        scale=1.1, background=True)
        self.hud_reps = Text(text="", position=window.top_right + Vec3(-0.02, -0.02, 0), origin=(0.5, 0.5),
                             scale=1.0, background=True)
        Text(text=AYUDA, position=window.bottom_left + Vec3(0.02, 0.03, 0), origin=(-0.5, -0.5), scale=0.8,
             color=color.hex("#9aa0aa"))

    # ---------------------------------------------------------------- entrada
    def tecla(self, tecla: str) -> None:
        tecla_comun(tecla, self.velocidad, self.alternar_pausa)

    def alternar_pausa(self) -> None:
        self.pausa = not self.pausa

    # ----------------------------------------------------------------- cuadro
    def actualizar_cuadro(self) -> None:
        from ursina import time
        self.encuadre.revisar()
        self.avanzar(0 if self.pausa else self.velocidad.pasos(time.dt))
        self.dibujar(self.velocidad.fraccion)
        self._textos()

    def avanzar(self, minutos: int) -> None:
        """Simula `minutos` minutos. Antes del último guarda dónde estaba cada moto, para
        que dibujar() la mueva suave entre esa posición y la nueva."""
        for i in range(minutos):
            if not self.modelo.running:
                break
            if i == minutos - 1:
                self._anterior = {r.unique_id: r.posicion_continua() for r in self.modelo.repartidores}
            self.modelo.step()

    def dibujar(self, fraccion: float) -> None:
        self._trafico()
        self._repartidores(fraccion)
        self._avisar_rechazos()
        self._pedidos()
        self._lluvia()
        self._mover_avisos()

    def _avisar_rechazos(self) -> None:
        """El contador `rechazos` sube cuando termina una espera habiendo pedidos (ver Repartidor)."""
        for r in self.modelo.repartidores:
            nuevos = r.rechazos - self.rechazos_vistos.get(r.unique_id, r.rechazos)
            self.rechazos_vistos[r.unique_id] = r.rechazos
            if nuevos > 0:
                self.avisar(("rechazo", r.unique_id), "rechazó", self.mundo(*r.posicion_continua(), altura=2.2),
                            "#ffb347")

    def _trafico(self) -> None:
        from ursina import Entity, Vec3, color, destroy
        estado = (self.modelo.eventos.trancon_activo, self.modelo.reloj.dia)
        if estado == self.ultimo_trafico:
            return  # solo se rehace cuando cambia el tráfico
        self.ultimo_trafico = estado
        if self.trancones is not None:
            destroy(self.trancones)
        self.trancones = Entity()
        for (u, v), (posicion, escala) in self.calles.items():
            if self.modelo.ciudad.grafo.edges[u, v]["factor_trancon"] > 1:
                Entity(parent=self.trancones, model="cube", position=posicion + Vec3(0, 0.03, 0),
                       scale=(escala[0], escala[1] + 0.02, escala[2]), color=color.hex("#c0392b"))
        if self.trancones.children:  # fuera de hora pico no hay calles trancadas que fusionar
            self.trancones.combine()

    def _repartidores(self, fraccion: float) -> None:
        from ursina import Entity, color
        for r in self.modelo.repartidores:
            moto = self.motos.get(r.unique_id)
            if moto is None:
                moto = Entity(model="cube", scale=(0.7, 0.5, 1.2))
                Entity(parent=moto, model="sphere", scale=(0.7, 0.8, 0.4), position=(0, 1.1, -0.1),
                       color=color.hex("#2b2b2b"))  # casco
                Entity(parent=moto, model="cube", scale=(0.9, 0.9, 0.5), position=(0, 1.0, -0.6),
                       color=color.hex("#ff6a3d"))  # caja de domicilios
                self.motos[r.unique_id] = moto
            (x1, y1) = r.posicion_continua()
            (x0, y0) = self._anterior.get(r.unique_id, (x1, y1))
            destino = self.mundo(x0 + (x1 - x0) * fraccion, y0 + (y1 - y0) * fraccion, altura=0.3)
            delta = destino - moto.position
            if delta.length() > 0.001:
                moto.rotation_y = math.degrees(math.atan2(delta.x, delta.z))
            moto.position = destino
            moto.color = color.hex(HEX_REPARTIDOR[r.estado])

    def _pedidos(self) -> None:
        from ursina import Entity, color, destroy, time
        activos = {p.id: p for p in self.modelo.pedidos_activos if not p.terminado}
        for pid in list(self.marcas_pedido):
            if pid not in activos:
                destroy(self.marcas_pedido.pop(pid))
                pedido = self.pedido_de_marca.pop(pid)
                if pedido.estado == EstadoPedido.CANCELADO:
                    self.avisar(("cancelado", pid), "cancelado", self.mundo(*pedido.destino, altura=2.4), "#ff3b3b")
        espera_maxima = self.modelo.cfg["pedidos"]["espera_maxima"]
        for pid, pedido in activos.items():
            marca = self.marcas_pedido.get(pid)
            if marca is None:
                marca = Entity(model="sphere", scale=0.45)
                self.marcas_pedido[pid] = marca
                self.pedido_de_marca[pid] = pedido
            vieja = pedido.edad(self.modelo.reloj.tick) / espera_maxima > 0.66
            marca.color = color.hex("#ff3b3b") if vieja else color.hex(HEX_PEDIDO[pedido.estado])
            marca.position = self.mundo(*pedido.destino, altura=1.6 + 0.2 * math.sin(time.time() * 3 + pid))

    def _lluvia(self) -> None:
        from ursina import time
        lloviendo = self.modelo.eventos.lloviendo
        if self.lluvia.enabled != lloviendo:  # prender/apagar solo cuando cambia
            self.lluvia.enabled = lloviendo
        if lloviendo:
            # módulo: la malla baja y vuelve a subir ALTO_LLUVIA, donde está su copia
            self.lluvia.y = -((-self.lluvia.y + VELOCIDAD_GOTA * min(time.dt, 0.1)) % ALTO_LLUVIA)

    def resumen(self) -> str:
        m = self.modelo
        entregados = [p.tiempo_entrega for p in m.pedidos if p.tiempo_entrega is not None]
        promedio = f"{sum(entregados) / len(entregados):.1f} min" if entregados else "-"
        creados = max(1, m.conteo["creados"])
        clima = "Lluvia" if m.eventos.lloviendo else "Seco"
        trafico = "Hora pico" if m.eventos.trancon_activo else "Tráfico normal"
        return (f"{clima}  |  {trafico}\n"
                f"Entregados {m.conteo['entregados']}   Cancelados {m.conteo['cancelados']} "
                f"({m.conteo['cancelados'] / creados:.0%})   Rechazos {sum(r.rechazos for r in m.repartidores)}\n"
                f"Tiempo promedio de entrega: {promedio}\n"
                f"Recompensa de los repartidores: {sum(r.recompensa_acumulada for r in m.repartidores):.1f}")

    def _textos(self) -> None:
        m = self.modelo
        poner_texto(self.hud, f"{m.reloj.texto()}   {self.velocidad.texto()}{'  PAUSA' if self.pausa else ''}\n"
                              f"Política: {m.cfg['repartidores']['politica']}\n" + self.resumen())
        poner_texto(self.hud_reps, "\n".join(
            f"Repartidor {r.unique_id}: {r.estado.value.replace('_', ' ')}  ${(r.ganancias - r.costos) / 1000:.1f}k"
            for r in m.repartidores))


# ------------------------------------------------------------------ dos ciudades
class VistaDuelo3D:
    """Antes vs después en 3D: la misma ciudad dos veces, una al lado de la otra.

    Las dos avanzan el mismo número de minutos en cada cuadro, así el reloj es
    siempre el mismo. Lo único distinto entre ellas es la política (sin entrenar
    vs entrenada); ver crear_duelo() en main.py.
    """

    def __init__(self, antes: CiudadModel, despues: CiudadModel, titulo_antes: str, titulo_despues: str):
        self.velocidad = Velocidad(despues.cfg["vistas"])
        separacion = despues.cfg["vista_duelo_3d"]["separacion"]
        corrimiento = (despues.ciudad.tamano + separacion) * ESCALA
        self.izq = Vista3D(antes, 0.0, titulo_antes, HEX_ANTES, self.velocidad)
        self.der = Vista3D(despues, corrimiento, titulo_despues, HEX_DESPUES, self.velocidad)
        self.titulos = (titulo_antes, titulo_despues)
        self.pausa = False

    def correr(self) -> None:
        try:
            from ursina import Ursina
        except ImportError:
            raise SystemExit("La vista 3D necesita Ursina: pip install ursina") from None
        app = Ursina(title="Antes vs después de entrenar (3D)", borderless=False, development_mode=False)
        self.construir()
        app.run()

    def construir(self) -> None:
        from ursina import Text, Vec3, color, window
        centro = (self.izq.centro() + self.der.centro()) / 2
        ancho_total = self.der.desplazamiento + self.der.lado
        # De frente (giro 0) para que las dos ciudades se vean del mismo tamaño
        ancho_visible = ancho_total * self.der.modelo.cfg["vista_duelo_3d"]["encuadre"]
        self.encuadre = Encuadre(preparar_escena(centro, ancho_visible, giro=0), ancho_visible)
        self.izq.construir_ciudad()
        self.der.construir_ciudad()
        self.hud_izq = Text(text="", position=window.top_left + Vec3(0.02, -0.02, 0), origin=(-0.5, 0.5),
                            scale=1.0, background=True, color=color.hex(HEX_ANTES))
        self.hud_der = Text(text="", position=window.top_right + Vec3(-0.02, -0.02, 0), origin=(0.5, 0.5),
                            scale=1.0, background=True, color=color.hex(HEX_DESPUES))
        self.hud_centro = Text(text="", position=window.top + Vec3(0, -0.02, 0), origin=(0, 0.5),
                               scale=1.3, background=True)
        Text(text=AYUDA + "   M: marcador", position=window.bottom_left + Vec3(0.02, 0.03, 0),
             origin=(-0.5, -0.5), scale=0.8, color=color.hex("#9aa0aa"))
        # Marcador final: fuente monoespaciada para que las columnas queden alineadas, sobre
        # un panel oscuro casi opaco para que la ciudad de atrás no estorbe al leer.
        from ursina import Entity, camera
        from ursina.shaders import unlit_shader
        # unlit: si no, el panel hereda la luz de la escena y se ve gris y transparente
        self.fondo_marcador = Entity(parent=camera.ui, model="quad", color=color.rgba(0.08, 0.09, 0.11, 0.96),
                                     shader=unlit_shader, z=0.1, enabled=False)
        self.panel_marcador = Text(text="", font="VeraMono.ttf", position=Vec3(0, 0, 0), origin=(0, 0),
                                   scale=1.25, enabled=False)
        self.marcador_mostrado = False
        self.controlador = crear_controlador(self.actualizar_cuadro, self.tecla)

    def tecla(self, tecla: str) -> None:
        if tecla == "m":
            self.panel_marcador.enabled = not self.panel_marcador.enabled
            return
        tecla_comun(tecla, self.velocidad, self.alternar_pausa)

    def alternar_pausa(self) -> None:
        self.pausa = not self.pausa

    def actualizar_cuadro(self) -> None:
        from ursina import time
        self.encuadre.revisar()
        minutos = 0 if self.pausa else self.velocidad.pasos(time.dt)
        for vista in (self.izq, self.der):
            vista.avanzar(minutos)
            vista.dibujar(self.velocidad.fraccion)
        poner_texto(self.hud_izq, f"{self.titulos[0]}\n{self.izq.resumen()}")
        poner_texto(self.hud_der, f"{self.titulos[1]}\n{self.der.resumen()}")
        m_izq, m_der = self.izq.modelo, self.der.modelo
        diferencia = (sum(r.recompensa_acumulada for r in m_der.repartidores)
                      - sum(r.recompensa_acumulada for r in m_izq.repartidores))
        poner_texto(self.hud_centro, f"{m_der.reloj.texto()}   {self.velocidad.texto()}{'   PAUSA' if self.pausa else ''}\n"
                                     f"Recompensa: después − antes = {diferencia:+.1f}")
        # Al terminar el día el marcador aparece solo, una vez (con M se oculta o se vuelve a ver).
        if not (m_izq.running or m_der.running) and not self.marcador_mostrado:
            self.marcador_mostrado = True
            self.panel_marcador.enabled = True
        self.fondo_marcador.enabled = self.panel_marcador.enabled
        if self.panel_marcador.enabled:
            poner_texto(self.panel_marcador, texto_marcador(m_izq, m_der, *self.titulos))
            from ursina import window
            cfg = self.der.modelo.cfg["vista_duelo_3d"]
            # Ancho fijo en proporción a la pantalla: Ursina mide mal el ancho de la fuente monoespaciada.
            self.fondo_marcador.scale = (window.aspect_ratio * cfg["ancho_marcador"],
                                         self.panel_marcador.height + cfg["margen_marcador"])
