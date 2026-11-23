"""Vista 3D con Ursina: la ciudad como maqueta low poly que vive un día entero.

Igual que la vista 2D, solo LEE el modelo y lo dibuja; nunca cambia su estado.

    python main.py --vista 3d --politica dqn       # una ciudad, manejada por la red entrenada
    python main.py --vista duelo3d                 # ANTES vs DESPUÉS: dos ciudades 3D lado a lado

Controles:
    clic derecho + arrastrar   girar la cámara
    rueda                      zoom
    C                          cámara girando sola (para mostrar)
    ESPACIO                    pausa / sigue
    ↑ / ↓                      más o menos minutos simulados por segundo
    ESC                        salir

Piezas: vistas/maqueta_3d.py hace la geometría, vistas/ciclo_dia.py los colores
según la hora y vistas/sombreado.py los shaders. Aquí solo se conectan con Ursina.
"""
from __future__ import annotations

import math
import random
import re

from sim.agentes import EstadoRepartidor
from sim.model import CiudadModel
from sim.pedidos import EstadoPedido
from sim.reloj import MINUTOS_POR_HORA
from vistas.ciclo_dia import estado_cielo
from vistas.geometria import Malla, hex_a_color, mezclar
from vistas.maqueta_3d import (ANCHO_CALLE, ESCALA, LADO_LOTE, LARGO_CARRIL, PROFUNDIDAD_DIORAMA, Maqueta,
                               malla_icono, malla_lluvia, malla_moto, malla_nube, malla_pin, malla_trancones)
from vistas.marcador import texto_marcador
from vistas.velocidad import Velocidad, dias_del_episodio, saltar_al_dia_siguiente

MASCARA_SOMBRA = 0b0001  # cámara de la luz del sol (ver ursina.lights.DirectionalLight)
AYUDA = ("clic der: girar   rueda: zoom   C: cámara automática   ESPACIO: pausa   flechas: velocidad   "
         "N: día siguiente   H: habitantes   ESC: salir")


# ------------------------------------------------------------------ utilidades
def a_malla_ursina(malla: Malla):
    from ursina import Mesh
    return Mesh(vertices=malla.vertices, colors=malla.colores, normals=malla.normales,
                uvs=malla.uvs if malla.usa_uv else None, mode="triangle", static=True)


def color_ursina(c):
    from ursina import color
    return color.rgba(*c)


def etiqueta(hex_color: str) -> str:
    """Etiqueta de color para los textos de Ursina: <rgb(r,g,b)>."""
    r, g, b, _ = hex_a_color(hex_color)
    return f"<rgb({r:.3f},{g:.3f},{b:.3f})>"


def abrir_ventana(titulo: str, cfg: dict):
    """Ursina con antialiasing. Por qué antes de crear la ventana: el multisample se pide
    al sistema al abrirla; después ya no se puede activar."""
    from panda3d.core import AntialiasAttrib, loadPrcFileData
    from ursina import Ursina, scene
    muestras = cfg["antialiasing"]
    if muestras:
        loadPrcFileData("", f"framebuffer-multisample 1\nmultisamples {muestras}")
    app = Ursina(title=titulo, borderless=False, development_mode=False)
    if muestras:
        scene.setAntialias(AntialiasAttrib.MMultisample)
    return app


def poner_texto(texto_ursina, contenido: str) -> bool:
    """Ursina rehace la geometría de las letras cada vez que se asigna .text: solo si cambió.

    Se compara con `raw_text` (lo que se asignó) y no con `.text`: ese lo reconstruye
    letra por letra en cada llamada y SIN las etiquetas de color, así que con colores
    nunca era igual y los paneles se rehacían en cada cuadro.
    """
    if getattr(texto_ursina, "raw_text", None) != contenido:
        texto_ursina.text = contenido
        return True
    return False


def tecla_comun(tecla: str, velocidad: Velocidad, alternar_pausa, camara=None) -> None:
    if tecla == "space":
        alternar_pausa()
    elif tecla == "up arrow":
        velocidad.subir()
    elif tecla == "down arrow":
        velocidad.bajar()
    elif tecla == "c" and camara is not None:
        camara.girando = not camara.girando
    elif tecla == "escape":
        from ursina import application
        application.quit()


def crear_controlador(al_actualizar, al_tecla):
    """Ursina llama a update() e input() de cada Entity: este objeto conecta la vista con ese ciclo."""
    from ursina import Entity

    class Controlador(Entity):
        def update(self):
            al_actualizar()

        def input(self, tecla):
            al_tecla(tecla)

    return Controlador()


class Encuadre:
    """Pone la cámara a la distancia justa para que quepa un ancho dado.

    Por qué no basta con fijar la distancia una vez: Ursina aplica el fov a veces en
    horizontal y a veces en vertical según cuándo termina de crearse la ventana, y con
    la misma distancia las ciudades salían recortadas. Aquí se lee el ángulo REAL de
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


# ---------------------------------------------------------------- panel del HUD
class Panel:
    """Texto sobre un fondo redondeado semitransparente, pegado a una esquina.

    Por qué no `Text(background=True)`: Ursina no ajusta ese fondo cuando el texto
    cambia. Aquí el fondo se rehace solo si el tamaño del texto cambió de verdad.
    """

    def __init__(self, cfg_hud: dict, ancla: str, escala: float = 1.0, margen=(0.02, 0.02),
                 titulo: str | None = None, escala_titulo: float = 2.0):
        from ursina import Entity, Text, camera
        from ursina.shaders import unlit_shader
        self.cfg = cfg_hud
        self.ancla = ancla
        self.margen = margen
        self.relleno = cfg_hud["relleno"]
        # El texto siempre se alinea a la izquierda (las columnas se leen mejor); el ancla
        # solo decide en qué esquina de la pantalla va el bloque.
        self.lado_x = {"arriba_izq": -1, "abajo_izq": -1, "arriba_der": 1, "abajo_der": 1, "arriba": 0}[ancla]
        self.lado_y = 1 if ancla.startswith("arriba") else -1
        origen = (-0.5, 0.5)
        self.fondo = Entity(parent=camera.ui, model="quad", shader=unlit_shader, z=0.2,
                            color=color_ursina(hex_a_color(cfg_hud["fondo"], cfg_hud["opacidad"])))
        self.titulo = None
        if titulo is not None:
            self.titulo = Text(text=titulo, parent=camera.ui, origin=origen, scale=escala_titulo,
                               color=color_ursina(hex_a_color(cfg_hud["texto"])))
        self.texto = Text(text="", parent=camera.ui, origin=origen, scale=escala,
                          color=color_ursina(hex_a_color(cfg_hud["texto"])))
        self._tamano = None

    def poner(self, contenido: str, titulo: str | None = None) -> None:
        cambio = poner_texto(self.texto, contenido)
        if titulo is not None and self.titulo is not None:
            cambio = poner_texto(self.titulo, titulo) or cambio
        if cambio or self._tamano is None:
            self._acomodar()

    def _acomodar(self) -> None:
        """Ubica textos y fondo según el ancla; rehace el fondo solo si cambió de tamaño."""
        from ursina import Quad, Vec3, window
        esquina = {"arriba_izq": window.top_left, "arriba_der": window.top_right, "arriba": window.top,
                   "abajo_izq": window.bottom_left, "abajo_der": window.bottom_right}[self.ancla]
        p = self.relleno
        # Text.width/height vienen sin la escala del texto: se multiplican aquí.
        alto_texto = self.texto.height * self.texto.scale_y
        alto_titulo = self.titulo.height * self.titulo.scale_y if self.titulo is not None else 0.0
        ancho = max(self.texto.width * self.texto.scale_x,
                    self.titulo.width * self.titulo.scale_x if self.titulo is not None else 0.0)
        separacion = p * 0.5 if self.titulo is not None else 0.0
        alto = alto_texto + alto_titulo + separacion
        izquierda = {-1: esquina.x + self.margen[0] + p, 1: esquina.x - self.margen[0] - p - ancho,
                     0: esquina.x - ancho / 2}[self.lado_x]
        arriba = esquina.y - self.margen[1] - p if self.lado_y > 0 else esquina.y + self.margen[1] + p + alto
        if self.titulo is not None:
            self.titulo.position = Vec3(izquierda, arriba, 0)
        self.texto.position = Vec3(izquierda, arriba - alto_titulo - separacion, 0)
        tamano = (round(ancho + 2 * p, 3), round(alto + 2 * p, 3))
        self.fondo.position = Vec3(izquierda + ancho / 2, arriba - alto / 2, 0.2)
        if tamano != self._tamano:
            self._tamano = tamano
            self.fondo.model = Quad(radius=self.cfg["radio"] / max(min(tamano), 0.001), aspect=tamano[0] / tamano[1])
            self.fondo.scale = tamano

    @property
    def habilitado(self) -> bool:
        return self.texto.enabled

    @habilitado.setter
    def habilitado(self, valor: bool) -> None:
        for e in (self.fondo, self.texto, self.titulo):
            if e is not None:
                e.enabled = valor


# ------------------------------------------------------- detalle de un habitante
def barra(fraccion: float, ancho: int, hex_lleno: str, hex_vacio: str) -> str:
    """Barra hecha de texto: la fuente no trae bloques, pero "|" seguidos se leen como barra."""
    llenos = round(max(0.0, min(1.0, fraccion)) * ancho)
    return f"{etiqueta(hex_lleno)}{'|' * llenos}{etiqueta(hex_vacio)}{'|' * (ancho - llenos)}"


def color_hambre(cfg: dict, hambre: float, umbral: float) -> tuple:
    """Verde (saciado) -> ámbar (al llegar al umbral) -> rojo (hambre máxima)."""
    c = {k: hex_a_color(v) for k, v in cfg["colores"]["hambre"].items()}
    if hambre <= umbral:
        return mezclar(c["saciado"], c["medio"], hambre / umbral)
    return mezclar(c["medio"], c["hambre"], (hambre - umbral) / max(1.0, 100 - umbral))


_ETIQUETA = re.compile(r"<[^>]*>")


def envolver(linea: str, ancho: int) -> list[str]:
    """Parte una línea larga en varias de a lo sumo `ancho` letras VISIBLES.

    Por qué contar sin etiquetas: "<rgb(0.9,0.9,0.9)>" son 18 letras que no se ven.
    Las continuaciones van con sangría; el color sigue igual porque en Ursina una
    etiqueta vale hasta la siguiente, aunque haya saltos de línea en medio.
    """
    def visible(texto: str) -> int:
        return len(_ETIQUETA.sub("", texto))

    if visible(linea) <= ancho:
        return [linea]
    palabras = linea.split(" ")
    lineas, actual = [], palabras[0]
    for palabra in palabras[1:]:
        if visible(actual) + 1 + visible(palabra) > ancho and visible(actual.strip()) > 0:
            lineas.append(actual)
            actual = "  " + palabra
        else:
            actual += " " + palabra
    lineas.append(actual)
    return lineas


class PanelDetalle:
    """Panel con lo que hace y piensa el agente al que se le dio clic: habitante, repartidor
    o restaurante.

    Por qué un panel y no letreros en la ciudad: son una docena de líneas (qué hace, qué
    comparó, qué viene, qué aprendió); en 3D taparían media ciudad. Uno solo para los tres
    tipos: así nunca quedan dos abiertos tapándose.
    """

    def __init__(self, cfg: dict, margen_y: float):
        self.cfg = cfg
        self.cd = cfg["detalle"]
        self.panel = Panel(cfg["hud"], "arriba_der", escala=self.cd["escala"], margen=(0.02, margen_y), titulo="",
                           escala_titulo=1.25)
        self.panel.habilitado = False
        self.vista = None
        self.agente = None

    def seleccionar(self, vista, agente) -> None:
        """Clic en un agente: lo abre; clic otra vez en el mismo: lo cierra."""
        if self.agente is agente:
            self.cerrar()
            return
        self.vista, self.agente = vista, agente
        self.panel.habilitado = True

    def cerrar(self) -> None:
        self.vista = self.agente = None
        self.panel.habilitado = False

    def actualizar(self) -> None:
        if self.agente is None:
            return
        from sim.agentes.habitante import Habitante
        from sim.agentes.repartidor import Repartidor
        hud = self.cfg["hud"]
        self.s, self.t, self.a = etiqueta(hud["suave"]), etiqueta(hud["texto"]), etiqueta(hud["acento"])
        if isinstance(self.agente, Habitante):
            lineas, titulo = self._habitante(self.agente)
        elif isinstance(self.agente, Repartidor):
            lineas, titulo = self._repartidor(self.agente)
        else:
            lineas, titulo = self._restaurante(self.agente)
        if self.vista.titulo:  # en el duelo: de cuál de las dos ciudades es
            titulo += f"  ·  {etiqueta(self.vista.color_titulo)}{self.vista.titulo.split(':')[0]}"
        lineas += ["", f"{self.s}(clic otra vez en él para cerrar)"]
        ancho = self.cd["ancho_texto"]
        cortas = [corta for linea in "\n".join(lineas).split("\n") for corta in envolver(linea, ancho)]
        self.panel.poner("\n".join(cortas), titulo=titulo)

    def _barra(self, fraccion: float, hex_lleno: str) -> str:
        return barra(fraccion, self.cd["ancho_barra"], hex_lleno, self.cfg["colores"]["barra_vacia"])

    def _filas(self, encabezado: str, filas: list[str], vacio: str) -> list[str]:
        """Una sección con a lo sumo max_filas filas; las demás se cuentan."""
        s, t, maximo = self.s, self.t, self.cd["max_filas"]
        if not filas:
            return [f"{s}{encabezado} {t}{vacio}"]
        lineas = [f"{s}{encabezado}"] + [f"{t}  {f}" for f in filas[:maximo]]
        if len(filas) > maximo:
            lineas.append(f"{s}  y {len(filas) - maximo} más")
        return lineas

    # ------------------------------------------------------------ habitante
    def _habitante(self, h) -> tuple[list[str], str]:
        from vistas import detalle_habitante as d
        cfg = self.cfg
        s, t, a = self.s, self.t, self.a
        umbral = h.cfg["umbral_hambre"]
        frase, clave = d.estado(h)
        tinte_hambre = "#{:02x}{:02x}{:02x}".format(*(int(x * 255) for x in color_hambre(cfg, h.hambre, umbral)[:3]))
        lineas = [
            f"{etiqueta(cfg['colores']['estado_habitante'][clave])}{frase}",
            f"{s}Ahora está {d.describir_actividad(h)}",
            "",
            f"{s}Hambre   {self._barra(h.hambre / 100, tinte_hambre)}  "
            f"{t}{h.hambre:.0f}{s} / 100  (come desde {umbral})",
            f"{s}Energía  {self._barra(h.energia / 100, cfg['colores']['barra_energia'])}  "
            f"{t}{h.energia:.0f}{s} / 100",
            f"{s}Plata    {t}{d.plata(self.vista.modelo, h)}",
            "",
            f"{a}Qué sigue: {t}{d.que_sigue(h)}",
        ]
        if h.pedido is None and h.comida_en_curso is None and h.actividad != "dormir":
            ahora = d.opciones_ahora(h)
            if ahora:
                lineas.append(self._opciones(ahora, "Si decidiera ya:  "))
        lineas += ["", f"{s}Última decisión {d.frase_decision(h.ultima_decision, a)}"]
        if h.ultima_decision and h.ultima_decision["opciones"]:
            lineas.append(self._opciones(h.ultima_decision["opciones"], "  "))
        aprendido = d.lo_que_aprendio(h)
        if aprendido:
            favorito = max(aprendido, key=lambda x: x[1])[0]
            lineas.append(f"{s}Espera de cada restaurante (veces que le pidió): " + "  ".join(
                f"{a if n == favorito else t}{n} {v:+.1f}{s} ({veces})" for n, v, veces in aprendido))
        lineas.append(f"{s}Hoy: {t}{'   '.join(d.comidas_de_hoy(self.vista.modelo, h)) or 'nada todavía'}")
        modelo = self.vista.modelo
        lineas += self._pedidos(d.resumen_pedidos(modelo, h), d.lista_pedidos(modelo, h), "no ha pedido domicilios")
        return lineas, f"{a}Habitante {h.unique_id}{s}  ·  {h.tipo}"

    def _pedidos(self, resumen: str, filas: list[str], vacio: str) -> list[str]:
        """Cuántos pedidos lleva y los últimos (max_pedidos), el más reciente primero."""
        s, t, a, maximo = self.s, self.t, self.a, self.cd["max_pedidos"]
        if not filas:
            return ["", f"{a}Pedidos: {t}{vacio}"]
        lineas = ["", f"{a}Pedidos: {t}{resumen}"] + [f"{s}  {f}" for f in filas[:maximo]]
        if len(filas) > maximo:
            lineas.append(f"{s}  y {len(filas) - maximo} más antes")
        return lineas

    def _opciones(self, opciones: dict, prefijo: str) -> str:
        """Las utilidades en fila; la que gana, resaltada."""
        from vistas.detalle_habitante import NOMBRE_CORTO
        mejor = max(opciones, key=opciones.get)
        partes = [f"{self.a if k == mejor else self.t}{NOMBRE_CORTO[k]} {v:+.1f}" for k, v in opciones.items()]
        return f"{self.s}{prefijo}" + "   ".join(partes)

    # ------------------------------------------------------------ repartidor
    def _repartidor(self, r) -> tuple[list[str], str]:
        from vistas import detalle_repartidor as d
        s, t, a = self.s, self.t, self.a
        frase, clave = d.estado(r)
        lineas = [f"{etiqueta(self.cfg['colores']['repartidor'][clave])}{frase}"]
        lineas += [f"{s}{linea}" for linea in d.trayecto(r)]
        lineas += [f"{a}Qué sigue: {t}{d.que_sigue(r)}", ""]
        explorando = f"  {a}(exploró: no fue la de mayor Q)" if d.exploro(r) else ""
        lineas.append(f"{s}Última decisión {d.encabezado_decision(r)}{explorando}")
        filas, sobran = d.opciones(r, self.cd["max_opciones"])
        if filas:
            lineas.append(f"{s}  (cuadras: hasta el restaurante + hasta el cliente)")
        for f in filas:
            marca = f"{a}> " if f["escogida"] else f"{t}  "
            valor = "" if f["valor"] is None else f"   {a if f['escogida'] else s}Q {f['valor']:+.2f}"
            lineas.append(f"  {marca}{f['texto']}{valor}")
        if sobran:
            lineas.append(f"{s}    y {sobran} más")
        if any(f["valor"] is not None for f in filas):
            lineas.append(f"{s}  {d.QUE_ES_Q}")
        lineas += [f"{s}Último resultado {t}{d.frase_resultado(r)}", "", f"{a}Plata: {t}{d.plata(r)}"]
        lineas += self._pedidos(d.balance(r), d.lista_pedidos(r), "todavía no ha terminado ninguno")
        return lineas, f"{a}Repartidor {r.unique_id}{s}  ·  política {d.politica(r)}"

    # ----------------------------------------------------------- restaurante
    def _restaurante(self, rest) -> tuple[list[str], str]:
        from vistas import detalle_restaurante as d
        s, t, a = self.s, self.t, self.a
        ocupados, fogones = d.ocupacion(rest)
        lineas = [f"{s}Fogones  {self._barra(ocupados / fogones, self.cfg['colores']['barra_fogones'])}  "
                  f"{t}{ocupados}{s}/{fogones} ocupados · {t}{len(d.esperando_fogon(rest))}{s} en fila",
                  f"{s}{d.ficha(rest)}", f"{s}{d.espera_nueva(rest)}", ""]
        lineas += self._filas("En el fogón:", d.en_cocina(rest), "nada")
        lineas += self._filas("En la fila (el más viejo entra primero):", d.en_cola(rest), "nadie")
        lineas += self._filas("Listos, esperando moto:", d.listos(rest), "ninguno")
        lineas += ["", f"{s}{d.hoy(rest)}"]
        opinion = d.opinion(rest)
        if opinion:
            lineas.append(f"{s}{opinion}")
        lineas += ["", f"{a}Plata: {t}{d.plata(rest)}"]
        lineas += self._pedidos(d.resumen_pedidos(rest), d.lista_pedidos(rest), "ninguno terminado todavía")
        color = self.cfg["colores"]["restaurantes"][rest.nombre]
        return lineas, f"{etiqueta(color)}{rest.nombre}{s}  ·  restaurante"


# ------------------------------------------------------------- cielo, sol y clima
class Ambiente:
    """Lo que es UNO por ventana aunque haya dos ciudades: sol, cielo, nubes, niebla y cámara.

    Cada cuadro recibe la hora y si llueve, y reparte los colores a todos los shaders.
    """

    def __init__(self, cfg: dict, limites: tuple, alto_maximo: float, rng: random.Random, ancho_visible: float):
        from ursina import DirectionalLight, EditorCamera, Entity, Vec3, camera, scene
        from vistas import sombreado
        self.cfg = cfg
        self.lluvia = 0.0
        x0, x1, z0, z1 = limites
        self.centro = Vec3((x0 + x1) / 2, 0, (z0 + z1) / 2)
        self.lado = max(x1 - x0, z1 - z0)

        resolucion = cfg["sombras"]["resolucion"]
        self.sol = DirectionalLight(shadow_map_resolution=(resolucion, resolucion), shadows=True)
        # La sombra se calcula sobre una caja que envuelve la maqueta (no sobre toda la
        # escena: el cielo y las nubes la agrandarían y las sombras saldrían borrosas).
        self.caja_sombra = Entity(model="cube", position=self.centro + Vec3(0, alto_maximo / 2 - 0.5, 0),
                                  scale=(x1 - x0, alto_maximo + PROFUNDIDAD_DIORAMA, z1 - z0), visible=False)
        self.sol.update_bounds(self.caja_sombra)
        self._luz_dir = None
        scene.set_shader_input("sombra_sesgo", cfg["sombras"]["sesgo"])
        scene.set_shader_input("sombra_texel", 1 / resolucion)

        camera.fov = 40
        camera.clip_plane_far = self.lado * 40
        c = cfg["camara"]
        self.camara = EditorCamera(position=self.centro, rotation=(c["inclinacion"], c["giro"], 0))
        self.camara.girando = False
        self.encuadre = Encuadre(self.camara, ancho_visible)

        self.cielo = Entity(model="sphere", scale=self.lado * 20, shader=sombreado.cielo, double_sided=True)
        self.cielo.setBin("background", 0)
        self.cielo.setDepthWrite(False)
        self.cielo.hide(MASCARA_SOMBRA)

        cn = cfg["nubes"]
        color_nube = hex_a_color(cfg["colores"]["nube"])
        self.nubes = []
        # Las nubes van en un arco DETRÁS de la maqueta (lado +z, lejos de la cámara): encima
        # taparían la ciudad y adelante se meterían en las esquinas de la pantalla.
        radio = self.lado / 2
        desde, hasta = (math.radians(a) for a in cn["arco"])
        for i in range(cn["cantidad"]):
            angulo = desde + (hasta - desde) * (i + rng.uniform(0.1, 0.9)) / cn["cantidad"]
            distancia = radio * rng.uniform(*cn["distancia"])
            nube = Entity(model=a_malla_ursina(malla_nube(rng, color_nube)), shader=sombreado.iluminado,
                          position=(self.centro.x + math.cos(angulo) * distancia, rng.uniform(*cn["altura"]),
                                    self.centro.z + math.sin(angulo) * distancia),
                          scale=rng.uniform(*cn["escala"]), rotation_y=rng.uniform(0, 360))
            nube.angulo, nube.distancia = angulo, distancia
            nube.hide(MASCARA_SOMBRA)
            self.nubes.append(nube)
        self.limites = limites
        self.estado = None
        # Relojes propios desde que abrió la ventana. Por qué no time.time(): son ~1.7e9 s y
        # en un float de la tarjeta de video (7 cifras) el titilar y los carros se congelan.
        self.reloj_real = 0.0
        self.reloj_trafico = 0.0   # este se detiene en pausa: los carros también se quedan quietos
        ct = cfg["trafico"]
        scene.set_shader_input("velocidad_trafico", ct["velocidad"])
        scene.set_shader_input("periodo_trafico", ct["periodo"])
        scene.set_shader_input("largo_carril", LARGO_CARRIL)
        scene.set_shader_input("tiempo_trafico", 0.0)

    def actualizar(self, minuto_del_dia: float, lloviendo: bool, pausa: bool = False) -> dict:
        from ursina import Vec3, camera, scene, time, window
        dt = min(time.dt, 0.1)
        self.reloj_real += dt
        if not pausa:
            self.reloj_trafico += dt
            scene.set_shader_input("tiempo_trafico", self.reloj_trafico)
        self.encuadre.revisar()
        if self.camara.girando:
            self.camara.rotation_y += self.cfg["camara"]["giro_auto"] * dt
        # La lluvia sube y baja poco a poco: el cielo se nubla, no cambia de golpe.
        paso = self.cfg["lluvia"]["transicion"] * dt
        objetivo = 1.0 if lloviendo else 0.0
        self.lluvia += max(-paso, min(paso, objetivo - self.lluvia))

        e = estado_cielo(self.cfg["cielo"], minuto_del_dia / MINUTOS_POR_HORA, self.lluvia)
        self.estado = e
        expo = self.cfg["exposicion"]
        luz = [c * e["intensidad"] * expo for c in e["sol"][:3]]
        self.sol.color = color_ursina((*luz, 1))
        if self._luz_dir is None or max(abs(a - b) for a, b in zip(e["luz_dir"], self._luz_dir)) > 1e-3:
            self._luz_dir = e["luz_dir"]
            self.sol.look_at(self.sol.world_position - Vec3(*e["luz_dir"]))
        rgb = lambda c, f=1.0: Vec3(c[0] * f, c[1] * f, c[2] * f)  # noqa: E731
        scene.set_shader_input("luz_cielo", rgb(e["cielo"], expo))
        scene.set_shader_input("luz_suelo", rgb(e["suelo"], expo))
        n = self.cfg["niebla"]
        scene.set_shader_input("niebla_color", color_ursina((*e["horizonte"][:3], n["intensidad"])))
        scene.set_shader_input("niebla_inicio", self.lado * n["inicio"])
        scene.set_shader_input("niebla_fin", self.lado * n["fin"])
        scene.set_shader_input("camara_mundo", camera.world_position)
        scene.set_shader_input("cielo_cenit", rgb(e["cenit"]))
        scene.set_shader_input("cielo_horizonte", rgb(e["horizonte"]))
        # Debajo del horizonte no hay suelo (la maqueta flota): un degradado suave del mismo cielo.
        scene.set_shader_input("cielo_bajo", rgb(mezclar(e["horizonte"], e["cenit"], self.cfg["fondo_bajo"])))
        scene.set_shader_input("sol_dir", Vec3(*e["sol_dir"]))
        scene.set_shader_input("sol_color", rgb(e["sol"]))
        scene.set_shader_input("sol_visible", e["sol_visible"])
        scene.set_shader_input("luna_dir", Vec3(*e["luna_dir"]))
        scene.set_shader_input("noche", e["noche"])
        scene.set_shader_input("tiempo", self.reloj_real)
        window.color = color_ursina(e["horizonte"])
        self.cielo.position = camera.world_position

        cn = self.cfg["nubes"]
        gris = 1 - cn["gris_lluvia"] * self.lluvia
        desde, hasta = (math.radians(a) for a in cn["arco"])
        for nube in self.nubes:  # avanzan despacio por el arco y vuelven a empezar
            nube.angulo += cn["velocidad"] * dt / nube.distancia
            if nube.angulo > hasta:
                nube.angulo = desde
            nube.x = self.centro.x + math.cos(nube.angulo) * nube.distancia
            nube.z = self.centro.z + math.sin(nube.angulo) * nube.distancia
            nube.color = color_ursina((gris, gris, gris, 1))
        return e


# --------------------------------------------------------------------- una ciudad
class Vista3D:
    """Una ciudad 3D. `desplazamiento` la corre en el eje x para poner dos en la misma escena."""

    def __init__(self, modelo: CiudadModel, desplazamiento: float = 0.0, titulo: str | None = None,
                 color_titulo: str = "#ffffff", velocidad: Velocidad | None = None):
        self.modelo = modelo
        self.cfg = modelo.cfg["vista_3d"]
        self.desplazamiento = desplazamiento
        self.titulo = titulo
        self.color_titulo = color_titulo
        self.velocidad = velocidad or Velocidad(modelo.cfg["vistas"])
        self.pausa = False
        # Solo adorno (árboles, colores de las casas): no toca el azar del modelo. Depende también
        # del mapa, para que cada mapa se vea distinto y no solo con los edificios corridos.
        self.semilla_adorno = f"{self.cfg['semilla_decorado']}-{modelo.cfg['ciudad']['semilla_mapa']}"
        self.rng = random.Random(self.semilla_adorno)
        self.maqueta = Maqueta(modelo.ciudad, self.cfg, self.rng, desplazamiento)
        self.col = self.maqueta.col
        self.motos: dict[int, dict] = {}
        self.marcas_pedido: dict[int, dict] = {}
        self.pedido_de_marca: dict[int, object] = {}  # pid -> Pedido, para saber si se canceló al desaparecer
        self.ultimo_trafico = None
        self.trancones = None
        self._anterior: dict[int, tuple] = {}  # posición de cada moto un minuto antes (para suavizar)
        self.rechazos_vistos: dict[int, int] = {r.unique_id: r.rechazos for r in modelo.repartidores}
        self.cfg_avisos = modelo.cfg["vistas"]["avisos"]
        self.avisos: dict = {}  # clave -> [letrero, momento en que nació, veces]
        self.ambiente: Ambiente | None = None
        self.iconos_habitante: dict[int, dict] = {}
        self.mostrar_habitantes = True
        self.panel_detalle: PanelDetalle | None = None  # lo pone quien arma el HUD (una o dos ciudades)
        self.linea_ruta = None
        self._clave_ruta = None

    # ------------------------------------------------------------ coordenadas
    def mundo(self, x: float, y: float, altura: float = 0.0):
        from ursina import Vec3
        return Vec3(*self.maqueta.mundo(x, y, altura))

    @property
    def lado(self) -> float:
        return self.maqueta.lado

    def centro(self):
        n = self.modelo.ciudad.tamano
        return self.mundo(n / 2, n / 2)

    # ----------------------------------------------------------------- inicio
    def correr(self) -> None:
        try:
            app = abrir_ventana("Ciudad de domicilios 3D", self.cfg)
        except ImportError:
            raise SystemExit("La vista 3D necesita Ursina: pip install ursina") from None
        self.construir()
        app.run()

    def construir(self) -> None:
        """Crea todo sin llamar a app.run(): así se puede probar o capturar sin ventana."""
        self.ambiente = Ambiente(self.cfg, self.maqueta.limites(), self.maqueta.alto_maximo, self.rng,
                                 self.lado * self.cfg["encuadre"])
        self.construir_ciudad()
        self.construir_hud()
        self.controlador = crear_controlador(self.actualizar_cuadro, self.tecla)

    def construir_ciudad(self) -> None:
        """Cada malla de la maqueta es UNA Entity: miles de piezas, pocas órdenes de dibujo."""
        from panda3d.core import ColorBlendAttrib
        from ursina import Entity, Text, Vec3, color, scene
        from vistas import sombreado

        q = self.maqueta
        Entity(model=a_malla_ursina(q.estatico), shader=sombreado.iluminado)
        self.ventanas_on = Entity(model=a_malla_ursina(q.ventanas_encendibles), shader=sombreado.emisivo)
        self.ventanas_off = Entity(model=a_malla_ursina(q.ventanas_apagadas), shader=sombreado.emisivo)
        self.focos = Entity(model=a_malla_ursina(q.focos), shader=sombreado.emisivo)
        # Halos: se SUMAN a lo que hay debajo (mezcla aditiva), como la luz de verdad.
        self.halos = Entity(model=a_malla_ursina(q.halos), shader=sombreado.emisivo, texture="radial_gradient")
        self.halos.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd, ColorBlendAttrib.OIncomingAlpha,
                                                   ColorBlendAttrib.OOne))
        self.halos.setDepthWrite(False)
        self.halos.setBin("transparent", 10)
        for e in (self.ventanas_on, self.ventanas_off, self.focos, self.halos):
            e.hide(MASCARA_SOMBRA)

        # Restaurantes: ícono que gira, anillo de su color en el piso y el nombre.
        self.iconos = []
        ci = self.cfg["icono"]
        for nombre, datos in q.restaurantes.items():
            x, y, z = datos["centro"]
            icono = Entity(model=a_malla_ursina(malla_icono(nombre, self.col)), shader=sombreado.iluminado,
                           position=(x, y + ci["altura"], z), scale=ci["escala"])
            # Zona de clic invisible del suelo hasta el ícono: más fácil de atinar que el ícono que gira.
            alto_clic = y + ci["altura"] + ci["escala"]
            clic = Entity(position=(x, alto_clic / 2, z), scale=(LADO_LOTE, alto_clic, LADO_LOTE), collider="box")
            restaurante = self.modelo.restaurantes[nombre]
            clic.on_click = lambda rest=restaurante: self.panel_detalle and self.panel_detalle.seleccionar(self, rest)
            self.iconos.append({"icono": icono, "clic": clic, "restaurante": restaurante, "escala": ci["escala"]})
            anillo = Malla()
            anillo.anillo_suelo((x, 0.06, z), 0.62, 0.72, 24, self.col["restaurantes"][nombre])
            e = Entity(model=a_malla_ursina(anillo), shader=sombreado.emisivo)
            e.hide(MASCARA_SOMBRA)
            letrero = Text(text=nombre, parent=scene, position=(x, y + ci["altura"] + 0.8, z), billboard=True,
                           scale=ci["letrero"], origin=(0, 0), color=color.white)
            letrero.create_background(padding=(0.025, 0.012), radius=0.008,
                                      color=color_ursina(hex_a_color(self.cfg["colores"]["restaurantes"][nombre], 0.92)))
            letrero.hide(MASCARA_SOMBRA)

        n = self.modelo.ciudad.tamano
        if self.titulo:  # letrero en el borde delantero de la ciudad (en el duelo)
            t = Text(text=self.titulo, parent=scene, position=self.mundo(n / 2, -self.cfg["borde"] * 0.75, 1.0),
                     billboard=True, scale=self.cfg["icono"]["letrero"] * 1.6, origin=(0, 0), color=color.white)
            t.create_background(padding=(0.03, 0.015), radius=0.01,
                                color=color_ursina(hex_a_color(self.color_titulo, 0.92)))
            t.hide(MASCARA_SOMBRA)

        cl = self.cfg["lluvia"]
        self.lluvia = Entity(model=a_malla_ursina(malla_lluvia(self.rng, q.limites(), cl["alto"], cl["gotas"],
                                                               hex_a_color(self.col_hex("gota")))),
                             shader=sombreado.emisivo, double_sided=True)
        self.lluvia.setTransparency(True)
        self.lluvia.setDepthWrite(False)
        self.lluvia.hide(MASCARA_SOMBRA)
        self.lluvia.enabled = False
        self.malla_pin = malla_pin()
        self.mallas_moto = malla_moto(self.col)
        self._sombreado = sombreado
        self._vec3 = Vec3

    def col_hex(self, nombre: str) -> str:
        return self.cfg["colores"][nombre]

    def construir_hud(self) -> None:
        h = self.cfg["hud"]
        self.panel_izq = Panel(h, "arriba_izq", escala=1.0, titulo="", escala_titulo=2.4)
        self.panel_der = Panel(h, "arriba_der", escala=0.95)
        self.panel_leyenda = Panel(h, "abajo_izq", escala=0.8)
        self.panel_leyenda.poner(self.leyenda())
        self.panel_detalle = PanelDetalle(self.cfg, self.cfg["detalle"]["margen_panel"])

    def leyenda(self) -> str:
        h, r, p = self.cfg["hud"], self.cfg["colores"]["repartidor"], self.cfg["colores"]["pedido"]
        s = etiqueta(h["suave"])
        return (f"{s}Aro y luz de la moto:  {etiqueta(r['libre'])}libre   {etiqueta(r['yendo_a_restaurante'])}va al "
                f"restaurante   {etiqueta(r['yendo_a_cliente'])}va al cliente {s}(con la comida flotando encima)\n"
                f"{s}Pedidos:  {etiqueta(p['creado'])}creado   {etiqueta(p['en_preparacion'])}en cocina   "
                f"{etiqueta(p['listo'])}listo   {etiqueta(p['en_camino'])}en camino   "
                f"{etiqueta(p['urgente'])}a punto de cancelarse   {s}Rojo en la calle: trancón\n"
                f"{s}Habitantes: el círculo es su hambre ({etiqueta(self.col_hex('hambre')['saciado'])}saciado"
                f"{s} a {etiqueta(self.col_hex('hambre')['hambre'])}con hambre{s})\n"
                f"{etiqueta(h['acento'])}Clic en un habitante, una moto o un restaurante{s}: qué hace, qué decide "
                f"y por qué\n"
                f"{s}{AYUDA}")

    # ---------------------------------------------------------------- entrada
    def tecla(self, tecla: str) -> None:
        if tecla == "n":
            self.saltar_dia()
            return
        if tecla == "h":
            self.alternar_habitantes()
            return
        tecla_comun(tecla, self.velocidad, self.alternar_pausa, self.ambiente.camara)

    def alternar_pausa(self) -> None:
        self.pausa = not self.pausa

    def alternar_habitantes(self) -> None:
        self.mostrar_habitantes = not self.mostrar_habitantes
        from sim.agentes.habitante import Habitante
        if (not self.mostrar_habitantes and self.panel_detalle is not None
                and isinstance(self.panel_detalle.agente, Habitante)):
            self.panel_detalle.cerrar()

    def saltar_dia(self) -> None:
        """Tecla N: a la mañana siguiente sin dibujar la noche."""
        saltar_al_dia_siguiente(self.modelo, self.modelo.cfg["vistas"]["hora_al_saltar"])
        self._anterior = {}  # las motos aparecen donde quedaron, sin "deslizarse" desde ayer

    # ----------------------------------------------------------------- cuadro
    def actualizar_cuadro(self) -> None:
        from ursina import time
        self.avanzar(0 if self.pausa else self.velocidad.pasos(time.dt))
        estado = self.ambiente.actualizar(self.minuto_suave(), self.modelo.eventos.lloviendo, self.pausa)
        self.dibujar(self.velocidad.fraccion, estado, self.ambiente.lluvia)
        self._textos()
        self.panel_detalle.actualizar()

    def minuto_suave(self) -> float:
        """El minuto del día con la fracción del siguiente: el sol se mueve sin saltos."""
        return self.modelo.reloj.minuto_del_dia + (0 if self.pausa else self.velocidad.fraccion)

    def avanzar(self, minutos: int) -> None:
        """Simula `minutos` minutos. Antes del último guarda dónde estaba cada moto, para
        que dibujar() la mueva suave entre esa posición y la nueva."""
        for i in range(minutos):
            if not self.modelo.running:
                break
            if i == minutos - 1:
                self._anterior = {r.unique_id: r.posicion_continua() for r in self.modelo.repartidores}
            self.modelo.step()

    def dibujar(self, fraccion: float, estado: dict, lluvia: float) -> None:
        self._luces(estado["noche"])
        self._trafico()
        if self.trancones:  # farolas y stops: tenues de día, encendidos de noche
            tenue = self.cfg["trafico"]["luces_dia"]
            brillo = tenue + (1 - tenue) * estado["noche"]
            self.trancones[2].color = color_ursina((brillo, brillo, brillo, 1))
        self._repartidores(fraccion, estado["noche"])
        self._ruta_elegida()
        self._avisar_rechazos()
        self._pedidos()
        self._lluvia(lluvia)
        self._mover_avisos()
        self._iconos()
        self._habitantes()

    def _luces(self, noche: float) -> None:
        """De día las ventanas son vidrio; de noche unas brillan y otras quedan oscuras."""
        c = self.col
        self.ventanas_on.color = color_ursina(mezclar(c["vidrio_dia"], c["ventana_noche"], noche))
        self.ventanas_off.color = color_ursina(mezclar(c["vidrio_dia"], c["ventana_apagada"], noche))
        self.focos.color = color_ursina(mezclar(c["foco_dia"], c["foco_noche"], noche))
        r, g, b, _ = c["halo"]
        self.halos.color = color_ursina((r, g, b, noche))
        self.halos.enabled = noche > 0.02

    def _elegido(self):
        """El agente cuyo panel está abierto, si es de ESTA ciudad (en el duelo hay dos)."""
        p = self.panel_detalle
        return p.agente if p is not None and p.vista is self else None

    def _iconos(self) -> None:
        """Giran siempre; el del restaurante elegido (o bajo el mouse) se agranda y late."""
        from ursina import mouse, time
        elegido = self._elegido()
        agrandar = self.cfg["detalle"]["agrandar"]
        for datos in self.iconos:
            icono = datos["icono"]
            icono.rotation_y += self.cfg["icono"]["giro"] * min(time.dt, 0.1)
            seleccionado = datos["restaurante"] is elegido
            grande = seleccionado or mouse.hovered_entity is datos["clic"]
            pulso = 1 + 0.08 * math.sin(time.time() * 6) if seleccionado else 1
            escala = datos["escala"] * (agrandar if grande else 1) * pulso
            if escala != icono.scale_x:
                icono.scale = escala

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
        letrero = Text(text=texto, parent=scene, position=posicion, billboard=True, scale=34,
                       origin=(0, 0), color=color.hex(hex_color))
        letrero.hide(MASCARA_SOMBRA)
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

    def _avisar_rechazos(self) -> None:
        """El contador `rechazos` sube cuando termina una espera habiendo pedidos (ver Repartidor)."""
        for r in self.modelo.repartidores:
            nuevos = r.rechazos - self.rechazos_vistos.get(r.unique_id, r.rechazos)
            self.rechazos_vistos[r.unique_id] = r.rechazos
            if nuevos > 0:
                self.avisar(("rechazo", r.unique_id), "rechazó", self.mundo(*r.posicion_continua(), altura=2.2),
                            self.cfg["hud"]["acento"])

    # ------------------------------------------------------------ dinámico
    def _trafico(self) -> None:
        from ursina import Entity, destroy
        estado = (self.modelo.eventos.trancon_activo, self.modelo.reloj.dia)
        if estado == self.ultimo_trafico:
            return  # solo se rehace cuando cambia el tráfico
        self.ultimo_trafico = estado
        for entidad in self.trancones or ():
            destroy(entidad)
        self.trancones = None
        grafo = self.modelo.ciudad.grafo
        trancadas = [(u, v) for u, v in grafo.edges if grafo.edges[u, v]["factor_trancon"] > 1]
        if trancadas:  # fuera de hora pico no hay calles trancadas
            franja, carros, luces = malla_trancones(self.maqueta, trancadas,
                                                    random.Random(self.semilla_adorno), self.cfg["trafico"])
            s = self._sombreado
            e_luces = Entity(model=a_malla_ursina(luces), shader=s.luces_trafico)
            e_luces.hide(MASCARA_SOMBRA)
            self.trancones = (Entity(model=a_malla_ursina(franja), shader=s.iluminado),
                              Entity(model=a_malla_ursina(carros), shader=s.trafico), e_luces)

    def _nueva_moto(self, r) -> dict:
        """La moto, y una caja invisible un poco más grande que ella para poder darle clic."""
        from ursina import Entity
        cuerpo, baliza, faro = self.mallas_moto
        raiz = Entity(scale=self.cfg["escala_moto"])
        ancho, alto, largo = self.cfg["detalle"]["caja_moto"]
        clic = Entity(parent=raiz, y=alto / 2, scale=(ancho, alto, largo), collider="box")
        clic.on_click = lambda: self.panel_detalle and self.panel_detalle.seleccionar(self, r)
        Entity(parent=raiz, model=a_malla_ursina(cuerpo), shader=self._sombreado.iluminado)
        e_baliza = Entity(parent=raiz, model=a_malla_ursina(baliza), shader=self._sombreado.emisivo)
        e_faro = Entity(parent=raiz, model=a_malla_ursina(faro), shader=self._sombreado.emisivo)
        aro = Malla()
        aro.anillo_suelo((0, 0.02, 0), 0.34, 0.42, 20, (1, 1, 1, 1))
        e_aro = Entity(parent=raiz, model=a_malla_ursina(aro), shader=self._sombreado.emisivo)
        e_aro.hide(MASCARA_SOMBRA)
        e_faro.hide(MASCARA_SOMBRA)
        return {"raiz": raiz, "clic": clic, "baliza": e_baliza, "faro": e_faro, "aro": e_aro, "rumbo": None,
                "lateral": (0.0, 0.0), "escala": self.cfg["escala_moto"]}

    def _repartidores(self, fraccion: float, noche: float) -> None:
        """Cada moto va por su carril derecho y gira suave hacia donde va; la baliza sobre la
        mochila y el aro en el piso dicen su estado. La elegida (o bajo el mouse) se agranda."""
        from ursina import mouse, time
        dt = min(time.dt, 0.1)
        colores = self.col["repartidor"]
        elegido = self._elegido()
        for r in self.modelo.repartidores:
            moto = self.motos.get(r.unique_id)
            if moto is None:
                moto = self.motos[r.unique_id] = self._nueva_moto(r)
            seleccionada = r is elegido
            grande = seleccionada or mouse.hovered_entity is moto["clic"]
            pulso = 1 + 0.08 * math.sin(time.time() * 6) if seleccionada else 1
            escala = self.cfg["escala_moto"] * (self.cfg["detalle"]["agrandar"] if grande else 1) * pulso
            if escala != moto["escala"]:
                moto["escala"] = escala
                moto["raiz"].scale = escala
            (x1, y1) = r.posicion_continua()
            (x0, y0) = self._anterior.get(r.unique_id, (x1, y1))
            x, y = x0 + (x1 - x0) * fraccion, y0 + (y1 - y0) * fraccion
            dx, dy = x1 - x0, y1 - y0
            if math.hypot(dx, dy) > 1e-4:
                objetivo = math.degrees(math.atan2(dx, dy))
                largo = math.hypot(dx, dy)
                moto["lateral"] = (dy / largo, -dx / largo)  # a la derecha del rumbo
                if moto["rumbo"] is None:
                    moto["rumbo"] = objetivo
                diferencia = (objetivo - moto["rumbo"] + 180) % 360 - 180
                moto["rumbo"] += diferencia * min(1.0, self.cfg["suavizado_giro"] * dt)
            raiz = moto["raiz"]
            lx, lz = moto["lateral"]
            px, _, pz = self.maqueta.mundo(x, y)
            carril = ANCHO_CALLE / 4
            raiz.position = (px + lx * carril, 0.0, pz + lz * carril)
            raiz.rotation_y = moto["rumbo"] or 0.0
            tinte = colores[r.estado.value]
            moto["baliza"].color = color_ursina(tinte)
            moto["aro"].color = color_ursina(tinte)
            moto["faro"].color = color_ursina(mezclar(self.col["foco_dia"], self.col["faro_moto"], noche))
            self._comida_en_moto(r, moto, dt)

    def _ruta_elegida(self) -> None:
        """Si el panel muestra un repartidor de esta ciudad, una línea con las calles que le faltan.

        Por qué: el panel dice "le faltan 6 cuadras por la ruta más rápida (A*)"; la línea
        deja VER esa ruta, y cómo cambia cuando entra un trancón y la recalcula.
        Se rehace solo cuando cambia la ruta (no en cada cuadro).
        """
        from sim.agentes.repartidor import Repartidor
        from ursina import Entity, Mesh, destroy
        from ursina.shaders import unlit_shader
        r = self._elegido()
        clave = (r.unique_id, r.nodo, tuple(r.ruta), r.estado) if isinstance(r, Repartidor) and r.ruta else None
        if clave == self._clave_ruta:
            return
        self._clave_ruta = clave
        if self.linea_ruta is not None:
            destroy(self.linea_ruta)
            self.linea_ruta = None
        if clave is None:
            return
        cr = self.cfg["detalle"]["ruta"]
        puntos = [self.mundo(x, y, cr["altura"]) for x, y in [r.nodo] + r.ruta]
        self.linea_ruta = Entity(model=Mesh(vertices=puntos, mode="line", thickness=cr["grosor"]),
                                 shader=unlit_shader, color=color_ursina(self.col["repartidor"][r.estado.value]))
        self.linea_ruta.hide(MASCARA_SOMBRA)

    def _comida_en_moto(self, r, moto: dict, dt: float) -> None:
        """Si la moto lleva un pedido, encima flota la comida de ese restaurante.

        Por qué: el color de la baliza obliga a mirar la leyenda; una pizza flotando
        sobre la moto se entiende sola ("esta moto lleva una pizza a alguien").
        """
        from ursina import time
        lleva = r.estado == EstadoRepartidor.YENDO_A_CLIENTE and r.pedido is not None
        nombre = r.pedido.restaurante.nombre if lleva else None
        comidas = moto.setdefault("comida", {})
        for otro, entidad in comidas.items():
            if (otro == nombre) != entidad.enabled:
                entidad.enabled = otro == nombre
        if nombre is None:
            return
        cc = self.cfg["comida_en_moto"]
        entidad = comidas.get(nombre)
        if entidad is None:
            entidad = comidas[nombre] = self._nueva_comida(nombre)
        rebote = cc["rebote"] * math.sin(time.time() * cc["frecuencia_rebote"] + r.unique_id)
        x, _, z = moto["raiz"].position
        entidad.position = (x, cc["altura"] + rebote, z)
        entidad.rotation_y += cc["giro"] * dt

    def _nueva_comida(self, nombre: str):
        """Ícono de la comida sobre un platico brillante del color del restaurante."""
        from ursina import Entity
        cc = self.cfg["comida_en_moto"]
        raiz = Entity(scale=cc["escala"])
        Entity(parent=raiz, model=a_malla_ursina(malla_icono(nombre, self.col)), shader=self._sombreado.iluminado)
        plato = Malla()
        plato.cilindro((0, -0.04, 0), 0.4, 0.04, 20, self.col["restaurantes"][nombre])
        e_plato = Entity(parent=raiz, model=a_malla_ursina(plato), shader=self._sombreado.emisivo)
        e_plato.hide(MASCARA_SOMBRA)
        return raiz

    def _pedidos(self) -> None:
        from ursina import Entity, destroy, time
        activos = {p.id: p for p in self.modelo.pedidos_activos if not p.terminado}
        for pid in list(self.marcas_pedido):
            if pid not in activos:
                marca = self.marcas_pedido.pop(pid)
                destroy(marca["pin"])
                destroy(marca["aro"])
                pedido = self.pedido_de_marca.pop(pid)
                if pedido.estado == EstadoPedido.CANCELADO:
                    self.avisar(("cancelado", pid), "cancelado", self._sobre_destino(pedido.destino, 0.9),
                                self.cfg["colores"]["pedido"]["urgente"])
        cp = self.cfg["pin"]
        espera_maxima = self.modelo.cfg["pedidos"]["espera_maxima"]
        colores = self.col["pedido"]
        ahora = time.time()
        for pid, pedido in activos.items():
            marca = self.marcas_pedido.get(pid)
            if marca is None:
                pin = Entity(model=a_malla_ursina(self.malla_pin), shader=self._sombreado.emisivo, scale=cp["escala"])
                aro_malla = Malla()
                aro_malla.anillo_suelo((0, 0, 0), 0.3, 0.4, 20, (1, 1, 1, 1))
                aro = Entity(model=a_malla_ursina(aro_malla), shader=self._sombreado.emisivo)
                aro.hide(MASCARA_SOMBRA)
                marca = self.marcas_pedido[pid] = {"pin": pin, "aro": aro}
                self.pedido_de_marca[pid] = pedido
            urgente = pedido.edad(self.modelo.reloj.tick) / espera_maxima > 0.66
            tinte = colores["urgente"] if urgente else colores[pedido.estado.value]
            # El urgente late más rápido: llama la atención antes de que se cancele.
            pulso = math.sin(ahora * (8 if urgente else 3) + pid)
            pin = marca["pin"]
            pin.color = color_ursina(tinte)
            pin.position = self._sobre_destino(pedido.destino, cp["altura"] + cp["rebote"] * pulso)
            pin.rotation_y += cp["giro"] * min(time.dt, 0.1)
            base = self._sobre_destino(pedido.destino, 0)
            aro = marca["aro"]
            aro.position = (base[0], 0.07, base[2])
            aro.color = color_ursina(tinte)
            aro.scale = 1 + 0.15 * pulso

    # ------------------------------------------------------------ habitantes
    def _nuevo_icono_habitante(self, h) -> dict:
        """Círculo con el hambre adentro y un aro con lo que está haciendo. Tiene colisionador
        para poder darle clic (Ursina llama a on_click cuando el mouse lo toca)."""
        from ursina import Entity, Text, color, scene
        raiz = Entity(parent=scene, billboard=True, scale=self.cfg["habitantes"]["escala"])
        aro = Entity(parent=raiz, model="quad", texture="circle", z=0.01, scale=1.3)
        fondo = Entity(parent=raiz, model="quad", texture="circle", collider="box")
        texto = Text(parent=raiz, text="", origin=(0, 0), scale=17, z=-0.01, color=color.hex("#15181f"))
        for e in (aro, fondo):
            e.setTransparency(True)
        raiz.hide(MASCARA_SOMBRA)
        fondo.on_click = lambda: self.panel_detalle and self.panel_detalle.seleccionar(self, h)
        return {"raiz": raiz, "aro": aro, "fondo": fondo, "texto": texto, "habitante": h,
                "aspecto": None, "escala": None}

    def _habitantes(self) -> None:
        """Un ícono por habitante sobre el edificio donde está ahora (su casa, la oficina o la
        universidad). Si hay varios en el mismo edificio, van en filas, ordenados por número.

        Por qué solo cuando cambia el minuto: lugar, hambre y estado cambian con la
        simulación, no con cada cuadro; tocar 30 íconos (posición, color, texto) en cada
        cuadro le quitaba un tercio de la velocidad a la demo. Lo único que se mira en
        cada cuadro es el agrandado del que está bajo el mouse o seleccionado.
        """
        from ursina import mouse, time
        visible = self.mostrar_habitantes
        if visible != getattr(self, "_iconos_visibles", None):
            self._iconos_visibles = visible
            for icono in self.iconos_habitante.values():
                icono["raiz"].enabled = visible
            self._minuto_iconos = None
        if not visible:
            return
        if self._minuto_iconos != self.modelo.reloj.tick:
            self._minuto_iconos = self.modelo.reloj.tick
            self._ubicar_iconos()
        ch = self.cfg["habitantes"]
        elegido = self._elegido()
        for icono in self.iconos_habitante.values():
            seleccionado = icono["habitante"] is elegido
            grande = seleccionado or mouse.hovered_entity is icono["fondo"]
            pulso = 1 + 0.08 * math.sin(time.time() * 6) if seleccionado else 1
            escala = ch["escala"] * (ch["agrandar"] if grande else 1) * pulso
            if escala != icono["escala"]:
                icono["escala"] = escala
                icono["raiz"].scale = escala

    def _ubicar_iconos(self) -> None:
        from ursina import Vec3
        from vistas.detalle_habitante import estado
        ch = self.cfg["habitantes"]
        por_lugar: dict[tuple, list] = {}
        for h in sorted(self.modelo.habitantes, key=lambda x: x.unique_id):
            por_lugar.setdefault(h.ubicacion, []).append(h)
        for lugar, gente in por_lugar.items():
            base = self._sobre_destino(lugar, ch["altura"])
            filas = (len(gente) + ch["por_fila"] - 1) // ch["por_fila"]
            for i, h in enumerate(gente):
                icono = self.iconos_habitante.get(h.unique_id)
                if icono is None:
                    icono = self.iconos_habitante[h.unique_id] = self._nuevo_icono_habitante(h)
                fila, columna = divmod(i, ch["por_fila"])
                en_fila = min(ch["por_fila"], len(gente) - fila * ch["por_fila"])
                posicion = (base.x + (columna - (en_fila - 1) / 2) * ch["espacio"],
                            base.y + (filas - 1 - fila) * ch["espacio"], base.z)
                _, clave = estado(h)
                aspecto = (posicion, round(h.hambre), clave)
                if aspecto == icono["aspecto"]:
                    continue  # nada que se vea cambió: no se toca la Entity
                icono["aspecto"] = aspecto
                icono["raiz"].position = Vec3(*posicion)
                icono["fondo"].color = color_ursina(color_hambre(self.cfg, h.hambre, h.cfg["umbral_hambre"]))
                icono["aro"].color = color_ursina(self.col["estado_habitante"][clave])
                poner_texto(icono["texto"], "z" if clave == "durmiendo" else f"{h.hambre:.0f}")

    def _sobre_destino(self, nodo: tuple, altura: float):
        """Encima del edificio del cliente (no en la esquina): ahí vive quien pidió."""
        from ursina import Vec3
        if nodo in self.maqueta.lotes:
            x, z = self.maqueta.lotes[nodo]
            return Vec3(x, self.maqueta.altura_lote[nodo] + altura, z)
        return self.mundo(*nodo, altura=altura + 0.5)

    def _lluvia(self, lluvia: float) -> None:
        from ursina import time
        visible = lluvia > 0.01
        if self.lluvia.enabled != visible:  # prender/apagar solo cuando cambia
            self.lluvia.enabled = visible
        if visible:
            cl = self.cfg["lluvia"]
            r, g, b, _ = hex_a_color(self.col_hex("gota"))
            self.lluvia.color = color_ursina((r, g, b, cl["opacidad"] * lluvia))
            # módulo: la malla baja y vuelve a subir `alto`, donde está su copia
            self.lluvia.y = -((-self.lluvia.y + cl["velocidad"] * min(time.dt, 0.1)) % cl["alto"])

    # --------------------------------------------------------------- textos
    def resumen(self) -> str:
        m = self.modelo
        h = self.cfg["hud"]
        s, t = etiqueta(h["suave"]), etiqueta(h["texto"])
        entregados = [p.tiempo_entrega for p in m.pedidos if p.tiempo_entrega is not None]
        promedio = f"{sum(entregados) / len(entregados):.1f} min" if entregados else "-"
        creados = max(1, m.conteo["creados"])
        filas = [("Entregados", f"{m.conteo['entregados']}"),
                 ("Cancelados", f"{m.conteo['cancelados']}  ({m.conteo['cancelados'] / creados:.0%})"),
                 ("Rechazos", f"{sum(r.rechazos for r in m.repartidores)}"),
                 ("Entrega promedio", promedio),
                 ("Recompensa", f"{sum(r.recompensa_acumulada for r in m.repartidores):+.1f}")]
        return "\n".join(f"{s}{nombre}:  {t}{valor}" for nombre, valor in filas)

    def chips(self) -> str:
        """Día, clima y tráfico en una línea, con color: se leen de un vistazo."""
        m, h = self.modelo, self.cfg["hud"]
        clima = f"{etiqueta('#8fc4ff')}Lluvia" if m.eventos.lloviendo else f"{etiqueta('#ffe08a')}Despejado"
        trafico = (f"{etiqueta('#ff6b5e')}Hora pico" if m.eventos.trancon_activo
                   else f"{etiqueta('#8fdc8f')}Tráfico normal")
        dia = min(m.reloj.dia + 1, dias_del_episodio(m))  # al terminar, el reloj ya marca el día siguiente
        return f"{etiqueta(h['suave'])}Día {dia} de {dias_del_episodio(m)}   {clima}   {trafico}"

    def mapa(self) -> str:
        h = self.cfg["hud"]
        return f"{etiqueta(h['suave'])}Mapa {etiqueta(h['acento'])}#{self.modelo.cfg['ciudad']['semilla_mapa']}"

    def reloj(self) -> str:
        r = self.modelo.reloj
        return f"{r.hora:02d}:{r.minuto_del_dia % MINUTOS_POR_HORA:02d}"

    def lista_repartidores(self) -> str:
        h = self.cfg["hud"]
        filas = []
        for r in self.modelo.repartidores:
            tinte = self.cfg["colores"]["repartidor"][r.estado.value]
            filas.append(f"{etiqueta(h['texto'])}Repartidor {r.unique_id}   {etiqueta(tinte)}"
                         f"{r.estado.value.replace('_', ' ')}   {etiqueta(h['suave'])}"
                         f"${(r.ganancias - r.costos) / 1000:.1f}k")
        return "\n".join(filas)

    def _textos(self) -> None:
        m, h = self.modelo, self.cfg["hud"]
        pausa = f"   {etiqueta(h['acento'])}PAUSA" if self.pausa else ""
        politica = m.cfg["repartidores"]["politica"]
        self.panel_izq.poner(f"{self.chips()}\n{etiqueta(h['suave'])}Velocidad {self.velocidad.texto()}{pausa}   "
                             f"Política {etiqueta(h['acento'])}{politica}   {self.mapa()}\n\n{self.resumen()}",
                             titulo=self.reloj())
        self.panel_der.poner(self.lista_repartidores())


# ------------------------------------------------------------------ dos ciudades
class VistaDuelo3D:
    """Antes vs después en 3D: la misma ciudad dos veces, una al lado de la otra.

    Las dos avanzan el mismo número de minutos en cada cuadro, así el reloj es
    siempre el mismo. Lo único distinto entre ellas es la política (sin entrenar
    vs entrenada); ver crear_duelo() en main.py.
    """

    def __init__(self, antes: CiudadModel, despues: CiudadModel, titulo_antes: str, titulo_despues: str):
        self.velocidad = Velocidad(despues.cfg["vistas"])
        self.cfg = despues.cfg["vista_duelo_3d"]
        self.cfg3d = despues.cfg["vista_3d"]
        corrimiento = (despues.ciudad.tamano + self.cfg["separacion"]) * ESCALA
        self.izq = Vista3D(antes, 0.0, titulo_antes, self.cfg["color_antes"], self.velocidad)
        self.der = Vista3D(despues, corrimiento, titulo_despues, self.cfg["color_despues"], self.velocidad)
        self.titulos = (titulo_antes, titulo_despues)
        self.pausa = False

    def correr(self) -> None:
        try:
            app = abrir_ventana("Antes vs después de entrenar (3D)", self.cfg3d)
        except ImportError:
            raise SystemExit("La vista 3D necesita Ursina: pip install ursina") from None
        self.construir()
        app.run()

    def construir(self) -> None:
        from ursina import Entity, Text, Vec3, camera, color
        from ursina.shaders import unlit_shader
        x0, _, z0, z1 = self.izq.maqueta.limites()
        _, x1, _, _ = self.der.maqueta.limites()
        ancho_visible = (x1 - x0) * self.cfg["encuadre"]
        alto = max(self.izq.maqueta.alto_maximo, self.der.maqueta.alto_maximo)
        self.ambiente = Ambiente(self.cfg3d, (x0, x1, z0, z1), alto, self.izq.rng, ancho_visible)
        self.ambiente.camara.rotation = (self.cfg3d["camara"]["inclinacion"], 0, 0)  # de frente: mismo tamaño
        for vista in (self.izq, self.der):
            vista.ambiente = self.ambiente
            vista.construir_ciudad()
        h = self.cfg3d["hud"]
        self.panel_izq = Panel(h, "arriba_izq", escala=0.95, titulo="", escala_titulo=1.3)
        self.panel_der = Panel(h, "arriba_der", escala=0.95, titulo="", escala_titulo=1.3)
        self.panel_centro = Panel(h, "arriba", escala=0.95, titulo="", escala_titulo=2.2)
        self.panel_leyenda = Panel(h, "abajo_izq", escala=0.8)
        self.panel_leyenda.poner(self.der.leyenda() + "   M: marcador")
        # Un solo panel para las dos ciudades: el mismo habitante existe en ambas (mismas semillas)
        self.panel_detalle = PanelDetalle(self.cfg3d, self.cfg3d["detalle"]["margen_panel_duelo"])
        self.izq.panel_detalle = self.der.panel_detalle = self.panel_detalle
        # Marcador final: fuente monoespaciada para que las columnas queden alineadas, sobre
        # un panel oscuro casi opaco para que la ciudad de atrás no estorbe al leer.
        self.fondo_marcador = Entity(parent=camera.ui, model="quad", color=color.rgba(0.06, 0.07, 0.1, 0.94),
                                     shader=unlit_shader, z=0.1, enabled=False)
        self.panel_marcador = Text(text="", font="VeraMono.ttf", position=Vec3(0, 0, 0), origin=(0, 0),
                                   scale=1.25, enabled=False)
        self.marcador_mostrado = False
        self.controlador = crear_controlador(self.actualizar_cuadro, self.tecla)

    def tecla(self, tecla: str) -> None:
        if tecla == "m":
            self.panel_marcador.enabled = not self.panel_marcador.enabled
            return
        if tecla == "n":  # las dos a la vez: el reloj de las dos ciudades siempre es el mismo
            self.izq.saltar_dia()
            self.der.saltar_dia()
            return
        if tecla == "h":
            self.izq.alternar_habitantes()
            self.der.alternar_habitantes()
            return
        tecla_comun(tecla, self.velocidad, self.alternar_pausa, self.ambiente.camara)

    def alternar_pausa(self) -> None:
        self.pausa = not self.pausa
        self.izq.pausa = self.der.pausa = self.pausa

    def actualizar_cuadro(self) -> None:
        from ursina import time
        minutos = 0 if self.pausa else self.velocidad.pasos(time.dt)
        m_izq, m_der = self.izq.modelo, self.der.modelo
        for vista in (self.izq, self.der):
            vista.avanzar(minutos)
        estado = self.ambiente.actualizar(self.der.minuto_suave(), m_der.eventos.lloviendo, self.pausa)
        for vista in (self.izq, self.der):
            vista.dibujar(self.velocidad.fraccion, estado, self.ambiente.lluvia)
        self.panel_detalle.actualizar()
        h = self.cfg3d["hud"]
        self.panel_izq.poner(self.izq.resumen(), titulo=f"{etiqueta(self.cfg['color_antes'])}{self.titulos[0]}")
        self.panel_der.poner(self.der.resumen(), titulo=f"{etiqueta(self.cfg['color_despues'])}{self.titulos[1]}")
        diferencia = (sum(r.recompensa_acumulada for r in m_der.repartidores)
                      - sum(r.recompensa_acumulada for r in m_izq.repartidores))
        pausa = f"   {etiqueta(h['acento'])}PAUSA" if self.pausa else ""
        self.panel_centro.poner(f"{self.der.chips()}\n{etiqueta(h['suave'])}Velocidad {self.velocidad.texto()}"
                                f"{pausa}   {self.der.mapa()}\n{etiqueta(h['suave'])}Recompensa después − antes:  "
                                f"{etiqueta(h['acento'])}{diferencia:+.1f}", titulo=self.der.reloj())
        # Al terminar el día el marcador aparece solo, una vez (con M se oculta o se vuelve a ver).
        if not (m_izq.running or m_der.running) and not self.marcador_mostrado:
            self.marcador_mostrado = True
            self.panel_marcador.enabled = True
        self.fondo_marcador.enabled = self.panel_marcador.enabled
        if self.panel_marcador.enabled:
            poner_texto(self.panel_marcador, texto_marcador(m_izq, m_der, *self.titulos))
            from ursina import window
            # Ancho fijo en proporción a la pantalla: Ursina mide mal el ancho de la fuente monoespaciada.
            self.fondo_marcador.scale = (window.aspect_ratio * self.cfg["ancho_marcador"],
                                         self.panel_marcador.height + self.cfg["margen_marcador"])
