"""La ciudad como maqueta low poly: terreno, calles, manzanas, edificios, árboles y faroles.

Solo construye geometría (listas de triángulos, ver vistas/geometria.py); no abre
ventana ni importa Ursina. La vista 3D convierte cada Malla en una sola malla de
Ursina: miles de piezas, pocas órdenes de dibujo.

Por qué separar lo que brilla (ventanas, focos, halos) de lo iluminado: de noche
se encienden sin rehacer la geometría, cambiando solo el color de esa malla.

Las medidas de las piezas (alto de un piso, ancho de una calle) son el DISEÑO de
la maqueta y viven aquí como constantes con nombre; los colores y las cantidades
que se quieran ajustar están en config/base.yaml (vista_3d).
"""
from __future__ import annotations

import math
import random

from vistas.geometria import BLANCO, Color, Malla, aclarar, girar_y, hex_a_color

ESCALA = 3.0            # unidades de mundo por cuadra
ANCHO_CALLE = 0.7
ANCHO_ANDEN = 0.18
ALTO_ANDEN = 0.05
ALTO_RAYA = 0.008       # rayas y cebras: apenas sobre el asfalto
LARGO_RAYA, HUECO_RAYA, GROSOR_RAYA = 0.28, 0.22, 0.04
FRANJAS_CEBRA, ANCHO_FRANJA = 5, 0.06
ESPESOR_PASTO, ESPESOR_TIERRA = 0.18, 1.4
ALTO_PISO = 0.26
SEPARACION_VENTANA = 0.004
PROFUNDIDAD_DIORAMA = ESPESOR_PASTO + ESPESOR_TIERRA

# Interior de una manzana (sin calle ni andén) y sus cuatro cuadrantes ("lotes")
_MARGEN = ANCHO_CALLE / 2 + ANCHO_ANDEN
LADO_LOTE = (ESCALA - 2 * _MARGEN) / 2


def _colores(cfg: dict) -> dict:
    """Convierte la paleta del YAML (hex) a tuplas RGBA una sola vez."""
    c = {}
    for nombre, valor in cfg["colores"].items():
        if isinstance(valor, list):
            c[nombre] = [hex_a_color(v) for v in valor]
        elif isinstance(valor, dict):
            c[nombre] = {k: hex_a_color(v) for k, v in valor.items()}
        else:
            c[nombre] = hex_a_color(valor)
    return c


class Maqueta:
    """Toda la geometría fija de una ciudad, corrida `desplazamiento` en x (para el duelo)."""

    def __init__(self, ciudad, cfg: dict, rng: random.Random, desplazamiento: float = 0.0):
        self.ciudad = ciudad
        self.cfg = cfg
        self.rng = rng
        self.desplazamiento = desplazamiento
        self.n = ciudad.tamano
        self.col = _colores(cfg)
        self.estatico = Malla()                 # todo lo que recibe luz y sombra
        self.ventanas_encendibles = Malla()     # se prenden de noche
        self.ventanas_apagadas = Malla()        # siguen oscuras de noche (nadie en casa)
        self.focos = Malla()                    # bombillos de los faroles
        self.halos = Malla()                    # charcos de luz en el piso (con textura)
        self.restaurantes: dict[str, dict] = {}
        self.lotes: dict[tuple, tuple] = {}     # nodo -> centro del lote de su edificio
        self.altura_lote: dict[tuple, float] = {}  # nodo -> alto del edificio (los pines flotan encima)
        self.alto_maximo = 0.0
        self._ocupados: set[tuple] = set()
        self.construir()

    # ------------------------------------------------------------ coordenadas
    def mundo(self, x: float, y: float, altura: float = 0.0) -> tuple:
        return (self.desplazamiento + x * ESCALA, altura, y * ESCALA)

    @property
    def lado(self) -> float:
        return self.n * ESCALA

    def limites(self) -> tuple[float, float, float, float]:
        """(x mínima, x máxima, z mínima, z máxima) del diorama, con el borde de pasto."""
        borde = self.cfg["borde"] * ESCALA
        return (self.desplazamiento - borde, self.desplazamiento + self.lado + borde, -borde, self.lado + borde)

    def lote_de_nodo(self, nodo: tuple) -> tuple[int, int, int, int]:
        """(manzana x, manzana z, cuadrante x, cuadrante z) donde se dibuja el edificio del nodo.
        Igual que antes: el edificio va en la manzana de arriba-derecha, salvo en el borde."""
        x, y = nodo
        bx, qx = (x, 0) if x < self.n else (self.n - 1, 1)
        bz, qz = (y, 0) if y < self.n else (self.n - 1, 1)
        return bx, bz, qx, qz

    def centro_lote(self, bx: int, bz: int, qx: int, qz: int) -> tuple[float, float]:
        x0 = self.desplazamiento + bx * ESCALA + _MARGEN
        z0 = bz * ESCALA + _MARGEN
        return x0 + LADO_LOTE * (qx + 0.5), z0 + LADO_LOTE * (qz + 0.5)

    # -------------------------------------------------------------- construir
    def construir(self) -> None:
        self._terreno()
        self._calles()
        self._manzanas()
        for nodo, tipo in sorted(self.ciudad.edificios.items()):
            self._edificio(nodo, tipo)
        self._parques()
        self._faroles()
        self._bosque_del_borde()

    def _terreno(self) -> None:
        """La maqueta: una losa de pasto sobre un bloque de tierra con dos vetas."""
        x0, x1, z0, z1 = self.limites()
        cx, cz, ax, az = (x0 + x1) / 2, (z0 + z1) / 2, x1 - x0, z1 - z0
        m, c = self.estatico, self.col
        m.caja((cx, -ESPESOR_PASTO / 2, cz), (ax, ESPESOR_PASTO, az), c["pasto_oscuro"], color_arriba=c["pasto"])
        y_tierra = -ESPESOR_PASTO - ESPESOR_TIERRA / 2
        m.caja((cx, y_tierra, cz), (ax, ESPESOR_TIERRA, az), c["tierra"])
        for altura, grosor in ((0.35, 0.08), (0.8, 0.12)):  # vetas más oscuras, fracción del bloque
            y = -ESPESOR_PASTO - ESPESOR_TIERRA * altura
            m.caja((cx, y, cz), (ax + 0.01, ESPESOR_TIERRA * grosor, az + 0.01), c["tierra_oscura"])

    def _calles(self) -> None:
        m, c = self.estatico, self.col
        mitad = ANCHO_CALLE / 2
        for u, v in self.ciudad.grafo.edges:
            (ax, _, az), (bx, _, bz) = self.mundo(*u), self.mundo(*v)
            x0, x1 = min(ax, bx) - mitad, max(ax, bx) + mitad
            z0, z1 = min(az, bz) - mitad, max(az, bz) + mitad
            m.rect_suelo(x0, z0, x1, z1, 0.002, c["asfalto"])
            self._raya_central(ax, az, bx, bz)
        for nodo in self.ciudad.grafo.nodes:
            if (nodo[0] + nodo[1]) % 2 == 0:
                self._cebras(nodo)

    def _raya_central(self, ax: float, az: float, bx: float, bz: float) -> None:
        """Línea discontinua amarilla por el centro de la calle, sin pisar los cruces."""
        largo = math.hypot(bx - ax, bz - az)
        ux, uz = (bx - ax) / largo, (bz - az) / largo
        d = ANCHO_CALLE
        while d + LARGO_RAYA < largo - ANCHO_CALLE:
            x0, z0 = ax + ux * d, az + uz * d
            x1, z1 = ax + ux * (d + LARGO_RAYA), az + uz * (d + LARGO_RAYA)
            g = GROSOR_RAYA / 2
            self.estatico.rect_suelo(min(x0, x1) - g * abs(uz), min(z0, z1) - g * abs(ux),
                                     max(x0, x1) + g * abs(uz), max(z0, z1) + g * abs(ux), ALTO_RAYA,
                                     self.col["raya"])
            d += LARGO_RAYA + HUECO_RAYA

    def _cebras(self, nodo: tuple) -> None:
        """Pasos de cebra en las cuatro bocas del cruce."""
        cx, _, cz = self.mundo(*nodo)
        mitad = ANCHO_CALLE / 2
        paso = ANCHO_CALLE / FRANJAS_CEBRA
        largo = ANCHO_FRANJA * 3
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            vecino = (nodo[0] + dx, nodo[1] + dz)
            if vecino not in self.ciudad.grafo:
                continue
            inicio = mitad + ANCHO_FRANJA
            for i in range(FRANJAS_CEBRA):
                lateral = -mitad + paso * (i + 0.5)
                if dx:
                    x0, x1 = sorted((cx + dx * inicio, cx + dx * (inicio + largo)))
                    self.estatico.rect_suelo(x0, cz + lateral - ANCHO_FRANJA / 2, x1, cz + lateral + ANCHO_FRANJA / 2,
                                             ALTO_RAYA, self.col["cebra"])
                else:
                    z0, z1 = sorted((cz + dz * inicio, cz + dz * (inicio + largo)))
                    self.estatico.rect_suelo(cx + lateral - ANCHO_FRANJA / 2, z0, cx + lateral + ANCHO_FRANJA / 2, z1,
                                             ALTO_RAYA, self.col["cebra"])

    def _manzanas(self) -> None:
        """Cada manzana: andén gris con bordillo y el interior de pasto un poco más alto."""
        m, c = self.estatico, self.col
        for bx in range(self.n):
            for bz in range(self.n):
                x0 = self.desplazamiento + bx * ESCALA + ANCHO_CALLE / 2
                z0 = bz * ESCALA + ANCHO_CALLE / 2
                lado = ESCALA - ANCHO_CALLE
                m.caja((x0 + lado / 2, ALTO_ANDEN / 2, z0 + lado / 2), (lado, ALTO_ANDEN, lado),
                       c["bordillo"], color_arriba=c["anden"])
                interior = lado - 2 * ANCHO_ANDEN
                m.rect_suelo(x0 + ANCHO_ANDEN, z0 + ANCHO_ANDEN, x0 + ANCHO_ANDEN + interior,
                             z0 + ANCHO_ANDEN + interior, ALTO_ANDEN + 0.003, c["pasto_manzana"])

    # -------------------------------------------------------------- edificios
    def _edificio(self, nodo: tuple, tipo: str) -> None:
        lote = self.lote_de_nodo(nodo)
        self._ocupados.add(lote)
        bx, bz, qx, qz = lote
        cx, cz = self.centro_lote(*lote)
        self.lotes[nodo] = (cx, cz)
        frente_z = 180.0 if qz == 0 else 0.0  # el frente mira a la calle del nodo
        frente_x = -90.0 if qx == 0 else 90.0
        y0 = ALTO_ANDEN
        # piso del lote: adoquín claro bajo el edificio
        mitad = LADO_LOTE / 2 - 0.02
        self.estatico.rect_suelo(cx - mitad, cz - mitad, cx + mitad, cz + mitad, y0 + 0.005, self.col["lote"])
        if tipo == "casa":
            giro = frente_z if self.rng.random() < 0.6 else frente_x
            alto = self._casa(cx, y0, cz, giro)
        elif tipo == "oficina":
            alto = self._oficina(cx, y0, cz, frente_z)
        elif tipo == "universidad":
            alto = self._universidad(cx, y0, cz, frente_z)
        else:
            nombre = next(r for r, nd in self.ciudad.restaurantes.items() if nd == nodo)
            alto = self._restaurante(nombre, cx, y0, cz, frente_z)
        self.altura_lote[nodo] = y0 + alto

    def _pieza(self, malla: Malla, origen: tuple, giro: float, local: tuple, tamano: tuple, color: Color,
               **extra) -> None:
        """Una caja en coordenadas del edificio (frente = +z local), girada con él."""
        dx, dy, dz = girar_y(local, giro)
        malla.caja((origen[0] + dx, origen[1] + dy, origen[2] + dz), tamano, color, giro=giro, **extra)

    def _ventanas_fachada(self, origen: tuple, giro: float, ancho: float, fondo: float, pisos: list[float],
                          por_cara: int, tam: tuple[float, float], caras=("frente", "atras", "izq", "der"),
                          prob_encendida: float | None = None) -> None:
        """Ventanas repartidas por piso en cada cara. Cada una va a la malla de las que se
        prenden o a la de las que no, al azar: así de noche no se ve un tablero uniforme."""
        prob = self.cfg["ventanas_encendidas"] if prob_encendida is None else prob_encendida
        normales = {"frente": (0, 0, 1), "atras": (0, 0, -1), "der": (1, 0, 0), "izq": (-1, 0, 0)}
        for cara in caras:
            nx, _, nz = normales[cara]
            largo = ancho if nz else fondo
            distancia = (fondo if nz else ancho) / 2
            for y in pisos:
                for i in range(por_cara):
                    t = (i + 0.5) / por_cara - 0.5
                    local = (nx * distancia + (t * largo if nz else 0), y, nz * distancia + (t * largo if nx else 0))
                    dx, dy, dz = girar_y(local, giro)
                    normal = girar_y((nx, 0, nz), giro)
                    destino = self.ventanas_encendibles if self.rng.random() < prob else self.ventanas_apagadas
                    tono = self.rng.choice(self.col["tonos_ventana"])
                    destino.ventana((origen[0] + dx, origen[1] + dy, origen[2] + dz), tam[0], tam[1], normal, tono,
                                    SEPARACION_VENTANA)

    def _casa(self, x: float, y0: float, z: float, giro: float) -> float:
        m, c, rng = self.estatico, self.col, self.rng
        o = (x, y0, z)
        ancho, fondo = rng.uniform(0.62, 0.78), rng.uniform(0.52, 0.66)
        pisos = 2 if rng.random() < 0.3 else 1
        alto = ALTO_PISO * 1.5 * pisos
        pared = rng.choice(c["casas"])
        self._pieza(m, o, giro, (0, alto / 2, 0), (ancho, alto, fondo), pared)
        techo = rng.choice(c["techos"])
        alto_techo = rng.uniform(0.22, 0.32)
        dx, dy, dz = girar_y((0, alto, 0), giro)
        m.techo_dos_aguas((x + dx, y0 + dy, z + dz), fondo + 0.1, ancho + 0.1, alto_techo, techo,
                          aclarar(pared, -0.08), giro=giro + 90)
        # puerta y escalón
        self._pieza(m, o, giro, (ancho * 0.22, 0.13, fondo / 2 + 0.01), (0.13, 0.26, 0.02), c["puerta"])
        self._pieza(m, o, giro, (ancho * 0.22, 0.015, fondo / 2 + 0.05), (0.2, 0.03, 0.08), c["bordillo"])
        alturas = [ALTO_PISO * 1.5 * (p + 0.55) for p in range(pisos)]
        self._ventanas_fachada(o, giro, ancho, fondo, alturas, 2, (0.13, 0.12), caras=("atras", "izq", "der"))
        self._ventanas_fachada(o, giro, ancho * 0.45, fondo, alturas, 1, (0.13, 0.12), caras=("frente",))
        if rng.random() < 0.5:  # chimenea
            self._pieza(m, o, giro, (-ancho * 0.25, alto + 0.2, -fondo * 0.15), (0.08, 0.26, 0.08), c["ladrillo"])
        for lado in (-1, 1):  # arbustos a lado y lado de la entrada
            if rng.random() < 0.6:
                dx, dy, dz = girar_y((lado * ancho * 0.5, 0.05, fondo / 2 + 0.1), giro)
                m.esfera((x + dx, y0 + dy, z + dz), (0.08, 0.07, 0.08), rng.choice(c["copas"]), rng, 0.15)
        return alto + alto_techo

    def _oficina(self, x: float, y0: float, z: float, giro: float) -> float:
        m, c, rng = self.estatico, self.col, self.rng
        o = (x, y0, z)
        lado = rng.uniform(0.7, 0.84)
        pisos = rng.randint(*self.cfg["pisos_oficina"])
        alto = ALTO_PISO * pisos
        cuerpo = rng.choice(c["oficinas"])
        escalonada = pisos >= 9 and rng.random() < 0.5
        bajo = alto * 0.65 if escalonada else alto
        self._pieza(m, o, giro, (0, bajo / 2, 0), (lado, bajo, lado), cuerpo, color_arriba=aclarar(cuerpo, -0.15))
        # recepción: franja de vidrio en el primer piso
        self._pieza(m, o, giro, (0, ALTO_PISO * 0.5, 0), (lado + 0.02, ALTO_PISO * 0.8, lado + 0.02), c["recepcion"])
        pisos_bajo = [ALTO_PISO * (p + 0.5) for p in range(1, int(bajo / ALTO_PISO))]
        self._ventanas_fachada(o, giro, lado, lado, pisos_bajo, 3, (lado / 3 - 0.06, ALTO_PISO * 0.6))
        techo_y = bajo
        if escalonada:
            arriba = lado * 0.7
            self._pieza(m, o, giro, (0, bajo + (alto - bajo) / 2, 0), (arriba, alto - bajo, arriba), cuerpo,
                        color_arriba=aclarar(cuerpo, -0.15))
            pisos_arriba = [ALTO_PISO * (p + 0.5) for p in range(int(bajo / ALTO_PISO), pisos)]
            self._ventanas_fachada(o, giro, arriba, arriba, pisos_arriba, 2, (arriba / 2 - 0.08, ALTO_PISO * 0.6))
            techo_y, lado = alto, arriba
        # cornisa, máquinas del aire y antena
        self._pieza(m, o, giro, (0, techo_y + 0.02, 0), (lado + 0.04, 0.04, lado + 0.04), aclarar(cuerpo, 0.2))
        for _ in range(rng.randint(1, 2)):
            px, pz = rng.uniform(-lado / 4, lado / 4), rng.uniform(-lado / 4, lado / 4)
            self._pieza(m, o, giro, (px, techo_y + 0.08, pz), (0.14, 0.1, 0.12), c["maquinas"])
        if pisos >= 10:
            m.cilindro((x, y0 + techo_y + 0.3, z), 0.015, 0.6, 4, c["maquinas"])
        self.alto_maximo = max(self.alto_maximo, alto + 0.6)
        return alto

    def _universidad(self, x: float, y0: float, z: float, giro: float) -> float:
        m, c = self.estatico, self.col
        o = (x, y0, z)
        ancho, fondo, alto = 0.86, 0.6, 0.62
        self._pieza(m, o, giro, (0, alto / 2, -0.04), (ancho, alto, fondo), c["universidad"])
        self._pieza(m, o, giro, (0, alto + 0.03, -0.04), (ancho + 0.05, 0.06, fondo + 0.05), c["techo_universidad"])
        # pórtico: columnas y frontón triangular
        for i in range(4):
            dx, dy, dz = girar_y((-0.24 + 0.16 * i, alto / 2, fondo / 2 + 0.04), giro)
            m.cilindro((x + dx, y0 + dy, z + dz), 0.025, alto, 6, BLANCO)
        dx, dy, dz = girar_y((0, alto + 0.06, fondo / 2 - 0.04), giro)
        m.techo_dos_aguas((x + dx, y0 + dy, z + dz), 0.6, 0.2, 0.16, c["techo_universidad"], c["universidad"],
                          giro=giro)
        self._pieza(m, o, giro, (0, 0.02, fondo / 2 + 0.1), (0.62, 0.04, 0.14), c["bordillo"])
        self._ventanas_fachada(o, giro, ancho, fondo, [0.2, 0.45], 4, (0.08, 0.14), caras=("izq", "der", "atras"))
        # torre del reloj con techo en punta y la bandera
        self._pieza(m, o, giro, (0, alto + 0.3, -0.12), (0.2, 0.6, 0.2), c["universidad"])
        dx, dy, dz = girar_y((0, alto + 0.6, -0.12), giro)
        m.cilindro((x + dx, y0 + dy + 0.13, z + dz), 0.16, 0.26, 4, c["techo_universidad"], radio_arriba=0.0)
        self._pieza(self.ventanas_encendibles, o, giro, (0, alto + 0.42, -0.015), (0.1, 0.1, 0.01), c["reloj"])
        dx, dy, dz = girar_y((ancho / 2 - 0.05, 0, fondo / 2 + 0.12), giro)
        m.cilindro((x + dx, y0 + 0.5, z + dz), 0.01, 1.0, 4, c["maquinas"])
        self._pieza(m, (x + dx, y0, z + dz), giro, (0.1, 0.9, 0), (0.2, 0.12, 0.01), c["bandera"])
        self.alto_maximo = max(self.alto_maximo, alto + 0.9)
        return alto + 0.86

    def _restaurante(self, nombre: str, x: float, y0: float, z: float, giro: float) -> float:
        m, c = self.estatico, self.col
        o = (x, y0, z)
        marca = c["restaurantes"][nombre]
        ancho, fondo, alto = 0.8, 0.62, 0.48
        self._pieza(m, o, giro, (0, alto / 2, -0.06), (ancho, alto, fondo), c["fachada_restaurante"])
        self._pieza(m, o, giro, (0, alto + 0.03, -0.06), (ancho + 0.04, 0.06, fondo + 0.04), marca)
        # vitrina iluminada y puerta
        self._pieza(self.ventanas_encendibles, o, giro, (-0.12, 0.2, fondo / 2 - 0.05), (0.42, 0.2, 0.01),
                    c["vitrina"])
        self._pieza(m, o, giro, (0.24, 0.14, fondo / 2 - 0.05), (0.14, 0.28, 0.01), c["puerta"])
        # toldo a rayas, inclinado hacia la calle
        franjas = 8
        for i in range(franjas):
            x0l = -ancho / 2 + ancho * i / franjas
            x1l = x0l + ancho / franjas
            y_alto, y_bajo, z_pared, z_borde = alto * 0.82, alto * 0.62, fondo / 2 - 0.06, fondo / 2 + 0.16
            esquinas = [(x0l, y_alto, z_pared), (x0l, y_bajo, z_borde), (x1l, y_bajo, z_borde), (x1l, y_alto, z_pared)]
            p = [tuple(a + b for a, b in zip(o, girar_y(e, giro))) for e in esquinas]
            m.cuadrilatero(p[0], p[1], p[2], p[3], marca if i % 2 == 0 else BLANCO)
        # letrero sobre el techo
        self._pieza(m, o, giro, (0, alto + 0.2, 0.0), (0.6, 0.24, 0.05), marca)
        self._pieza(self.ventanas_encendibles, o, giro, (0, alto + 0.2, 0.03), (0.5, 0.15, 0.01), BLANCO)
        # mesas con parasol en el andén
        for lado in (-1, 1):
            dx, dy, dz = girar_y((lado * 0.3, 0, fondo / 2 + 0.3), giro)
            px, pz = x + dx, z + dz
            m.cilindro((px, y0 + 0.08, pz), 0.06, 0.02, 8, c["maquinas"])
            m.cilindro((px, y0 + 0.13, pz), 0.008, 0.26, 4, c["maquinas"])
            m.cono((px, y0 + 0.24, pz), 0.13, 0.08, 8, marca)
        frente = girar_y((0, 0, 1), giro)
        self.restaurantes[nombre] = {"centro": (x, y0 + alto, z), "frente": frente}
        self.alto_maximo = max(self.alto_maximo, alto + 2.0)
        return alto + 0.32

    # ------------------------------------------------------------- vegetación
    def arbol(self, x: float, z: float, y0: float, tamano: float = 1.0) -> None:
        m, c, rng = self.estatico, self.col, self.rng
        tronco = rng.uniform(0.1, 0.18) * tamano
        m.caja((x, y0 + tronco / 2, z), (0.05 * tamano, tronco, 0.05 * tamano), c["tronco"])
        copa = rng.choice(c["copas"])
        tipo = rng.random()
        if tipo < 0.55:  # frondoso
            r = rng.uniform(0.15, 0.23) * tamano
            m.esfera((x, y0 + tronco + r * 0.75, z), (r, r * 1.15, r), copa, rng, 0.14)
        elif tipo < 0.9:  # pino de dos pisos
            r, h = rng.uniform(0.15, 0.2) * tamano, rng.uniform(0.3, 0.42) * tamano
            oscuro = aclarar(copa, -0.18)
            m.cono((x, y0 + tronco * 0.7, z), r, h, 6, oscuro)
            m.cono((x, y0 + tronco * 0.7 + h * 0.45, z), r * 0.72, h * 0.8, 6, copa)
        else:  # arbusto
            for _ in range(3):
                ox, oz = rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)
                r = rng.uniform(0.07, 0.11) * tamano
                m.esfera((x + ox, y0 + r * 0.6, z + oz), (r, r * 0.8, r), copa, rng, 0.15)

    def _parques(self) -> None:
        """Los cuadrantes sin edificio: casi siempre parque con árboles; a veces plaza con flores."""
        m, c, rng = self.estatico, self.col, self.rng
        y0 = ALTO_ANDEN + 0.003
        for bx in range(self.n):
            for bz in range(self.n):
                for qx in (0, 1):
                    for qz in (0, 1):
                        if (bx, bz, qx, qz) in self._ocupados:
                            continue
                        cx, cz = self.centro_lote(bx, bz, qx, qz)
                        mitad = LADO_LOTE / 2 - 0.08
                        if rng.random() < self.cfg["prob_parque"]:
                            cantidad = rng.randint(*self.cfg["arboles_por_parque"])
                            for _ in range(cantidad):
                                self.arbol(cx + rng.uniform(-mitad, mitad), cz + rng.uniform(-mitad, mitad), y0)
                        else:
                            m.rect_suelo(cx - mitad, cz - mitad, cx + mitad, cz + mitad, y0 + 0.003, c["plaza"])
                            m.cilindro((cx, y0 + 0.04, cz), 0.16, 0.08, 10, c["bordillo"])
                            m.cilindro((cx, y0 + 0.05, cz), 0.13, 0.08, 10, rng.choice(c["flores"]))
                            for lado in (-1, 1):
                                m.caja((cx + lado * mitad * 0.7, y0 + 0.04, cz), (0.06, 0.03, 0.22), c["tronco"])

    def _bosque_del_borde(self) -> None:
        """Árboles en el pasto alrededor de la ciudad: la maqueta no termina en seco."""
        x0, x1, z0, z1 = self.limites()
        ciudad = (self.desplazamiento - ANCHO_CALLE, self.desplazamiento + self.lado + ANCHO_CALLE,
                  -ANCHO_CALLE, self.lado + ANCHO_CALLE)
        area = (x1 - x0) * (z1 - z0) - (ciudad[1] - ciudad[0]) * (ciudad[3] - ciudad[2])
        margen = 0.25
        for _ in range(int(area * self.cfg["densidad_bosque"])):
            x, z = self.rng.uniform(x0 + margen, x1 - margen), self.rng.uniform(z0 + margen, z1 - margen)
            if ciudad[0] < x < ciudad[1] and ciudad[2] < z < ciudad[3]:
                continue
            self.arbol(x, z, 0.0, tamano=self.rng.uniform(1.1, 1.6))

    def _faroles(self) -> None:
        """Un farol en la esquina de la manzana de cada tantas intersecciones, con su
        bombillo (se prende de noche) y su charco de luz en el piso."""
        m, c = self.estatico, self.col
        cada = self.cfg["faroles_cada"]
        radio = self.cfg["radio_halo"]
        y0 = ALTO_ANDEN
        for (nx, ny) in self.ciudad.grafo.nodes:
            if nx >= self.n or ny >= self.n or (nx + ny) % cada:
                continue
            ex, _, ez = self.mundo(nx, ny)
            px, pz = ex + ANCHO_CALLE / 2 + 0.06, ez + ANCHO_CALLE / 2 + 0.06
            m.cilindro((px, y0 + 0.35, pz), 0.022, 0.7, 6, c["poste"])
            m.caja((px - 0.07, y0 + 0.69, pz - 0.07), (0.17, 0.025, 0.025), c["poste"], giro=45)
            bx, bz = px - 0.13, pz - 0.13
            m.caja((bx, y0 + 0.69, bz), (0.1, 0.03, 0.1), c["poste"])
            self.focos.caja((bx, y0 + 0.665, bz), (0.07, 0.02, 0.07), BLANCO, sin_base=False)
            self.halos.rect_suelo(bx - radio, bz - radio, bx + radio, bz + radio, ALTO_ANDEN + 0.012, BLANCO,
                                  con_uv=True)


# ---------------------------------------------------------------- piezas móviles
def malla_trancones(maqueta: Maqueta, aristas: list[tuple], rng: random.Random) -> Malla:
    """Calles trancadas: carros en fila en los dos carriles sobre una franja roja.

    Se rehace solo cuando cambia el tráfico (unas veces al día), así que puede ser
    una malla nueva cada vez.
    """
    m = Malla()
    c = maqueta.col
    for u, v in aristas:
        (ax, _, az), (bx, _, bz) = maqueta.mundo(*u), maqueta.mundo(*v)
        largo = math.hypot(bx - ax, bz - az)
        ux, uz = (bx - ax) / largo, (bz - az) / largo
        giro = math.degrees(math.atan2(ux, uz))
        mitad = ANCHO_CALLE / 2 - 0.04
        x0, x1 = sorted((ax + ux * ANCHO_CALLE / 2, bx - ux * ANCHO_CALLE / 2))
        z0, z1 = sorted((az + uz * ANCHO_CALLE / 2, bz - uz * ANCHO_CALLE / 2))
        m.rect_suelo(x0 - mitad * abs(uz), z0 - mitad * abs(ux), x1 + mitad * abs(uz), z1 + mitad * abs(ux),
                     ALTO_RAYA + 0.002, c["trancon"])
        for carril in (-1, 1):
            lateral = carril * ANCHO_CALLE / 4
            d = ANCHO_CALLE / 2 + rng.uniform(0.15, 0.3)
            while d < largo - ANCHO_CALLE / 2 - 0.3:
                px, pz = ax + ux * d + uz * lateral, az + uz * d - ux * lateral
                _carro(m, px, pz, giro + (180 if carril < 0 else 0), rng.choice(c["carros"]), c)
                d += rng.uniform(0.38, 0.5)
    return m


def _carro(m: Malla, x: float, z: float, giro: float, color: Color, c: dict) -> None:
    m.caja((x, 0.07, z), (0.16, 0.08, 0.3), color, giro=giro)
    dx, _, dz = girar_y((0, 0, -0.02), giro)
    m.caja((x + dx, 0.14, z + dz), (0.14, 0.07, 0.16), c["vidrio_carro"], giro=giro, color_arriba=color)


def malla_moto(c: dict) -> tuple[Malla, Malla, Malla]:
    """(moto y repartidor, caja de domicilios, farola). Frente = +z.

    La caja va aparte porque su color dice el estado del repartidor, y la farola
    aparte porque brilla de noche.
    """
    cuerpo, caja, faro = Malla(), Malla(), Malla()
    llanta, rin = c["llanta"], c["rin"]
    for z in (-0.28, 0.3):
        cuerpo.cilindro((0, 0.13, z), 0.13, 0.08, 10, llanta, eje="x", color_tapa=rin)
    moto, oscuro = c["moto"], c["chaqueta"]
    cuerpo.caja((0, 0.25, 0.0), (0.2, 0.12, 0.5), moto)
    cuerpo.caja((0, 0.33, 0.26), (0.16, 0.2, 0.08), moto)                  # escudo delantero
    cuerpo.caja((0, 0.48, 0.28), (0.04, 0.2, 0.04), c["rin"])              # manubrio (columna)
    cuerpo.caja((0, 0.57, 0.27), (0.38, 0.03, 0.03), c["rin"])              # manubrio
    cuerpo.caja((0, 0.36, -0.1), (0.2, 0.06, 0.28), llanta)                # sillín
    cuerpo.caja((0, 0.42, 0.02), (0.22, 0.1, 0.26), c["pantalon"])         # piernas
    cuerpo.caja((0, 0.62, -0.06), (0.28, 0.32, 0.18), oscuro)              # torso
    for lado in (-1, 1):
        cuerpo.caja((lado * 0.15, 0.62, 0.12), (0.06, 0.06, 0.28), oscuro)  # brazos al manubrio
    cuerpo.esfera((0, 0.89, -0.04), (0.12, 0.12, 0.13), c["casco"])
    cuerpo.caja((0, 0.89, 0.07), (0.16, 0.06, 0.03), c["visera"])
    caja.caja((0, 0.74, -0.3), (0.36, 0.32, 0.32), BLANCO)
    caja.caja((0, 0.91, -0.3), (0.38, 0.03, 0.34), aclarar(BLANCO, -0.15))
    faro.caja((0, 0.4, 0.31), (0.08, 0.06, 0.02), BLANCO)
    return cuerpo, caja, faro


def malla_pin() -> Malla:
    """El alfiler de mapa que marca un pedido. Blanco: la vista lo tiñe con el estado."""
    m = Malla()
    m.cilindro((0, 0.22, 0), 0.0, 0.44, 8, BLANCO, radio_arriba=0.16)
    m.esfera((0, 0.5, 0), (0.2, 0.2, 0.2), BLANCO)
    return m


def malla_icono(nombre: str, c: dict) -> Malla:
    """Ícono 3D que gira sobre cada restaurante: se reconoce el restaurante sin leer."""
    m = Malla()
    if nombre == "Pizza":
        masa, queso, salsa = c["masa"], c["queso"], c["pepperoni"]
        punta, izq, der = (0, 0, -0.32), (-0.26, 0, 0.24), (0.26, 0, 0.24)
        alto = 0.06
        arriba = [(p[0], alto, p[2]) for p in (punta, izq, der)]
        m._triangulo(arriba[0], arriba[1], arriba[2], queso)
        m._triangulo(punta, der, izq, queso)
        for a, b in ((punta, izq), (izq, der), (der, punta)):
            m.cuadrilatero(a, b, (b[0], alto, b[2]), (a[0], alto, a[2]), masa)
        m.cilindro((0, 0.04, 0.26), 0.05, 0.56, 6, masa, eje="x")
        for px, pz in ((0, -0.08), (-0.08, 0.1), (0.09, 0.08)):
            m.cilindro((px, alto + 0.012, pz), 0.045, 0.02, 8, salsa)
    elif nombre == "Hamburguesas":
        pan = c["pan"]
        m.cilindro((0, 0.04, 0), 0.26, 0.08, 10, pan)
        m.cilindro((0, 0.11, 0), 0.28, 0.06, 10, c["lechuga"])
        m.cilindro((0, 0.17, 0), 0.25, 0.08, 10, c["carne"])
        m.caja((0, 0.22, 0), (0.36, 0.02, 0.36), c["queso"], giro=45)
        m.esfera((0, 0.27, 0), (0.27, 0.16, 0.27), pan)
    else:  # Sushi: dos maki y un nigiri
        for px in (-0.2, 0.2):
            m.cilindro((px, 0.1, 0), 0.14, 0.2, 10, c["alga"], color_tapa=c["arroz"])
            m.cilindro((px, 0.101, 0), 0.05, 0.2, 6, c["salmon"])
        m.caja((0, 0.05, 0.25), (0.3, 0.1, 0.14), c["arroz"])
        m.caja((0, 0.12, 0.25), (0.34, 0.04, 0.16), c["salmon"])
    return m


def malla_nube(rng: random.Random, color: Color) -> Malla:
    m = Malla()
    for _ in range(rng.randint(3, 6)):
        r = rng.uniform(0.6, 1.2)
        m.esfera((rng.uniform(-1.2, 1.2), rng.uniform(0, 0.4), rng.uniform(-0.6, 0.6)), (r * 1.3, r * 0.7, r),
                 color, rng, 0.1)
    return m


def malla_lluvia(rng: random.Random, limites: tuple, alto: float, gotas: int, color: Color) -> Malla:
    """Gotas como rayitas verticales, repetidas una altura más arriba: la vista baja la
    malla entera y la "enrolla" cada `alto`, así parece que llueve sin parar."""
    m = Malla()
    x0, x1, z0, z1 = limites
    largo, grosor = 0.45, 0.0075
    for _ in range(gotas):
        x, z, y = rng.uniform(x0, x1), rng.uniform(z0, z1), rng.uniform(0, alto)
        for extra in (0, alto):
            b, a = y + extra, y + extra + largo
            # dos planos en cruz: la gota se ve desde cualquier ángulo (la entidad es de doble cara)
            m.cuadrilatero((x - grosor, b, z), (x + grosor, b, z), (x + grosor, a, z), (x - grosor, a, z), color)
            m.cuadrilatero((x, b, z - grosor), (x, b, z + grosor), (x, a, z + grosor), (x, a, z - grosor), color)
    return m
