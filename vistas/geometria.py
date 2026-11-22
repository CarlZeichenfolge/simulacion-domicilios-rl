"""Constructor de mallas low poly en Python puro (sin Ursina ni Panda3D).

Por qué existe: la ciudad tiene miles de piezas (ventanas, árboles, rayas de la
calle). Crear una Entity de Ursina por pieza y luego fusionarlas tarda segundos
y deja normales raras. Aquí cada pieza se escribe directo como triángulos en
listas planas, y la vista crea UNA sola malla con ellas. Al no depender de
Ursina, se puede probar con pytest sin abrir ventana.

Convenciones:
- Eje y hacia arriba (como Ursina).
- Cada triángulo lleva sus propios vértices: sombreado plano, el look low poly.
- Ursina usa un sistema de coordenadas de mano izquierda: la cara visible de un
  triángulo es la que tiene la normal OPUESTA al producto cruz (b-a)×(c-a).
  Por eso _triangulo() recibe los vértices en el orden "matemático" y los
  escribe al revés.
"""
from __future__ import annotations

import math

Vector = tuple[float, float, float]
Color = tuple[float, float, float, float]

BLANCO: Color = (1.0, 1.0, 1.0, 1.0)
PROPORCION_AUREA = (1 + math.sqrt(5)) / 2
_VERTICES_ICOSAEDRO = [(-1, PROPORCION_AUREA, 0), (1, PROPORCION_AUREA, 0), (-1, -PROPORCION_AUREA, 0),
                       (1, -PROPORCION_AUREA, 0), (0, -1, PROPORCION_AUREA), (0, 1, PROPORCION_AUREA),
                       (0, -1, -PROPORCION_AUREA), (0, 1, -PROPORCION_AUREA), (PROPORCION_AUREA, 0, -1),
                       (PROPORCION_AUREA, 0, 1), (-PROPORCION_AUREA, 0, -1), (-PROPORCION_AUREA, 0, 1)]
_CARAS_ICOSAEDRO = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
                    (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
                    (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]


def hex_a_color(valor: str, alfa: float = 1.0) -> Color:
    valor = valor.lstrip("#")
    r, g, b = (int(valor[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alfa)


def mezclar(a: Color, b: Color, t: float) -> Color:
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def aclarar(c: Color, cuanto: float) -> Color:
    """cuanto > 0 aclara hacia blanco, < 0 oscurece hacia negro."""
    destino = BLANCO if cuanto > 0 else (0.0, 0.0, 0.0, c[3])
    r, g, b, _ = mezclar(c, destino, abs(cuanto))
    return (r, g, b, c[3])


def _restar(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cruz(a: Vector, b: Vector) -> Vector:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _normalizar(v: Vector) -> Vector:
    largo = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) or 1.0
    return (v[0] / largo, v[1] / largo, v[2] / largo)


def girar_y(p: Vector, grados: float) -> Vector:
    """Giro alrededor del eje vertical. Con 90° el frente (+z) queda mirando a +x."""
    if not grados:
        return p
    a = math.radians(grados)
    c, s = math.cos(a), math.sin(a)
    return (p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c)


class Malla:
    """Acumula triángulos con color y normal; la vista la convierte en un Mesh de Ursina."""

    def __init__(self):
        self.vertices: list[float] = []
        self.normales: list[float] = []
        self.colores: list[float] = []
        self.uvs: list[float] = []
        self.usa_uv = False
        self._uv_fijo = (0.0, 0.0)

    def fijar_uv(self, u: float, v: float) -> None:
        """Las piezas que se agreguen después llevan este (u, v) en TODOS sus vértices.

        Por qué: los carros del trancón se mueven en la tarjeta de video, no en Python.
        Cada carro guarda en su uv dónde arranca y por qué carril va, y el shader lo corre.
        """
        self._uv_fijo = (u, v)
        self.usa_uv = True

    def __len__(self) -> int:
        """Número de vértices."""
        return len(self.vertices) // 3

    def extender(self, otra: "Malla") -> None:
        self.vertices += otra.vertices
        self.normales += otra.normales
        self.colores += otra.colores
        self.uvs += otra.uvs
        self.usa_uv = self.usa_uv or otra.usa_uv

    # ------------------------------------------------------------- primitivas
    def _triangulo(self, a: Vector, b: Vector, c: Vector, color: Color, uv: tuple | None = None) -> None:
        if uv is None:
            uv = (self._uv_fijo,) * 3
        normal = _normalizar(_cruz(_restar(b, a), _restar(c, a)))
        for punto, coord in ((a, uv[0]), (c, uv[2]), (b, uv[1])):  # al revés: ver docstring del módulo
            self.vertices += punto
            self.normales += normal
            self.colores += color
            self.uvs += coord

    def cuadrilatero(self, a: Vector, b: Vector, c: Vector, d: Vector, color: Color,
                     con_uv: bool = False) -> None:
        """Cuatro esquinas en orden antihorario vistas desde el lado que se ve."""
        if con_uv:
            self.usa_uv = True
            self._triangulo(a, b, c, color, ((0, 0), (1, 0), (1, 1)))
            self._triangulo(a, c, d, color, ((0, 0), (1, 1), (0, 1)))
        else:
            self._triangulo(a, b, c, color)
            self._triangulo(a, c, d, color)

    def rect_suelo(self, x0: float, z0: float, x1: float, z1: float, y: float, color: Color,
                   con_uv: bool = False) -> None:
        """Rectángulo horizontal mirando hacia arriba (rayas, andenes, manchas de luz)."""
        self.cuadrilatero((x0, y, z0), (x0, y, z1), (x1, y, z1), (x1, y, z0), color, con_uv)

    def caja(self, centro: Vector, tamano: Vector, color: Color, giro: float = 0.0,
             color_arriba: Color | None = None, sin_base: bool = True) -> None:
        """Caja alineada a los ejes (o girada en y). Sin base por defecto: nadie mira
        los edificios desde abajo y son 2 triángulos menos por pieza."""
        cx, cy, cz = centro
        sx, sy, sz = (t / 2 for t in tamano)

        def p(x, y, z):
            gx, gy, gz = girar_y((x, y, z), giro)
            return (cx + gx, cy + gy, cz + gz)

        e = [p(-sx, -sy, -sz), p(sx, -sy, -sz), p(sx, -sy, sz), p(-sx, -sy, sz),
             p(-sx, sy, -sz), p(sx, sy, -sz), p(sx, sy, sz), p(-sx, sy, sz)]
        self.cuadrilatero(e[4], e[7], e[6], e[5], color_arriba or color)  # arriba
        self.cuadrilatero(e[3], e[2], e[6], e[7], color)  # frente (+z)
        self.cuadrilatero(e[1], e[0], e[4], e[5], color)  # atrás (-z)
        self.cuadrilatero(e[2], e[1], e[5], e[6], color)  # derecha (+x)
        self.cuadrilatero(e[0], e[3], e[7], e[4], color)  # izquierda (-x)
        if not sin_base:
            self.cuadrilatero(e[0], e[1], e[2], e[3], color)

    def techo_dos_aguas(self, centro_base: Vector, ancho: float, largo: float, alto: float,
                        color: Color, color_fachada: Color, giro: float = 0.0) -> None:
        """Prisma triangular: la cumbrera va a lo largo de z (antes de girar)."""
        cx, cy, cz = centro_base
        a, l = ancho / 2, largo / 2

        def p(x, y, z):
            gx, gy, gz = girar_y((x, y, z), giro)
            return (cx + gx, cy + gy, cz + gz)

        i0, d0, c0 = p(-a, 0, -l), p(a, 0, -l), p(0, alto, -l)
        i1, d1, c1 = p(-a, 0, l), p(a, 0, l), p(0, alto, l)
        self.cuadrilatero(i0, i1, c1, c0, color)   # agua izquierda
        self.cuadrilatero(d1, d0, c0, c1, color)   # agua derecha
        self._triangulo(i1, d1, c1, color_fachada)  # frontón delantero
        self._triangulo(d0, i0, c0, color_fachada)  # frontón trasero

    def cilindro(self, centro: Vector, radio: float, alto: float, lados: int, color: Color,
                 eje: str = "y", color_tapa: Color | None = None, radio_arriba: float | None = None) -> None:
        """Cilindro (o cono truncado si radio_arriba difiere). `centro` es el centro del cilindro."""
        r1 = radio if radio_arriba is None else radio_arriba
        h = alto / 2

        def p(x, y, z):  # construido en y; se reordena para los ejes x y z
            if eje == "x":
                x, y, z = y, -x, z
            elif eje == "z":
                x, y, z = x, z, -y
            return (centro[0] + x, centro[1] + y, centro[2] + z)

        tapa = color_tapa or color
        abajo_c, arriba_c = p(0, -h, 0), p(0, h, 0)
        for i in range(lados):
            a0 = 2 * math.pi * i / lados
            a1 = 2 * math.pi * (i + 1) / lados
            b0, b1 = p(radio * math.cos(a0), -h, radio * math.sin(a0)), p(radio * math.cos(a1), -h, radio * math.sin(a1))
            t0, t1 = p(r1 * math.cos(a0), h, r1 * math.sin(a0)), p(r1 * math.cos(a1), h, r1 * math.sin(a1))
            self.cuadrilatero(b1, b0, t0, t1, color)
            if r1 > 0:
                self._triangulo(arriba_c, t1, t0, tapa)
            self._triangulo(abajo_c, b0, b1, tapa)

    def cono(self, centro_base: Vector, radio: float, alto: float, lados: int, color: Color) -> None:
        x, y, z = centro_base
        self.cilindro((x, y + alto / 2, z), radio, alto, lados, color, radio_arriba=0.0)

    def esfera(self, centro: Vector, radios: Vector, color: Color, rng=None, rugosidad: float = 0.0) -> None:
        """Icosaedro (20 caras): la "bola" low poly de copas de árbol, nubes y cascos.
        Con `rng` y `rugosidad` cada vértice se mueve un poco para que no sean todas iguales."""
        escala = math.sqrt(1 + PROPORCION_AUREA ** 2)
        puntos = []
        for vx, vy, vz in _VERTICES_ICOSAEDRO:
            f = 1 + (rng.uniform(-rugosidad, rugosidad) if rng else 0.0)
            puntos.append((centro[0] + vx / escala * radios[0] * f, centro[1] + vy / escala * radios[1] * f,
                           centro[2] + vz / escala * radios[2] * f))
        for a, b, c in _CARAS_ICOSAEDRO:
            self._triangulo(puntos[a], puntos[b], puntos[c], color)

    def anillo_suelo(self, centro: Vector, radio_interno: float, radio_externo: float, lados: int,
                     color: Color) -> None:
        cx, cy, cz = centro
        for i in range(lados):
            a0, a1 = 2 * math.pi * i / lados, 2 * math.pi * (i + 1) / lados
            i0 = (cx + radio_interno * math.cos(a0), cy, cz + radio_interno * math.sin(a0))
            i1 = (cx + radio_interno * math.cos(a1), cy, cz + radio_interno * math.sin(a1))
            e0 = (cx + radio_externo * math.cos(a0), cy, cz + radio_externo * math.sin(a0))
            e1 = (cx + radio_externo * math.cos(a1), cy, cz + radio_externo * math.sin(a1))
            self.cuadrilatero(i0, i1, e1, e0, color)

    def ventana(self, centro: Vector, ancho: float, alto: float, normal: Vector, color: Color,
                separacion: float) -> None:
        """Rectángulo vertical pegado a una fachada; `separacion` lo despega un poco para que
        no pelee con la pared por el mismo plano (z-fighting)."""
        nx, _, nz = normal
        cx, cy, cz = centro[0] + nx * separacion, centro[1], centro[2] + nz * separacion
        tx, tz = nz, -nx  # tangente horizontal de la fachada
        a, h = ancho / 2, alto / 2
        self.cuadrilatero((cx - tx * a, cy - h, cz - tz * a), (cx + tx * a, cy - h, cz + tz * a),
                          (cx + tx * a, cy + h, cz + tz * a), (cx - tx * a, cy + h, cz - tz * a), color)
