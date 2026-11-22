"""Antes vs después: la misma ciudad, el mismo día, sin entrenar y entrenado, lado a lado.

    python main.py --vista duelo --politica dqn          # la red de resultados/red_dqn.pt
    python main.py --vista duelo                         # la tabla Q de resultados/tabla_q.npz

Izquierda: la MISMA política (misma red o misma tabla) recién creada, sin un
solo día de entrenamiento: decide con sus pesos iniciales (o con la tabla en
ceros). Derecha: esa política después de entrenar, congelada (sin explorar ni
aprender). Lo único distinto entre las dos es lo aprendido.
Abajo, dos curvas que se dibujan en vivo: cancelaciones y recompensa acumuladas.

Las dos ciudades arrancan con la misma semilla (mismo mapa, mismos habitantes,
mismo clima inicial). Como cada política consume el azar distinto, los días
se separan poco a poco; la comparación justa con muchos días está en los
experimentos. Esto es para VER la diferencia, no para medirla.

Controles: ESPACIO pausa · ↑/↓ velocidad · ESC salir
"""
from __future__ import annotations

import math

import pygame

from sim.model import CiudadModel
from vistas.velocidad import Velocidad
from vistas.vista_2d import FONDO, PANEL, TEXTO, TEXTO_SUAVE, Vista2D

# Azul = sin entrenar, ámbar = entrenada. Tonos claros porque el fondo es oscuro.
COLOR_ANTES = (110, 160, 240)
COLOR_DESPUES = (240, 180, 40)
GRILLA = (55, 59, 68)


class _Mapa(Vista2D):
    """Una Vista2D que solo dibuja el mapa en una superficie cuadrada (sin panel)."""

    def __init__(self, modelo: CiudadModel, lado: int, fuentes: tuple):
        super().__init__(modelo)
        self.alto = lado
        self.margen = lado // 20
        self.lado_mapa = lado - 2 * self.margen
        self.celda = self.lado_mapa / modelo.ciudad.tamano
        self.fuente, self.fuente_grande, self.fuente_chica = fuentes
        self.superficie = pygame.Surface((lado, lado))

    def dibujar_mapa(self) -> pygame.Surface:
        s = self.superficie
        s.fill(FONDO)
        self._calles(s)
        self._edificios(s)
        self._pedidos(s)
        self._habitantes(s)
        self._repartidores(s)
        if self.modelo.eventos.lloviendo:
            self._lluvia(s)
        return s


def _paso_redondo(rango: float) -> float:
    """Separación de la grilla en 1, 2 o 5 × potencia de 10, para unas 4 líneas: ejes legibles."""
    crudo = rango / 4
    base = 10 ** math.floor(math.log10(crudo))
    return next(m * base for m in (1, 2, 5, 10) if m * base >= crudo)


def _recompensa(modelo: CiudadModel) -> float:
    return sum(r.recompensa_acumulada for r in modelo.repartidores)


class VistaDuelo:
    def __init__(self, antes: CiudadModel, despues: CiudadModel, titulo_antes: str, titulo_despues: str):
        self.antes, self.despues = antes, despues
        self.titulos = (titulo_antes, titulo_despues)
        cfg = despues.cfg["vista_duelo"]
        self.ancho, self.alto = cfg["ancho"], cfg["alto"]
        self.lado = cfg["lado_mapa"]
        self.velocidad = Velocidad(despues.cfg["vistas"])
        self.fps = cfg["fps"]
        self.muestrear_cada = cfg["muestrear_cada"]
        self.pausa = False
        # Series para las curvas: (minuto, cancelados, recompensa) de cada ciudad, desde donde arrancó la demo.
        self.series = {clave: [(m.reloj.tick, m.conteo["cancelados"], _recompensa(m))]
                       for clave, m in (("antes", antes), ("despues", despues))}

    def correr(self) -> None:
        pygame.init()
        pantalla = pygame.display.set_mode((self.ancho, self.alto))
        pygame.display.set_caption("Antes vs después de entrenar")
        fuentes = (pygame.font.SysFont("consolas", 16), pygame.font.SysFont("consolas", 22, bold=True),
                   pygame.font.SysFont("consolas", 12))
        self.fuente, self.fuente_grande, self.fuente_chica = fuentes
        self.mapas = {"antes": _Mapa(self.antes, self.lado, fuentes), "despues": _Mapa(self.despues, self.lado, fuentes)}
        reloj = pygame.time.Clock()
        while True:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT or (evento.type == pygame.KEYDOWN and evento.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_SPACE:
                        self.pausa = not self.pausa
                    elif evento.key == pygame.K_UP:
                        self.velocidad.subir()
                    elif evento.key == pygame.K_DOWN:
                        self.velocidad.bajar()
            dt = reloj.tick(self.fps) / 1000
            if not self.pausa:
                self._avanzar(self.velocidad.pasos(dt))
            self._dibujar(pantalla)
            pygame.display.flip()

    def _avanzar(self, minutos: int) -> None:
        """Avanza las dos ciudades los mismos minutos (mismo reloj) y anota las curvas."""
        for clave, mapa in self.mapas.items():
            for i in range(minutos):
                modelo = mapa.modelo
                if not modelo.running:
                    break
                mapa.avanzar(1)  # de a un minuto para no perder ningún punto de las curvas
                if modelo.reloj.tick % self.muestrear_cada == 0:
                    self.series[clave].append((modelo.reloj.tick, modelo.conteo["cancelados"], _recompensa(modelo)))
            mapa.fraccion = self.velocidad.fraccion

    # ----------------------------------------------------------------- dibujo
    def _dibujar(self, pantalla) -> None:
        pantalla.fill(PANEL)
        mitad = self.ancho // 2
        for i, (clave, modelo, titulo, color) in enumerate((
                ("antes", self.antes, self.titulos[0], COLOR_ANTES),
                ("despues", self.despues, self.titulos[1], COLOR_DESPUES))):
            x0 = i * mitad + (mitad - self.lado) // 2
            self._encabezado(pantalla, modelo, titulo, color, x0)
            pantalla.blit(self.mapas[clave].dibujar_mapa(), (x0, 84))
        alto_graf = self.alto - self.lado - 84 - 44
        y_graf = 84 + self.lado + 12
        ancho_graf = mitad - 60
        self._grafica(pantalla, (40, y_graf, ancho_graf, alto_graf), "Pedidos cancelados (acumulado)", 1)
        self._grafica(pantalla, (mitad + 20, y_graf, ancho_graf, alto_graf),
                      "Recompensa de los repartidores (acumulada, miles de $)", 2)
        pie = f"{self.despues.reloj.texto()}   {self.velocidad.texto()}  {'PAUSA' if self.pausa else ''}" \
              "   ESPACIO pausa  ↑↓ velocidad  ESC salir"
        pantalla.blit(self.fuente_chica.render(pie, True, TEXTO_SUAVE), (40, self.alto - 18))

    def _encabezado(self, pantalla, m: CiudadModel, titulo: str, color, x0: int) -> None:
        pygame.draw.rect(pantalla, color, (x0, 14, 6, 60), border_radius=3)
        pantalla.blit(self.fuente_grande.render(titulo, True, TEXTO), (x0 + 16, 12))
        creados = max(1, m.conteo["creados"])
        clima = "lluvia" if m.eventos.lloviendo else "seco"
        texto = (f"entregados {m.conteo['entregados']:>3}   cancelados {m.conteo['cancelados']:>3} "
                 f"({m.conteo['cancelados'] / creados:4.0%})   recompensa {_recompensa(m):7.1f}   {clima}")
        pantalla.blit(self.fuente.render(texto, True, TEXTO_SUAVE), (x0 + 16, 46))

    def _grafica(self, pantalla, caja, titulo: str, columna: int) -> None:
        """Dos líneas finas (antes / después), grilla tenue, leyenda y valor final al borde."""
        x, y, ancho, alto = caja
        pantalla.blit(self.fuente.render(titulo, True, TEXTO), (x, y))
        # Leyenda en la fila del título, fuera del área de datos (la identidad no depende
        # solo del color: también va el nombre junto al final de cada línea).
        lx = x + ancho - 170
        for color, nombre in ((COLOR_ANTES, "antes"), (COLOR_DESPUES, "después")):
            pygame.draw.line(pantalla, color, (lx, y + 9), (lx + 18, y + 9), 2)
            pantalla.blit(self.fuente_chica.render(nombre, True, TEXTO_SUAVE), (lx + 24, y + 2))
            lx += 85
        y, alto = y + 34, alto - 34
        valores = [p[columna] for s in self.series.values() for p in s]
        paso = _paso_redondo(max(1.0, max(valores)) - min(0.0, min(valores)))
        minimo = math.floor(min(0.0, min(valores)) / paso) * paso
        maximo = math.ceil(max(1.0, max(valores)) / paso) * paso
        duracion = self.despues.duracion
        nivel = minimo
        while nivel <= maximo + 1e-9:  # grilla horizontal recesiva en valores redondos
            gy = y + alto - (nivel - minimo) / (maximo - minimo) * alto
            pygame.draw.line(pantalla, GRILLA, (x, gy), (x + ancho, gy), 1)
            etiqueta = self.fuente_chica.render(f"{nivel:.0f}", True, TEXTO_SUAVE)
            pantalla.blit(etiqueta, (x + ancho + 4, gy - 7))
            nivel += paso
        if minimo < 0:  # la línea del cero, un poco más marcada
            cy = y + alto - (0 - minimo) / (maximo - minimo) * alto
            pygame.draw.line(pantalla, TEXTO_SUAVE, (x, cy), (x + ancho, cy), 1)

        def a_pixel(minuto, valor):
            return (x + minuto / duracion * ancho, y + alto - (valor - minimo) / (maximo - minimo) * alto)

        etiquetas = []
        for clave, color, nombre in (("antes", COLOR_ANTES, "antes"), ("despues", COLOR_DESPUES, "después")):
            puntos = [a_pixel(p[0], p[columna]) for p in self.series[clave]]
            if len(puntos) > 1:
                pygame.draw.lines(pantalla, color, False, puntos, 2)
            px, py = puntos[-1]
            pygame.draw.circle(pantalla, color, (int(px), int(py)), 4)
            superficie = self.fuente_chica.render(f"{nombre} {self.series[clave][-1][columna]:.0f}", True, TEXTO)
            if px + 10 + superficie.get_width() <= x + ancho:
                etiquetas.append([py - 7, px + 10, superficie])  # a la derecha del punto
            else:  # sin espacio a la derecha: encima de la línea, terminando en el punto
                etiquetas.append([py - 20, px - superficie.get_width(), superficie])
        # Si los dos finales quedan muy cerca, se separan las etiquetas para que no se pisen.
        etiquetas.sort(key=lambda e: e[0])
        if etiquetas[1][0] - etiquetas[0][0] < 15:
            centro = (etiquetas[0][0] + etiquetas[1][0]) / 2
            etiquetas[0][0], etiquetas[1][0] = centro - 8, centro + 8
        for ey, ex, superficie in etiquetas:
            pantalla.blit(superficie, (ex, ey))
