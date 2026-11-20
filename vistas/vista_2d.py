"""Vista 2D con Pygame. Solo LEE el modelo y lo dibuja; nunca cambia su estado.

Controles:
    ESPACIO  pausa / sigue
    ↑ / ↓    más o menos minutos simulados por cuadro
    D        avanza un solo minuto (en pausa)
    ESC      salir
"""
from __future__ import annotations

import pygame

from sim.agentes import EstadoRepartidor
from sim.model import CiudadModel
from sim.pedidos import EstadoPedido

FONDO = (24, 26, 31)
CALLE = (70, 74, 84)
PANEL = (34, 37, 44)
TEXTO = (225, 228, 235)
TEXTO_SUAVE = (150, 155, 165)
EDIFICIO = {"casa": (150, 120, 90), "oficina": (95, 115, 145), "universidad": (140, 95, 170)}
RESTAURANTE = {"Pizza": (220, 70, 60), "Hamburguesas": (235, 160, 40), "Sushi": (40, 180, 160)}
REPARTIDOR = {EstadoRepartidor.LIBRE: (200, 200, 200),
              EstadoRepartidor.YENDO_A_RESTAURANTE: (255, 215, 0),
              EstadoRepartidor.YENDO_A_CLIENTE: (80, 200, 255)}
PEDIDO = {EstadoPedido.CREADO: (180, 180, 180), EstadoPedido.EN_PREPARACION: (235, 160, 40),
          EstadoPedido.LISTO: (120, 220, 120), EstadoPedido.EN_CAMINO: (80, 200, 255)}


class Vista2D:
    def __init__(self, modelo: CiudadModel):
        self.modelo = modelo
        cfg = modelo.cfg["vista_2d"]
        self.ancho, self.alto = cfg["ancho"], cfg["alto"]
        self.ticks_por_cuadro = cfg["ticks_por_cuadro"]
        self.fps = cfg["fps"]
        self.pausa = False
        self.margen = 40
        self.lado_mapa = self.alto - 2 * self.margen
        self.celda = self.lado_mapa / modelo.ciudad.tamano

    # ------------------------------------------------------------- coordenadas
    def a_pantalla(self, x: float, y: float) -> tuple[int, int]:
        # y crece hacia arriba en la ciudad y hacia abajo en la pantalla
        return (int(self.margen + x * self.celda),
                int(self.margen + self.lado_mapa - y * self.celda))

    # ------------------------------------------------------------------- bucle
    def correr(self) -> None:
        pygame.init()
        pantalla = pygame.display.set_mode((self.ancho, self.alto))
        pygame.display.set_caption("Ciudad de domicilios")
        self.fuente = pygame.font.SysFont("consolas", 16)
        self.fuente_grande = pygame.font.SysFont("consolas", 22, bold=True)
        self.fuente_chica = pygame.font.SysFont("consolas", 12)
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
                        self.ticks_por_cuadro = min(60, self.ticks_por_cuadro * 2)
                    elif evento.key == pygame.K_DOWN:
                        self.ticks_por_cuadro = max(1, self.ticks_por_cuadro // 2)
                    elif evento.key == pygame.K_d and self.modelo.running:
                        self.modelo.step()
            if not self.pausa:
                for _ in range(self.ticks_por_cuadro):
                    if self.modelo.running:
                        self.modelo.step()
            self.dibujar(pantalla)
            pygame.display.flip()
            reloj.tick(self.fps)

    # ----------------------------------------------------------------- dibujo
    def dibujar(self, pantalla) -> None:
        pantalla.fill(FONDO)
        self._calles(pantalla)
        self._edificios(pantalla)
        self._pedidos(pantalla)
        self._habitantes(pantalla)
        self._repartidores(pantalla)
        if self.modelo.eventos.lloviendo:
            self._lluvia(pantalla)
        self._panel(pantalla)

    def _calles(self, pantalla) -> None:
        for u, v, datos in self.modelo.ciudad.grafo.edges(data=True):
            factor = datos["factor_trancon"]
            if factor > 1:  # más rojo y más grueso cuanto peor el trancón
                intensidad = min(1.0, (factor - 1) / 2)
                color = (int(120 + 135 * intensidad), 60, 60)
                grosor = 4 + int(3 * intensidad)
            else:
                color, grosor = CALLE, 4
            pygame.draw.line(pantalla, color, self.a_pantalla(*u), self.a_pantalla(*v), grosor)

    def _edificios(self, pantalla) -> None:
        lado = int(self.celda * 0.32)
        for nodo, tipo in self.modelo.ciudad.edificios.items():
            if tipo == "restaurante":
                continue
            x, y = self.a_pantalla(*nodo)
            pygame.draw.rect(pantalla, EDIFICIO[tipo], (x - lado // 2, y - lado // 2, lado, lado), border_radius=3)
        for nombre, restaurante in self.modelo.restaurantes.items():
            x, y = self.a_pantalla(*restaurante.nodo)
            lado = int(self.celda * 0.55)
            pygame.draw.rect(pantalla, RESTAURANTE[nombre], (x - lado // 2, y - lado // 2, lado, lado), border_radius=6)
            etiqueta = self.fuente.render(nombre[0], True, (20, 20, 20))
            pantalla.blit(etiqueta, etiqueta.get_rect(center=(x, y)))
            cola = len(restaurante.cola) + len(restaurante.cocina)
            if cola:
                texto = self.fuente_chica.render(f"{cola}", True, TEXTO)
                pantalla.blit(texto, (x + lado // 2 + 2, y - lado // 2 - 4))

    def _pedidos(self, pantalla) -> None:
        for pedido in self.modelo.pedidos_activos:
            if pedido.terminado:
                continue
            x, y = self.a_pantalla(*pedido.destino)
            color = PEDIDO[pedido.estado]
            pygame.draw.circle(pantalla, color, (x, y), 9, 2)
            # Arco que se llena con la edad: qué tan cerca está de cancelarse.
            fraccion = min(1.0, pedido.edad(self.modelo.reloj.tick) / self.modelo.cfg["pedidos"]["espera_maxima"])
            if fraccion > 0.66:
                pygame.draw.circle(pantalla, (230, 60, 60), (x, y), 12, 2)
            if pedido.repartidor is None:
                pygame.draw.line(pantalla, (90, 90, 100), self.a_pantalla(*pedido.restaurante.nodo), (x, y), 1)

    def _habitantes(self, pantalla) -> None:
        umbral = self.modelo.cfg["habitantes"]["umbral_hambre"]
        por_nodo: dict[tuple, int] = {}
        for h in self.modelo.habitantes:
            nodo = h.ubicacion
            k = por_nodo.get(nodo, 0)
            por_nodo[nodo] = k + 1
            x, y = self.a_pantalla(*nodo)
            x += int(self.celda * 0.25) + (k % 4) * 7
            y += int(self.celda * 0.25) + (k // 4) * 7
            if h.actividad == "dormir":
                color = (80, 80, 110)
            elif h.pedido is not None:
                color = (255, 150, 50)
            elif h.comida_en_curso is not None:
                color = (120, 200, 120)
            elif h.hambre >= umbral:
                color = (240, 80, 80)
            else:
                color = (200, 200, 200)
            pygame.draw.circle(pantalla, color, (x, y), 3)

    def _repartidores(self, pantalla) -> None:
        for r in self.modelo.repartidores:
            x, y = self.a_pantalla(*r.posicion_continua())
            if r.ruta:  # ruta planeada por A*
                puntos = [(x, y)] + [self.a_pantalla(*n) for n in r.ruta]
                pygame.draw.lines(pantalla, REPARTIDOR[r.estado], False, puntos, 1)
            pygame.draw.circle(pantalla, (20, 20, 20), (x, y), 10)
            pygame.draw.circle(pantalla, REPARTIDOR[r.estado], (x, y), 8)
            etiqueta = self.fuente_chica.render(str(r.unique_id), True, (20, 20, 20))
            pantalla.blit(etiqueta, etiqueta.get_rect(center=(x, y)))

    def _lluvia(self, pantalla) -> None:
        capa = pygame.Surface((self.alto, self.alto), pygame.SRCALPHA)
        capa.fill((60, 90, 160, 45))
        rng_visual = self.modelo.reloj.tick * 7919  # gotas deterministas, sin tocar el rng del modelo
        for i in range(120):
            gx = (rng_visual + i * 7621) % self.alto
            gy = (rng_visual * 3 + i * 3571) % self.alto
            pygame.draw.line(capa, (160, 190, 255, 120), (gx, gy), (gx - 3, gy + 9), 1)
        pantalla.blit(capa, (0, 0))

    def _panel(self, pantalla) -> None:
        m = self.modelo
        x0 = self.alto
        pygame.draw.rect(pantalla, PANEL, (x0, 0, self.ancho - x0, self.alto))
        y = 20

        def linea(texto, fuente=None, color=TEXTO, salto=22):
            nonlocal y
            pantalla.blit((fuente or self.fuente).render(texto, True, color), (x0 + 20, y))
            y += salto

        linea(m.reloj.texto(), self.fuente_grande, salto=32)
        clima = "Lluvia" if m.eventos.lloviendo else "Seco"
        trafico = "Hora pico (trancón)" if m.eventos.trancon_activo else "Tráfico normal"
        linea(f"{clima}  |  {trafico}", color=TEXTO_SUAVE, salto=30)
        linea(f"Política: {m.cfg['repartidores']['politica']}", salto=30)
        linea("Pedidos", self.fuente_grande, salto=28)
        linea(f"  creados     {m.conteo['creados']:>5}")
        linea(f"  entregados  {m.conteo['entregados']:>5}")
        linea(f"  cancelados  {m.conteo['cancelados']:>5}")
        linea(f"  activos     {len(m.pedidos_activos):>5}", salto=30)
        entregados = [p.tiempo_entrega for p in m.pedidos if p.tiempo_entrega is not None]
        if entregados:
            linea(f"  tiempo prom {sum(entregados) / len(entregados):5.1f} min", salto=30)
        if m.satisfacciones:
            linea(f"  satisfacción {sum(m.satisfacciones) / len(m.satisfacciones):5.2f}", salto=30)
        linea("Repartidores", self.fuente_grande, salto=28)
        for r in m.repartidores:
            linea(f"  #{r.unique_id:<3} {r.estado.value:<20}", color=REPARTIDOR[r.estado], salto=18)
            linea(f"       ${(r.ganancias - r.costos) / 1000:6.1f}k  {r.entregas} entregas",
                  color=TEXTO_SUAVE, salto=22)
        y = self.alto - 150
        linea("Habitantes", self.fuente_chica, TEXTO_SUAVE, salto=16)
        for color, texto in [((240, 80, 80), "con hambre"), ((255, 150, 50), "esperando domicilio"),
                             ((120, 200, 120), "cocinando / comiendo afuera"),
                             ((80, 80, 110), "durmiendo")]:
            pygame.draw.circle(pantalla, color, (x0 + 26, y + 7), 4)
            pantalla.blit(self.fuente_chica.render(texto, True, TEXTO_SUAVE), (x0 + 38, y))
            y += 16
        y = self.alto - 70
        linea(f"x{self.ticks_por_cuadro} min/cuadro  {'PAUSA' if self.pausa else ''}", color=TEXTO_SUAVE)
        linea("ESPACIO pausa  ↑↓ velocidad  D paso  ESC salir", self.fuente_chica, TEXTO_SUAVE)
