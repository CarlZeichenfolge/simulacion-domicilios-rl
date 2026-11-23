"""Lo que se cuenta de un restaurante cuando se le da clic en la vista 3D.

Por qué aparte y sin Ursina: es pura lectura de su cocina (cola, fogones, pedidos
listos) y de lo que los habitantes aprendieron de él; así se prueba sin ventana.
"""
from __future__ import annotations

from sim.agentes.repartidor import EstadoRepartidor
from sim.pedidos import EstadoPedido
from vistas.detalle_habitante import pesos
from vistas.detalle_repartidor import minutos_de_ruta


def falta_fogon(p, t: int) -> int:
    return max(0, p.tiempo_preparacion - (t - p.aceptado_rest))


def cocinando(rest) -> list:
    """Como en Restaurante.percibir(): un cancelado sigue en la lista hasta su próximo paso, pero ya no cuenta."""
    return [p for p in rest.cocina if p.estado == EstadoPedido.EN_PREPARACION]


def esperando_fogon(rest) -> list:
    return [p for p in rest.cola if p.estado == EstadoPedido.CREADO]


def ocupacion(rest) -> tuple[int, int]:
    """(fogones ocupados, fogones)."""
    return len(cocinando(rest)), rest.capacidad


def moto(p, t: int) -> str:
    """Quién se lleva el pedido y cuándo llega por él."""
    r = p.repartidor
    if r is None:
        return "sin moto todavía"
    if r.estado == EstadoRepartidor.YENDO_A_RESTAURANTE and r.ruta:
        return f"repartidor {r.unique_id} llega en ~{minutos_de_ruta(r):.0f} min"
    return f"repartidor {r.unique_id} ya está en la puerta"


def en_cocina(rest) -> list[str]:
    t = rest.t
    return [f"#{p.id} hab. {p.habitante.unique_id}  lleva {t - p.aceptado_rest} de {p.tiempo_preparacion} min  "
            f"· {moto(p, t)}" for p in sorted(cocinando(rest), key=lambda p: falta_fogon(p, t))]


def en_cola(rest) -> list[str]:
    """La cola en orden FIFO. Al primero se le dice cuándo se libera un fogón."""
    t = rest.t
    fogones = cocinando(rest)
    libres = rest.capacidad - len(fogones)
    proximos = sorted(falta_fogon(p, t) for p in fogones)
    lineas = []
    for i, p in enumerate(esperando_fogon(rest)):
        if i < libres:
            cuando = "entra al fogón en el próximo minuto"
        elif i - libres < len(proximos):
            cuando = f"entra al fogón en ~{proximos[i - libres]} min"
        else:
            cuando = f"puesto {i + 1} en la fila"
        lineas.append(f"#{p.id} hab. {p.habitante.unique_id}  espera hace {p.edad(t)} min · {cuando}")
    return lineas


def listos(rest) -> list[str]:
    """Pedidos ya cocinados que siguen en el mostrador: la comida espera a la moto."""
    t = rest.t
    maximo = rest.model.cfg["pedidos"]["espera_maxima"]
    return [f"#{p.id} hab. {p.habitante.unique_id}  listo hace {t - p.listo} min · {moto(p, t)} · "
            f"se cancela en {max(0, maximo - p.edad(t))} min"
            for p in rest.model.pedidos_activos
            if p.restaurante is rest and p.estado == EstadoPedido.LISTO]


def hoy(rest) -> str:
    """Pedidos recibidos en el día que va y la hora con más pedidos."""
    dia = rest.model.reloj.dia
    por_hora = {h: n for (d, h), n in rest.historial_demanda.items() if d == dia}
    total = sum(por_hora.values())
    if not total:
        return "Hoy: ningún pedido todavía"
    pico = max(por_hora, key=por_hora.get)
    return f"Hoy: {total} pedidos · {rest.preparados} preparados en total · hora pico {pico:02d}:00 ({por_hora[pico]})"


def opinion(rest) -> str:
    """Lo que los habitantes aprendieron de él con su bandit (satisfacción esperada al pedirle)."""
    gente = [h for h in rest.model.habitantes if rest.nombre in getattr(h.politica, "q", {})]
    if not gente:
        return ""
    promedio = sum(h.politica.q[rest.nombre] for h in gente) / len(gente)
    favorito = sum(1 for h in gente if max(h.politica.q, key=h.politica.q.get) == rest.nombre)
    return (f"Los habitantes esperan {promedio:+.1f} de satisfacción al pedirle; "
            f"es el favorito de {favorito} de {len(gente)}")


def ficha(rest) -> str:
    return (f"Precio medio {pesos(rest.precio_medio)} · cocina en {rest.prep_media} ± {rest.prep_desv} min · "
            f"{rest.capacidad} fogones")


def espera_nueva(rest) -> str:
    return f"Un pedido nuevo saldría en ~{rest.estimar_espera():.0f} min (su modelo interno de la cocina)"


def terminados(rest) -> list:
    """Sus pedidos ya cerrados (entregados o cancelados), el más reciente primero."""
    return [p for p in reversed(rest.model.pedidos) if p.restaurante is rest and p.terminado]


def plata(rest) -> str:
    """Ventas: la comida de los pedidos entregados. El cliente solo paga lo que le llega,
    así que un cancelado es comida hecha (o empezada) que nadie paga."""
    vendido = sum(p.precio for p in terminados(rest) if p.estado == EstadoPedido.ENTREGADO)
    return f"{pesos(vendido)} vendidos (solo cobra los que llegan al cliente)"


def resumen_pedidos(rest) -> str:
    pedidos = terminados(rest)
    entregados = sum(p.estado == EstadoPedido.ENTREGADO for p in pedidos)
    return f"{entregados} entregados · {len(pedidos) - entregados} cancelados"


def lista_pedidos(rest) -> list[str]:
    lineas = []
    for p in terminados(rest):
        quien = f"#{p.id} hab. {p.habitante.unique_id}"
        if p.estado == EstadoPedido.ENTREGADO:
            lineas.append(f"{quien}: llegó en {p.tiempo_entrega} min (repartidor {p.repartidor.unique_id}) · "
                          f"{pesos(p.precio)}")
        else:
            lineas.append(f"{quien}: cancelado a los {p.cancelado - p.creado} min, no lo cobró")
    return lineas
