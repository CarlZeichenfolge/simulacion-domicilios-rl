"""Lo que se cuenta de un repartidor cuando se le da clic en la vista 3D.

Por qué aparte y sin Ursina: igual que detalle_habitante, es pura lectura del agente
(dónde va, qué comparó, cuánto le pagaron) y así se prueba con pytest sin ventana.
Nada de aquí gasta azar ni hace aprender a la política: solo mira.
"""
from __future__ import annotations

from sim.agentes.repartidor import EstadoRepartidor
from sim.pedidos import EstadoPedido
from vistas.detalle_habitante import NOMBRE_ESTADO_PEDIDO, hora, pesos

QUE_ES_Q = "Q = lo que espera ganar de aquí en adelante (miles de $)."


def politica(r) -> str:
    return r.model.cfg["repartidores"]["politica"]


def espera_maxima(r) -> int:
    return r.model.cfg["pedidos"]["espera_maxima"]


def minutos_de_ruta(r) -> float:
    """Minutos que le faltan por la ruta que ya planeó, con el tráfico y la lluvia de ahora."""
    if not r.ruta:
        return 0.0
    ciudad, fv = r.model.ciudad, r.model.eventos.factor_velocidad
    nodos = [r.nodo] + r.ruta
    total = sum(ciudad.tiempo_arista(a, b, fv) for a, b in zip(nodos, nodos[1:]))
    return total - r.progreso * ciudad.tiempo_arista(r.nodo, r.ruta[0], fv)


def cuadras_de_ruta(r) -> float:
    return max(0.0, len(r.ruta) - r.progreso)


def cocina(pedido, t: int) -> str:
    """Cómo va la comida en el restaurante (eso lo sabe la cocina, no el repartidor)."""
    if pedido.estado == EstadoPedido.CREADO:
        return "en la cola de la cocina, sin fogón todavía"
    if pedido.estado == EstadoPedido.EN_PREPARACION:
        falta = max(0, pedido.tiempo_preparacion - (t - pedido.aceptado_rest))
        return f"en el fogón, le faltan ~{falta} min"
    return NOMBRE_ESTADO_PEDIDO[pedido.estado.value]


def estado(r) -> tuple[str, str]:
    """(frase, clave). La clave escoge el color (la misma del aro de la moto)."""
    p = r.pedido
    if r.estado == EstadoRepartidor.YENDO_A_RESTAURANTE:
        if not r.ruta:
            return f"En la puerta de {p.restaurante.nombre}, esperando el pedido #{p.id}", r.estado.value
        return (f"Va a {p.restaurante.nombre} a recoger el pedido #{p.id} "
                f"(habitante {p.habitante.unique_id})", r.estado.value)
    if r.estado == EstadoRepartidor.YENDO_A_CLIENTE:
        return (f"Lleva el pedido #{p.id} de {p.restaurante.nombre} "
                f"al habitante {p.habitante.unique_id}", r.estado.value)
    if r.espera_hasta is not None:
        return f"Rechazó los pedidos: espera {max(0, r.espera_hasta - r.t)} min y vuelve a decidir", "libre"
    disponibles = len(r.model.pedidos_disponibles())
    if disponibles:
        return f"Libre: en el próximo minuto escoge entre {disponibles} pedidos", "libre"
    return "Libre y quieto: no hay pedidos sin repartidor", "libre"


def trayecto(r) -> list[str]:
    """Dónde va, cuánto le falta y si el pedido alcanza a llegar antes de cancelarse."""
    p = r.pedido
    if p is None:
        return []
    lineas = []
    edad, maximo = p.edad(r.t), espera_maxima(r)
    if r.ruta:
        lluvia = " (con lluvia las calles son más lentas)" if r.model.eventos.lloviendo else ""
        cuadras = cuadras_de_ruta(r)
        falta = "menos de 1 cuadra" if cuadras < 1 else f"{cuadras:.0f} cuadras"
        lineas.append(f"Le faltan {falta}, ~{max(1, round(minutos_de_ruta(r)))} min por la "
                      f"ruta más rápida (A*){lluvia}")
    if r.estado == EstadoRepartidor.YENDO_A_RESTAURANTE:
        lineas.append(f"La comida: {cocina(p, r.t)}")
    llegada = edad + minutos_de_ruta(r)
    if r.estado == EstadoRepartidor.YENDO_A_CLIENTE:
        veredicto = "llega a tiempo" if llegada < maximo else "no alcanza: se va a cancelar"
        lineas.append(f"El pedido lleva {edad} de {maximo} min; llegaría al minuto {llegada:.0f}: {veredicto}")
    else:
        lineas.append(f"El pedido lleva {edad} de {maximo} min (se cancela en {max(0, maximo - edad)} min)")
    return lineas


def que_sigue(r) -> str:
    """Lo próximo que va a pasar y cuánto le cuesta o le paga (la recompensa que va a ver)."""
    p = r.pedido
    if r.estado == EstadoRepartidor.YENDO_A_RESTAURANTE:
        cuadras = r.model.ciudad.distancia_cuadras(p.restaurante.nodo, p.destino)
        return (f"Luego recoge y lo lleva {cuadras} cuadras al cliente. Cobra {pesos(p.pago_repartidor)} "
                f"menos {pesos(r.costo_por_minuto)} por minuto trabajado.")
    if r.estado == EstadoRepartidor.YENDO_A_CLIENTE:
        # La última decisión es la de aceptar este pedido: desde ahí corre el costo por minuto.
        minutos = r.t - r.ultima_decision["tick"] + minutos_de_ruta(r)
        neto = p.pago_repartidor - r.costo_por_minuto * minutos
        return (f"Al entregar: {pesos(p.pago_repartidor)} − {pesos(r.costo_por_minuto)}×{minutos:.0f} min ≈ "
                f"{pesos(neto)} (recompensa {neto * r.escala:+.1f})."
                f" Si se cancela pierde {pesos(r.penalizacion_cancelado)} más el tiempo. Después queda libre.")
    if r.espera_hasta is not None:
        costo = r.costo_espera_por_minuto * r.duracion_espera + r.penalizacion_rechazo
        return (f"Esperar le cuesta {pesos(costo)} ({pesos(r.penalizacion_rechazo)} son por rechazar); "
                f"al terminar decide otra vez.")
    return "Cuando aparezca un pedido decide si lo acepta o espera."


def opciones(r, maximo: int) -> tuple[list[dict], int]:
    """Las opciones de la última decisión, la escogida primero y luego por valor.

    Cada una: {"texto", "valor" (None en reglas fijas), "escogida"}. También devuelve
    cuántas quedaron por fuera del máximo. "Esperar" siempre aparece: en las
    políticas que aprenden es una opción más, y su valor dice por qué no se escogió.
    """
    d = r.ultima_decision
    if d is None:
        return [], 0
    percepcion, accion = d["percepcion"], d["accion"]
    valores = r.politica.valorar(percepcion)
    filas = []
    for i, info in enumerate(percepcion["pedidos"]):
        p = info["pedido"]
        filas.append({"texto": f"#{p.id} {p.restaurante.nombre}  {info['distancia']}+{info['distancia_entrega']} "
                               f"cuadras  {info['edad']} min  {pesos(info['pago'])}"
                               f"{'  listo' if info['listo'] else ''}",
                      "valor": None if valores is None else valores[i],
                      "escogida": accion["tipo"] == "aceptar" and accion["pedido"] is p})
    filas.append({"texto": "Esperar (rechazar todos)", "valor": None if valores is None else valores[-1],
                  "escogida": accion["tipo"] == "esperar"})
    filas.sort(key=lambda f: (not f["escogida"], -(f["valor"] if f["valor"] is not None else 0.0)))
    return filas[:maximo], max(0, len(filas) - maximo)


def exploro(r) -> bool:
    """¿La última elección NO fue la de mayor valor? Entonces fue azar (exploración)."""
    d = r.ultima_decision
    if d is None:
        return False
    filas, _ = opciones(r, len(d["percepcion"]["pedidos"]) + 1)
    valores = [f["valor"] for f in filas]
    if None in valores:
        return False
    return filas[0]["escogida"] and filas[0]["valor"] < max(valores)


def encabezado_decision(r) -> str:
    d = r.ultima_decision
    if d is None:
        return "(todavía no ha tenido que escoger)"
    p = d["percepcion"]
    lluvia = ", lloviendo" if p["lluvia"] else ""
    return f"({hora(d['minuto_del_dia'])}{lluvia}) vio {len(p['pedidos'])} pedido{'' if len(p['pedidos']) == 1 else 's'} sin repartidor"


def frase_resultado(r) -> str:
    res = r.ultimo_resultado
    if res is None:
        return "(todavía no ha terminado ninguna acción)"
    cuando = hora(r.model.reloj.minuto_inicio + res["tick"])
    valor = f"{pesos(res['pesos'])}, recompensa {res['recompensa']:+.2f}"
    if res["motivo"] == "entrego":
        return f"({cuando}) entregó #{res['pedido'].id} en {res['minutos']} min: {valor}"
    if res["motivo"] == "cancelado":
        return f"({cuando}) se le canceló #{res['pedido'].id} tras {res['minutos']} min: {valor}"
    return f"({cuando}) esperó {res['minutos']} min habiendo pedidos (rechazo): {valor}"


def plata(r) -> str:
    """Gastos = gasolina y tiempo por minuto, espera quieto y multas por rechazar."""
    return f"{pesos(r.ganancias - r.costos)}  (cobró {pesos(r.ganancias)}, gastó {pesos(r.costos)})"


def balance(r) -> str:
    return f"{r.entregas} entregados · {r.cancelaciones} cancelados · {r.rechazos} rechazos"


def lista_pedidos(r) -> list[str]:
    """Los pedidos que ya terminó (entregados o cancelados en sus manos), el más reciente primero."""
    lineas = []
    for p in reversed(r.model.pedidos):
        if p.repartidor is not r or not p.terminado:
            continue
        destino = f"#{p.id} {p.restaurante.nombre} al hab. {p.habitante.unique_id}"
        if p.estado == EstadoPedido.ENTREGADO:
            lineas.append(f"{destino}: en {p.tiempo_entrega} min, cobró {pesos(p.pago_repartidor)}")
        else:
            lineas.append(f"{destino}: se canceló antes de entregarlo, no cobró")
    return lineas
