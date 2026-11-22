"""Lo que se cuenta de un habitante cuando se le da clic en la vista 3D.

Por qué aparte y sin Ursina: es pura lectura del agente (su cuerpo, su agenda, su
última decisión y lo que aprendió), y así se prueba con pytest sin abrir ventana.
La vista solo le pone colores.
"""
from __future__ import annotations

from sim.agentes.habitante import ACTIVIDADES_FUERA
from sim.reloj import MINUTOS_POR_DIA, MINUTOS_POR_HORA

NOMBRE_OPCION = {"cocinar": "Cocinar", "salir": "Salir a comer", "pedir": "Pedir domicilio"}
NOMBRE_CORTO = {"cocinar": "Cocinar", "salir": "Salir", "pedir": "Pedir"}
NOMBRE_ACTIVIDAD = {"dormir": "durmiendo", "trabajar": "trabajando en la oficina",
                    "estudiar": "estudiando en la universidad", "ejercicio": "haciendo ejercicio",
                    "libre": "en la casa, tiempo libre"}
NOMBRE_ESTADO_PEDIDO = {"creado": "recién pedido", "en_preparacion": "en la cocina", "listo": "listo, sin moto",
                        "en_camino": "en camino", "entregado": "entregado", "cancelado": "cancelado"}


def hora(tick_absoluto_del_dia: int) -> str:
    minuto = tick_absoluto_del_dia % MINUTOS_POR_DIA
    return f"{minuto // MINUTOS_POR_HORA:02d}:{minuto % MINUTOS_POR_HORA:02d}"


def estado(h) -> tuple[str, str]:
    """(frase, clave). La clave escoge el color del aro del ícono."""
    if h.pedido is not None:
        p = h.pedido
        return (f"Esperando su domicilio de {p.restaurante.nombre} "
                f"({NOMBRE_ESTADO_PEDIDO[p.estado.value]}, lleva {p.edad(h.t)} min)", "esperando")
    if h.comida_en_curso == "cocinar":
        return f"Cocinando (le faltan {max(0, h.ocupado_hasta - h.t)} min)", "cocinando"
    if h.comida_en_curso == "salir":
        return f"Comiendo afuera (vuelve en {max(0, h.ocupado_hasta - h.t)} min)", "afuera"
    if h.actividad == "dormir":
        return "Durmiendo", "durmiendo"
    if h.hambre >= h.cfg["umbral_hambre"]:
        return "Con hambre y sin cómo comer (no le alcanza la plata o la energía)", "hambre"
    return "Bien, sin hambre todavía", "bien"


def fraccion_hambre(h) -> float:
    return max(0.0, min(1.0, h.hambre / 100))


def opciones_ahora(h) -> dict[str, float]:
    """Lo que valdría cada opción si decidiera en este momento (sin azar: no cambia nada)."""
    return h.opciones_comida(h.percibir())


def frase_decision(d: dict | None, resaltar: str = "") -> str:
    """En una línea: "(08:28, lloviendo): Cocinar". `resaltar` es una etiqueta de color
    que la vista pone antes de lo que escogió (aquí no se sabe de colores)."""
    if d is None:
        return "(todavía no ha tenido que decidir)"
    cuando = f"({hora(d['minuto_del_dia'])}{', lloviendo' if d['lluvia'] else ''})"
    if d["eleccion"] is None:
        return f"{cuando}: sin plata ni energía, aguantó hambre"
    escogio = NOMBRE_OPCION[d["eleccion"]]
    if d["eleccion"] == "pedir":
        escogio = f"Pedir a {d['restaurante']}" + (" (explorando)" if d["exploro"] else "")
    return f"{cuando}: {resaltar}{escogio}"


def lo_que_aprendio(h) -> list[tuple[str, float, int]]:
    """(restaurante, satisfacción esperada, veces que pidió), del bandit del habitante."""
    q = getattr(h.politica, "q", {})
    conteos = getattr(h.politica, "conteos", {})
    return [(nombre, valor, conteos.get(nombre, 0)) for nombre, valor in q.items()]


def comidas_de_hoy(modelo, h) -> list[str]:
    """Lo que comió en el día que va, en orden."""
    hoy = modelo.reloj.dia
    inicio = modelo.reloj.minuto_inicio
    lineas = []
    for c in modelo.comidas:
        if c["habitante"] != h.unique_id or c["dia"] != hoy:
            continue
        cuando = hora(inicio + c["tick"])
        if c["tipo"] == "pedir":
            lineas.append(f"{cuando} pidió {c['restaurante']}")
        else:
            lineas.append(f"{cuando} {'cocinó' if c['tipo'] == 'cocinar' else 'salió a comer'}")
    return lineas


def donde(h) -> str:
    lugar = "la universidad" if h.tipo == "estudiante" else "la oficina"
    return f"en {lugar}" if h.actividad in ACTIVIDADES_FUERA else "en la casa"


def describir_actividad(h) -> str:
    return NOMBRE_ACTIVIDAD[h.actividad]
