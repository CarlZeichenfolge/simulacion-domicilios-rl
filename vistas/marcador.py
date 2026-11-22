"""Marcador final de los duelos: la tabla que aparece cuando termina el día.

Sin pygame ni ursina: lo usan el duelo 2D y el 3D, y se puede probar sin ventana.
"""
from __future__ import annotations

from sim.metricas import resumen_episodio
from sim.model import CiudadModel
from sim.reloj import MINUTOS_POR_DIA

DIAS_SEMANA = 7

# (clave de resumen_episodio, etiqueta, formato, ¿más es mejor?)
METRICAS = [
    ("tasa_cancelacion", "Pedidos cancelados", "{:.1%}", False),
    ("entregados", "Pedidos entregados", "{:.0f}", True),
    ("rechazos", "Veces que dejaron pedidos esperando", "{:.0f}", False),
    ("tiempo_entrega_promedio", "Tiempo de entrega (min)", "{:.1f}", False),
    ("satisfaccion_promedio", "Satisfacción de los clientes", "{:.2f}", True),
    ("recompensa_total", "Recompensa de los repartidores", "{:+.0f}", True),
]


def filas_marcador(antes: CiudadModel, despues: CiudadModel) -> list[tuple[str, str, str, str]]:
    """(etiqueta, valor antes, valor después, quién ganó) para cada métrica.

    "Quién ganó" sale de la dirección de la métrica: en cancelaciones gana el MENOR
    número, en entregas el mayor. Así el marcador no depende de que el lector lo sepa.
    """
    ra, rd = resumen_episodio(antes), resumen_episodio(despues)
    filas = []
    for clave, etiqueta, formato, mas_es_mejor in METRICAS:
        a, d = ra[clave], rd[clave]
        if a == d or a != a or d != d:  # empate o dato faltante (NaN != NaN)
            ganador = "="
        else:
            ganador = "después" if (d > a) == mas_es_mejor else "antes"
        filas.append((etiqueta, formato.format(a), formato.format(d), ganador))
    return filas


def titulo_marcador(modelo: CiudadModel) -> str:
    """Al terminar: "FIN DEL DÍA 1" o "FIN DE LA SEMANA". A mitad de camino (tecla M): "HASTA AHORA"."""
    dias = round(modelo.duracion / MINUTOS_POR_DIA)
    if modelo.running:
        return f"HASTA AHORA (DÍA {modelo.reloj.dia + 1} DE {dias})"
    if dias == 1:
        return f"FIN DEL DÍA {modelo.reloj.dia}"
    return "FIN DE LA SEMANA" if dias == DIAS_SEMANA else f"FIN DE LOS {dias} DÍAS"


def texto_marcador(antes: CiudadModel, despues: CiudadModel, titulo_antes: str, titulo_despues: str) -> str:
    """El marcador como texto de ancho fijo (las vistas lo pintan con una fuente monoespaciada o tal cual)."""
    filas = filas_marcador(antes, despues)
    ancho = max(len(f[0]) for f in filas)
    lineas = [titulo_marcador(despues), "",
              f"{'':<{ancho}}   {'ANTES':>10}   {'DESPUÉS':>10}",
              ]
    for etiqueta, a, d, ganador in filas:
        # Sin "◄" (no todas las fuentes lo traen) ni "<" (Ursina lo lee como etiqueta de color).
        marca = "  (mejor)" if ganador == "después" else ("  (mejor antes)" if ganador == "antes" else "")
        lineas.append(f"{etiqueta:<{ancho}}   {a:>10}   {d:>10}{marca}")
    lineas += ["", f"ANTES = {titulo_antes.split(': ', 1)[-1]}   ·   DESPUÉS = {titulo_despues.split(': ', 1)[-1]}"]
    return "\n".join(lineas)
