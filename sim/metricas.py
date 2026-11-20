"""Métricas por tick (DataCollector) y resumen por episodio."""
from __future__ import annotations

from statistics import mean

import mesa


def _repartidores(m):
    from sim.agentes import Repartidor
    return m.agents_by_type.get(Repartidor, [])


def crear_datacollector() -> mesa.DataCollector:
    """Serie de tiempo para la vista y para gráficas dentro de un día.

    Por qué no se guarda cada tick en los experimentos: 500 episodios x 1440
    ticks es mucho dato que no se usa; allí basta el resumen por episodio.
    """
    return mesa.DataCollector(model_reporters={
        "tick": lambda m: m.reloj.tick,
        "creados": lambda m: m.conteo["creados"],
        "entregados": lambda m: m.conteo["entregados"],
        "cancelados": lambda m: m.conteo["cancelados"],
        "activos": lambda m: len(m.pedidos_activos),
        "lluvia": lambda m: int(m.eventos.lloviendo),
        "trancon": lambda m: int(m.eventos.trancon_activo),
        "ganancia_neta": lambda m: sum(r.ganancias - r.costos for r in _repartidores(m)),
        "recompensa": lambda m: sum(r.recompensa_acumulada for r in _repartidores(m)),
    })


def _promedio(valores) -> float:
    valores = list(valores)
    return mean(valores) if valores else float("nan")


def resumen_episodio(modelo) -> dict:
    """Una fila por episodio: es lo que se compara entre políticas y escenarios."""
    entregados = [p for p in modelo.pedidos if p.entregado is not None]
    reps = list(_repartidores(modelo))
    creados = modelo.conteo["creados"]
    fila = {
        "semilla": modelo.semilla,
        "pedidos_creados": creados,
        "entregados": modelo.conteo["entregados"],
        "cancelados": modelo.conteo["cancelados"],
        "tasa_cancelacion": modelo.conteo["cancelados"] / creados if creados else float("nan"),
        "tiempo_entrega_promedio": _promedio(p.tiempo_entrega for p in entregados),
        "espera_hasta_recogida": _promedio(p.recogido - p.creado for p in entregados),
        "ganancia_neta_por_repartidor": _promedio(r.ganancias - r.costos for r in reps),
        "recompensa_total": sum(r.recompensa_acumulada for r in reps),
        "rechazos": sum(r.rechazos for r in reps),
        "satisfaccion_promedio": _promedio(modelo.satisfacciones),
        "fraccion_lluvia": modelo.minutos_lluvia / max(1, modelo.reloj.tick),
        "repartidores": len(reps),
    }
    for tipo in ("cocinar", "salir", "pedir"):
        fila[f"comidas_{tipo}"] = sum(1 for c in modelo.comidas if c["tipo"] == tipo)
    for nombre in modelo.restaurantes:
        fila[f"pedidos_{nombre}"] = sum(1 for p in modelo.pedidos if p.restaurante.nombre == nombre)
    return fila
