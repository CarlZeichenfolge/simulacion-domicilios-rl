"""Vista 3D con Ursina (fase 6, pendiente).

Debe seguir la misma regla que la 2D: solo LEER el modelo. Plan:
  - calles: planos delgados entre nodos (color rojo si factor_trancon > 1)
  - edificios: cubos con altura por tipo (oficina alta, casa baja, restaurante con color propio)
  - repartidores: cubo/moto en r.posicion_continua(), color por r.estado
  - lluvia: ursina.prefabs (partículas) cuando modelo.eventos.lloviendo
  - en update(): llamar modelo.step() N veces y mover las entidades
Instalar con:  pip install ursina
"""
from __future__ import annotations


class Vista3D:
    def __init__(self, modelo):
        self.modelo = modelo

    def correr(self) -> None:
        try:
            import ursina  # noqa: F401
        except ImportError:
            raise SystemExit("La vista 3D necesita Ursina: pip install ursina") from None
        raise SystemExit("La vista 3D todavía no está implementada (fase 6). Usa --vista 2d.")
