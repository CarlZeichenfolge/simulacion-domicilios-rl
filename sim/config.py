"""Carga de la configuración YAML.

Por qué YAML: cada experimento es un archivo con solo los parámetros que
cambian, y el resto se hereda de base.yaml. Así nadie toca código para
correr un experimento distinto.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
CONFIG_BASE = RAIZ / "config" / "base.yaml"


def fusionar(base: dict, cambios: dict) -> dict:
    """Mezcla recursiva: lo de `cambios` pisa a `base` sin borrar lo demás.

    Por qué recursiva: un experimento que solo cambia
    `repartidores.politica` no debe perder `repartidores.qlearning`.
    """
    resultado = copy.deepcopy(base)
    for clave, valor in (cambios or {}).items():
        if isinstance(valor, dict) and isinstance(resultado.get(clave), dict):
            resultado[clave] = fusionar(resultado[clave], valor)
        else:
            resultado[clave] = copy.deepcopy(valor)
    return resultado


def leer_yaml(ruta: str | Path) -> dict:
    with open(ruta, encoding="utf-8") as archivo:
        return yaml.safe_load(archivo) or {}


def cargar_config(ruta: str | Path | None = None, cambios: dict | None = None) -> dict:
    """Devuelve base.yaml fusionado con el archivo `ruta` y luego con `cambios`."""
    config = leer_yaml(CONFIG_BASE)
    if ruta is not None and Path(ruta).resolve() != CONFIG_BASE:
        config = fusionar(config, leer_yaml(ruta))
    return fusionar(config, cambios or {})
