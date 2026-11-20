import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import cargar_config  # noqa: E402


@pytest.fixture
def config():
    """Config base sin recolección por tick (las pruebas no la necesitan)."""
    return cargar_config(cambios={"metricas": {"recolectar_cada": 0}})


@pytest.fixture
def config_vacia(config):
    """Ciudad sin agentes móviles: sirve para probar el entorno solo (fase 1)."""
    from sim.config import fusionar
    return fusionar(config, {"habitantes": {"cantidad": 0}, "repartidores": {"cantidad": 0}})
