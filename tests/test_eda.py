"""Prueba de humo del notebook de EDA: corre de punta a punta sobre un panel sintético, sin tocar `data/`."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from demanda.datos import calidad

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "eda.py"


def test_notebook_corre_sobre_datos_sinteticos(generar_tablas, carpeta_csv, tmp_path):
    # 240 días para que haya periodo de evaluación desde 2023-07-01
    raw = carpeta_csv(generar_tablas(dias=240))
    panel, reporte = tmp_path / "panel.parquet", tmp_path / "reporte.json"
    resumen = tmp_path / "eda" / "resumen.json"
    calidad.ejecutar(raw, panel, reporte)

    args = ["--panel", str(panel), "--reporte", str(reporte), "--salida", str(resumen)]
    corrida = subprocess.run(
        [sys.executable, str(NOTEBOOK), *args], capture_output=True, text=True, timeout=300
    )
    assert corrida.returncode == 0, corrida.stderr[-3000:]

    r = json.loads(resumen.read_text())
    assert {"fuente", "calidad", "demanda", "promociones", "tiendas", "sistema_actual"} <= set(r)
    efectos = r["demanda"]["efectos_calendario"]
    # El generador multiplica el fin de semana por 1.5 y no le da efecto a la quincena
    assert efectos["fin_de_semana"] == pytest.approx(1.5, rel=0.1)
    assert efectos["quincena"] == pytest.approx(1.0, rel=0.1)
    # Eventos sin días en el periodo: null, no error
    assert efectos["buen_fin"] is None
    assert efectos["semana_santa"] is None
