import json
from datetime import date, timedelta

import polars as pl
import pytest

from demanda.datos import calidad, panel
from demanda.datos.contratos import ContratoRoto

INICIO = date(2023, 1, 1)  # primera fecha de los datos sintéticos de conftest


def _quitar_dias(tablas, tienda: str, dias: list[int]):
    """Quita de `transactions` todas las categorías de una tienda en esos días (huecos de la malla)."""
    fechas = [INICIO + timedelta(days=d) for d in dias]
    t = tablas.transacciones.filter(~((pl.col("store_id") == tienda) & pl.col("date").is_in(fechas)))
    return tablas._replace(transacciones=t)


def test_panel_completa_la_malla_sin_imputar(tablas):
    con_huecos = _quitar_dias(tablas, "STR_002", [10, 11])
    p = panel.construir(con_huecos)

    assert p.height == tablas.transacciones.height
    faltantes = p.filter("flag_fila_faltante")
    assert faltantes.height == 2 * 2  # dos días × dos categorías
    assert faltantes["units_sold"].null_count() == faltantes.height
    # Los nulos de una fila que no existe se atribuyen a la fila, no a la columna
    assert not faltantes.select(pl.col("^flag_nulo_.*$").any()).row(0)[0]
    # Los atributos de tienda y calendario sí se llenan
    assert faltantes["store_format"].null_count() == 0
    assert faltantes["is_weekend"].null_count() == 0


def test_banderas_coinciden_con_los_datos(tablas):
    p = panel.construir(tablas)
    t = tablas.transacciones
    inconsistentes = t.filter(
        pl.col("cash_transactions") + pl.col("card_transactions") < pl.col("total_transactions")
    ).height
    assert p["flag_conteo_inconsistente"].sum() == inconsistentes
    assert p["flag_nulo_units_sold"].sum() == t["units_sold"].null_count()
    assert p.schema["units_sold"] == pl.Int64


def test_malla_reporta_bloques_de_dias_consecutivos(tablas):
    p = panel.construir(_quitar_dias(tablas, "STR_002", [10, 11, 30]))
    malla = calidad.reporte_malla(p)

    assert malla["faltantes"] == 3 * 2
    assert malla["esperadas"] == malla["observadas"] + malla["faltantes"]
    assert [(b["desde"], b["dias"]) for b in malla["bloques"]] == [("2023-01-11", 2), ("2023-01-31", 1)]


def test_conteos_y_avg_ticket(tablas):
    conteos = calidad.reporte_conteos(tablas.transacciones)

    assert 15 < conteos["pct_inconsistentes"] < 25
    assert conteos["avg_ticket"]["pct_igual_a_monto_entre_cash_mas_card"] == 100.0
    assert conteos["avg_ticket"]["pct_igual_a_monto_entre_total"] < 100.0
    assert conteos["avg_ticket"]["sobreestimacion_media_pct_en_inconsistentes"] > 0


def test_patron_de_nulos(tablas):
    nulos = calidad.reporte_nulos(tablas.transacciones)

    assert nulos["replenishment_signal"]["patron"] == "bloque"
    assert nulos["replenishment_signal"]["fechas"] == ["2023-01-01", "2023-01-02"]
    assert nulos["units_sold"]["patron"] == "disperso"
    assert nulos["cash_transactions"]["tambien_nulo_pct"]["amount_cash"] == 100.0
    assert nulos["avg_ticket"]["tambien_nulo_pct"]["cash_transactions"] == 100.0
    assert "total_transactions" not in nulos


@pytest.mark.parametrize(
    ("ventana", "usa_futuro", "rezago"),
    [((-6, 0), False, 0), ((-5, 1), True, 1), ((-9, -1), False, 0)],
    ids=["hasta el día", "un día adelante", "hasta el día anterior"],
)
def test_prueba_de_fuga_encuentra_la_ventana(generar_tablas, ventana, usa_futuro, rezago):
    fuga = calidad.reporte_fuga(panel.construir(generar_tablas(ventana_senal=ventana)))

    assert (fuga["mejor_ventana"]["desde"], fuga["mejor_ventana"]["hasta"]) == ventana
    assert fuga["usa_informacion_futura"] is usa_futuro
    assert fuga["rezago_minimo_desde_el_origen"] == rezago
    assert fuga["factor"] == pytest.approx(0.53, abs=0.03)


def test_ejecutar_escribe_panel_y_reporte(tablas, carpeta_csv, tmp_path):
    raw = carpeta_csv(tablas)
    reporte = calidad.ejecutar(
        raw, tmp_path / "interim" / "panel.parquet", tmp_path / "calidad" / "reporte.json"
    )

    escrito = json.loads((tmp_path / "calidad" / "reporte.json").read_text())
    assert escrito == json.loads(json.dumps(reporte))
    assert escrito["contratos"] == {
        "transacciones": "ok",
        "tiendas": "ok",
        "calendario": "ok",
        "uniones": "ok",
    }
    assert set(escrito["fuente"]) == {"calendar.csv", "stores.csv", "transactions.csv"}
    assert pl.read_parquet(tmp_path / "interim" / "panel.parquet").height == tablas.transacciones.height


def test_ejecutar_no_escribe_nada_si_el_contrato_se_rompe(tablas, carpeta_csv, tmp_path):
    roto = tablas._replace(tiendas=tablas.tiendas.with_columns(pl.lit("Mega").alias("store_format")))
    raw = carpeta_csv(roto)
    with pytest.raises(ContratoRoto):
        calidad.ejecutar(raw, tmp_path / "panel.parquet", tmp_path / "reporte.json")
    assert not (tmp_path / "panel.parquet").exists()
    assert not (tmp_path / "reporte.json").exists()
