"""Punto de entrada: `uv run demanda <comando>`. Cada etapa del pipeline agrega aquí su comando."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from demanda.config import RAIZ, get_config
from demanda.datos import calidad, fuente

NOTEBOOK_EDA = RAIZ / "notebooks" / "eda.py"


def _data(args: argparse.Namespace) -> None:
    cfg = get_config()
    manifiesto = cfg.data_dir / "manifest.sha256"
    if args.src:
        hashes = fuente.preparar(Path(args.src), cfg.raw_dir, manifiesto)
    else:
        hashes = fuente.verificar(cfg.raw_dir, manifiesto)
    for nombre, digest in hashes.items():
        print(f"ok  {nombre}  {digest[:12]}")


def _calidad(args: argparse.Namespace) -> None:
    cfg = get_config()
    fuente.verificar(cfg.raw_dir, cfg.data_dir / "manifest.sha256")
    panel_path = cfg.interim_dir / "panel.parquet"
    reporte_path = cfg.artifacts_dir / "calidad" / "reporte.json"
    r = calidad.ejecutar(cfg.raw_dir, panel_path, reporte_path)
    fuga = r["fuga_replenishment_signal"]["mejor_ventana"]
    print(f"contratos ok · {r['dimensiones']['filas']} filas · {r['dimensiones']['series']} series")
    print(f"malla: {r['malla']['faltantes']} filas faltantes en {len(r['malla']['bloques'])} bloques")
    print(f"conteos inconsistentes: {r['conteos']['pct_inconsistentes']} %")
    print("nulos: " + ", ".join(f"{c} {n['pct']} %" for c, n in r["nulos"].items()))
    print(
        f"replenishment_signal ≈ media de {fuga['dias']} días, de t{fuga['desde']:+d} a t{fuga['hasta']:+d}"
    )
    print(f"escrito: {panel_path} y {reporte_path}")


def _eda(args: argparse.Namespace) -> None:
    cfg = get_config()
    panel_path = cfg.interim_dir / "panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"no existe {panel_path}; corre antes `demanda calidad`")
    salida = cfg.reports_dir / "eda.html"
    salida.parent.mkdir(parents=True, exist_ok=True)
    # Exportar ejecuta el notebook completo; marimo sale con error si alguna celda falla
    comando = [sys.executable, "-m", "marimo", "export", "html", str(NOTEBOOK_EDA), "-o", str(salida), "-f"]
    if subprocess.run(comando, check=False).returncode:
        raise SystemExit(1)
    print(f"escrito: {salida} y {cfg.artifacts_dir / 'eda' / 'resumen.json'}")


def _pct(x: float | None, signo: bool = False) -> str:
    return "—" if x is None else f"{x:{'+' if signo else ''}.2f} %"


def _progreso(mensaje: str) -> None:
    print(mensaje, file=sys.stderr, flush=True)


def _paneles() -> tuple[Path, Path]:
    """El panel y el reporte de calidad, que necesitan el backtest y el ajuste."""
    cfg = get_config()
    panel_path = cfg.interim_dir / "panel.parquet"
    reporte_path = cfg.artifacts_dir / "calidad" / "reporte.json"
    for ruta in (panel_path, reporte_path):
        if not ruta.exists():
            raise FileNotFoundError(f"no existe {ruta}; corre antes `demanda calidad`")
    return panel_path, reporte_path


def _backtest(args: argparse.Namespace) -> None:
    # statsforecast, LightGBM y MLflow tardan en importarse: solo los carga este comando
    from demanda.backtest import etapa

    cfg = get_config()
    panel_path, reporte_path = _paneles()
    salida = cfg.artifacts_dir / "backtest"
    modelos = [m.strip() for m in args.modelos.split(",") if m.strip()] if args.modelos else None
    r = etapa.ejecutar(
        panel_path,
        reporte_path,
        cfg.artifacts_dir / "eda" / "resumen.json",
        salida,
        cfg.mlflow_tracking_uri,
        cfg.mlruns_dir,
        n_jobs=-1,
        modelos=modelos,
        progreso=_progreso,
    )
    c = r.metricas["conciliacion_e2"]
    print(
        f"conciliación con E2: sistema actual {_pct(c['wape_pct'])} contra {_pct(c['referencia_e2_pct'])} "
        f"(diferencia {'—' if c['diferencia_pp'] is None else c['diferencia_pp']} pp, "
        f"{c['origenes_evaluados']:,} orígenes)"
    )
    s = r.metricas["cobertura"]["semanal"]
    print(
        f"cobertura semanal: {s['pares_completos']:,} de {s['pares_total']:,} pares completos ({s['pct']} %)"
    )
    columnas = ("WAPE sem", "sesgo sem", "WAPE día", "MASE", "pinball", "cob 80")
    print(f"{'modelo':<18}" + "".join(f"{c:>11}" for c in columnas))
    for modelo, g in r.globales.items():
        mase = g.get("diaria_mase")
        celdas = (
            _pct(g.get("semanal_wape_pct")),
            _pct(g.get("semanal_sesgo_pct"), signo=True),
            _pct(g.get("diaria_wape_pct")),
            "—" if mase is None else f"{mase:.3f}",
            _pct(g.get("diaria_pinball_medio_pct")),
            _pct(g.get("diaria_cobertura_80_pct")),
        )
        print(f"{modelo:<18}" + "".join(f"{c:>11}" for c in celdas))
    print(f"escrito: {salida / 'pronosticos.parquet'} y {salida / 'metricas.json'}")
    print(f"MLflow: {cfg.mlflow_tracking_uri}")


def _ajustar(args: argparse.Namespace) -> None:
    from demanda.modelos import ajuste

    cfg = get_config()
    panel_path, reporte_path = _paneles()
    salida = cfg.artifacts_dir / "modelos" / "ajuste_lightgbm.json"
    r = ajuste.ejecutar(
        panel_path, reporte_path, salida, cfg.mlflow_tracking_uri, cfg.mlruns_dir, progreso=_progreso
    )
    print(f"orígenes: {r['protocolo']['origenes'][0]} a {r['protocolo']['origenes'][-1]} (solo P50)")
    print(f"{'configuración':<30}{'WAPE sem':>11}")
    for fila in (*r["rejilla"], {"nombre": r["referencia"]["modelo"], **r["referencia"]}):
        print(f"{fila['nombre']:<30}{_pct(fila['wape_semanal_pct']):>11}")
    print(f"ganadora: {r['ganadora']['nombre']}; cópiala en `gbm.AJUSTADOS` si cambió")
    print(f"escrito: {salida}")


def _seleccion(args: argparse.Namespace) -> None:
    from demanda.seleccion import campeon

    cfg = get_config()
    backtest = cfg.artifacts_dir / "backtest"
    metricas_path, pronosticos_path = backtest / "metricas.json", backtest / "pronosticos.parquet"
    for ruta in (metricas_path, pronosticos_path):
        if not ruta.exists():
            raise FileNotFoundError(f"no existe {ruta}; corre antes `demanda backtest`")
    salida = cfg.artifacts_dir / "modelos" / "seleccion.json"
    s = campeon.ejecutar(metricas_path, pronosticos_path, salida)
    print(f"incumbente: {s['incumbente']} · mejor baseline: {(s['mejor_baseline'] or {}).get('modelo', '—')}")
    nombres = list(campeon.CRITERIOS)
    print(f"{'modelo':<18}{'WAPE sem':>11}{'resto':>11}{'|sesgo| máx':>13}  " + "  ".join(nombres))
    for modelo, m in s["modelos"].items():
        celdas = [_pct(m["wape_semanal_global_pct"]), _pct(m["wape_semanal_resto_pct"])]
        criterios = m["criterios"] or {}
        marcas = "  ".join(
            f"{('sí' if criterios[n]['cumple'] else 'no') if criterios else '—':<{len(n)}}" for n in nombres
        )
        print(
            f"{modelo:<18}{celdas[0]:>11}{celdas[1]:>11}{_pct(m['sesgo_semanal_max_abs_pct']):>13}  {marcas}"
        )
    print("Diebold-Mariano (d = |E_a| − |E_b| por corte):")
    for p in s["pruebas_dm"]:
        diferencia = "—" if p["diferencia_wape_pp"] is None else f"{p['diferencia_wape_pp']:+.2f} pp"
        valor_p = "— (menos de 3 cortes)" if p["p_valor"] is None else f"{p['p_valor']:.4f}"
        print(
            f"  {p['a']:<16} vs {p['b']:<16} {p['segmento']:<7} n={p['n_cortes']:<3} "
            f"Δ WAPE {diferencia}  p = {valor_p}"
        )
    print(f"campeón: {s['campeon']}")
    print(s["justificacion"])
    print(f"escrito: {salida}")


def _requeridos(*rutas: tuple[Path, str]) -> None:
    """Falla con un mensaje claro si falta un artefacto previo: cada ruta va con el comando que la escribe."""
    for ruta, comando in rutas:
        if not ruta.exists():
            raise FileNotFoundError(f"no existe {ruta}; corre antes `demanda {comando}`")


def _artefactos_e4() -> tuple[Path, Path, Path]:
    """Los pronósticos y las métricas del backtest, y la selección del campeón."""
    cfg = get_config()
    backtest = cfg.artifacts_dir / "backtest"
    rutas = (
        backtest / "pronosticos.parquet",
        backtest / "metricas.json",
        cfg.artifacts_dir / "modelos" / "seleccion.json",
    )
    _requeridos((rutas[0], "backtest"), (rutas[1], "backtest"), (rutas[2], "seleccion"))
    return rutas


def _calibracion(args: argparse.Namespace) -> None:
    from demanda.backtest import calibracion

    cfg = get_config()
    panel_path, reporte_path = _paneles()
    _, _, seleccion_path = _artefactos_e4()
    salida = cfg.artifacts_dir / "backtest" / "calibracion.parquet"
    p = calibracion.ejecutar(panel_path, reporte_path, seleccion_path, salida, n_jobs=-1, progreso=_progreso)
    origenes = p["origen"].unique().sort()
    print(f"modelos: {', '.join(p['modelo'].unique(maintain_order=True))}")
    print(f"orígenes: {origenes[0]} a {origenes[-1]} ({origenes.len()})")
    print(f"escrito: {salida} y {calibracion.registro(salida)}")


def _calibracion_previa() -> Path:
    ruta = get_config().artifacts_dir / "backtest" / "calibracion.parquet"
    _requeridos((ruta, "calibracion"))
    return ruta


def _decision(args: argparse.Namespace) -> None:
    from demanda.decision import etapa

    cfg = get_config()
    panel_path, _ = _paneles()
    pronosticos_path, metricas_path, seleccion_path = _artefactos_e4()
    salida = cfg.artifacts_dir / "decision" / "impacto.json"
    r = etapa.ejecutar(
        pronosticos_path, metricas_path, seleccion_path, _calibracion_previa(), panel_path, salida
    )
    print(f"campeón: {r['campeon']} · retador: {r['retador']}")
    columnas = ("costo", "sobreinv.", "quiebre", "fill rate", "servicio")
    for segmento, nivel in (("global", "global"), ("evento", "resto")):
        print(f"{segmento}={nivel}")
        print(f"  {'política':<28}" + "".join(f"{c:>15}" for c in columnas))
        for f in r["impacto"]:
            if (f["segmento"], f["nivel"]) != (segmento, nivel):
                continue
            celdas = (
                f"{f['costo']:,.0f}",
                _pct(f["sobreinventario_pct"]),
                _pct(f["quiebre_pct"]),
                _pct(f["fill_rate_pct"]),
                _pct(f["nivel_servicio_pct"]),
            )
            print(f"  {f['politica']:<28}" + "".join(f"{c:>15}" for c in celdas))
    print("Diebold-Mariano sobre el costo por corte:")
    for p in r["pruebas_costo"]:
        valor_p = "—" if p["p_valor"] is None else f"{p['p_valor']:.4f}"
        print(
            f"  {p['a']:<26} vs {p['b']:<26} {p['segmento']:<7} n={p['n_cortes']:<3} "
            f"Δ {_pct(p['diferencia_pct'], signo=True)}  p = {valor_p}"
        )
    print(f"orden se mantiene en la sensibilidad: {'sí' if r['sensibilidad']['orden_se_mantiene'] else 'no'}")
    print(r["conclusion"])
    print(f"escrito: {salida}")


def _drift(args: argparse.Namespace) -> None:
    from demanda.drift import etapa

    cfg = get_config()
    panel_path, _ = _paneles()
    pronosticos_path, metricas_path, seleccion_path = _artefactos_e4()
    salida = cfg.artifacts_dir / "drift" / "monitor.json"
    r = etapa.ejecutar(
        panel_path, pronosticos_path, metricas_path, seleccion_path, _calibracion_previa(), salida
    )
    print("falsas alarmas en las semanas de calibración (alertas inesperadas por chequeo):")
    print("  " + " · ".join(f"{c} {n}" for c, n in r["resumen"]["falsas_alarmas_calibracion"].items()))
    print("semanas del backtest:")
    for s in r["semanas"]:
        if s["periodo"] != "backtest":
            continue
        inesperadas = sum(
            a["severidad"] == "alerta" and a["clasificacion"] == "inesperada" for a in s["alertas"]
        )
        esperadas = sum(a["severidad"] == "alerta" and a["clasificacion"] == "esperada" for a in s["alertas"])
        codigos = ", ".join(rec["codigo"] for rec in s["recomendaciones"])
        print(
            f"  {s['origen']} {s['evento']:<14} inesperadas {inesperadas:>3} · esperadas {esperadas:>3}"
            f" · {codigos}"
        )
    v = r["vigilancia_retador"]
    final = v["final"] or {}
    print(
        f"retador {r['retador']}: DM acumulada p = {final.get('p_valor', '—')} al último corte "
        f"(mínima {(v['minima'] or {}).get('p_valor', '—')}); coincide con seleccion.json: "
        f"{v['coincide_con_seleccion']}"
    )
    val = r["validacion"]
    for e in val["escenarios"]:
        print(f"  {e['escenario']:<17} {e['origen']} detectado: {'sí' if e['detectado'] else 'no'}")
    print(f"escrito: {salida}")


def _causal(args: argparse.Namespace) -> None:
    from demanda.causal import etapa

    cfg = get_config()
    panel_path = cfg.interim_dir / "panel.parquet"
    _requeridos((panel_path, "calidad"))
    salida = cfg.artifacts_dir / "causal" / "promociones.json"
    r = etapa.ejecutar(panel_path, salida, progreso=_progreso)
    b = r["balance"]
    print(
        f"muestra: {r['muestra']['filas']:,} filas · {r['muestra']['series']} series · "
        f"{r['muestra']['tiendas']} tiendas"
    )
    print(
        f"balance: B1 p = {b['b1']['p_valor']:.3g} · B3 p = {b['b3']['p_valor']:.3g} · p de Holm mínima "
        f"{b['familia']['p_holm_min']:.3g} en {b['familia']['n_pruebas']} pruebas · se sostiene: "
        f"{'sí' if b['se_sostiene'] else 'no (' + ', '.join(b['fallan']) + ')'}"
    )

    def efecto(e: dict) -> str:
        ic = e["ic95_pct"]
        return f"{e['efecto_pct']:+.2f} % [{ic[0]:+.2f}, {ic[1]:+.2f}]  p = {e['p_valor']:.3g}"

    e = r["efecto_principal"]
    print(f"efecto principal: {efecto(e)} · por aleatorización p = {e['p_aleatorizacion']:.3g}")
    aa = r["calibracion_aa"]
    print(
        f"A/A: {aa['tasa_pct']:.1f} % de rechazos (intervalo {aa['intervalo_99_pct'][0]:.1f} a "
        f"{aa['intervalo_99_pct'][1]:.1f} %) · inferencia principal: {aa['inferencia_principal']}"
    )
    for nombre, c in r["canales"].items():
        print(f"  canal {nombre:<32} {efecto(c)}")
    for nombre, v in r["robustez"].items():
        if "efecto_pct" in v:
            print(f"  robustez {nombre:<29} {efecto(v)}")
    for dimension in ("categoria", "formato"):
        for g, v in r["heterogeneidad"][dimension]["grupos"].items():
            print(f"  {dimension} {g:<30} {efecto(v)}  Holm {v['p_holm']:.3g}")
    d = r["dinamica"]
    print(f"dinámica: neto {efecto(d['neto'])} · adelanto: {'sí' if d['adelanto'] else 'no'}")
    print(f"cruzado: {efecto(r['cruzado']['otras_categorias'])} · {r['cruzado']['interpretacion']}")
    pot = r["potencia"]
    sem = pot["diseno_semanal"]
    print(
        f"MDE diario: {pot['diseno_actual']['global']['mde_pct']:.2f} % · semanal "
        f"({sem['semanas']} semanas): {sem['mde_actual']['mde_pct']:.2f} %"
    )
    print(
        "  MDE semanal por W: "
        + " · ".join(f"{f['semanas']}: {f['mde_pct']:.2f} %" for f in sem["por_semanas"])
    )
    print(
        "  semanas para "
        + ", ".join(f"{f['efecto_pct']:g} %: {f['semanas'] or '—'}" for f in sem["semanas_necesarias"])
    )
    for f in r["negocio"]["por_descuento"]:
        print(
            f"  descuento {100 * f['descuento']:.3g} % ({f['origen']}): "
            + " · ".join(f"{v} {', '.join(f[v]) or '—'}" for v in ("se_paga", "no_se_paga", "indeterminado"))
        )
    print(r["conclusion"]["texto"])
    print(f"escrito: {salida}")


def _insights(args: argparse.Namespace) -> None:
    # Pydantic AI imprime un aviso de observabilidad en cada proceso; aquí solo es ruido en stderr
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    from demanda.agentes import etapa

    cfg = get_config()
    r = etapa.ejecutar(cfg, args.modo, progreso=_progreso)
    print(
        f"modo: {r['modo']} (pedido: {r['modo_solicitado']}) · modelo: {r['modelo']} · "
        f"prompts {r['prompt_version']}"
    )
    if r["aviso"]:
        print(f"aviso: {r['aviso']}")
    for dominio, d in r["dominios"].items():
        v = d["validacion"]
        estado = "ok" if v["valida"] else f"{len(v['rechazadas'])} rechazadas"
        motivo = f" · {d['motivo']}" if d["motivo"] else ""
        print(
            f"  {dominio:<12} {d['origen']:<10} {len(d['insights'])} insights · {v['cifras']} cifras {estado}"
            f" · reintentos {d['reintentos']}{motivo}"
        )
    if r["gasto"] is not None:
        g = r["gasto"]
        c = g["corrida"]
        print(
            f"gasto estimado: USD {c['costo_usd']:.4f} en {c['solicitudes']} solicitudes "
            f"({c['tokens_entrada']:,} tokens de entrada, {c['tokens_salida']:,} de salida) · acumulado USD "
            f"{g['acumulado_usd']:.4f} de {g['presupuesto_usd']:g}"
        )
    print(f"escrito: {cfg.artifacts_dir / 'insights' / 'insights.json'}")


def _mcp(args: argparse.Namespace) -> None:
    from demanda.mcp_server import servidor

    servidor.servir(get_config().artifacts_dir, http=args.http, host=args.host, puerto=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="demanda")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_data = sub.add_parser("data", help="coloca los CSV en data/raw/ y verifica sus hashes")
    p_data.add_argument("--src", help="carpeta o zip con los CSV; sin --src solo verifica data/raw/")
    p_data.set_defaults(func=_data)

    p_calidad = sub.add_parser("calidad", help="valida contratos, escribe el panel y el reporte de calidad")
    p_calidad.set_defaults(func=_calidad)

    p_eda = sub.add_parser("eda", help="corre el notebook de EDA: reporte HTML y resumen para los agentes")
    p_eda.set_defaults(func=_eda)

    p_backtest = sub.add_parser("backtest", help="backtest de origen rodante: pronósticos, métricas y MLflow")
    p_backtest.add_argument(
        "--modelos", help="subconjunto del registro, separado por comas (p. ej. ma7,lightgbm)"
    )
    p_backtest.set_defaults(func=_backtest)

    p_ajustar = sub.add_parser("ajustar", help="ajusta LightGBM en orígenes anteriores al backtest")
    p_ajustar.set_defaults(func=_ajustar)

    p_seleccion = sub.add_parser("seleccion", help="aplica la regla de campeón a las salidas del backtest")
    p_seleccion.set_defaults(func=_seleccion)

    p_calibracion = sub.add_parser(
        "calibracion", help="pronostica con el campeón y el retador en los orígenes previos al backtest"
    )
    p_calibracion.set_defaults(func=_calibracion)

    p_decision = sub.add_parser(
        "decision", help="newsvendor: niveles de reposición y su impacto por política"
    )
    p_decision.set_defaults(func=_decision)

    p_drift = sub.add_parser(
        "drift", help="monitor de drift semana a semana, vigilancia del retador y validación"
    )
    p_drift.set_defaults(func=_drift)

    p_causal = sub.add_parser(
        "causal", help="experimento de promociones: balance, efecto, heterogeneidad, potencia y negocio"
    )
    p_causal.set_defaults(func=_causal)

    p_insights = sub.add_parser(
        "insights", help="interpreta los artefactos: agentes en vivo, caché o plantillas (insights.json)"
    )
    p_insights.add_argument(
        "--modo",
        choices=("auto", "vivo", "cache", "plantilla"),
        default="auto",
        help="auto: vivo si hay key, si no la caché (por omisión)",
    )
    p_insights.set_defaults(func=_insights)

    p_mcp = sub.add_parser(
        "mcp", help="servidor MCP de solo lectura sobre los artefactos (stdio por omisión)"
    )
    p_mcp.add_argument("--http", action="store_true", help="streamable HTTP en lugar de stdio")
    p_mcp.add_argument("--host", default="127.0.0.1", help="con --http (en Docker, 0.0.0.0)")
    p_mcp.add_argument("--port", type=int, default=8765, help="con --http")
    p_mcp.set_defaults(func=_mcp)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
