"""Punto de entrada: `uv run demanda <comando>`. Cada etapa del pipeline agrega aquí su comando."""

import argparse
import sys
from pathlib import Path

from demanda.config import get_config
from demanda.datos import calidad, fuente


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="demanda")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_data = sub.add_parser("data", help="coloca los CSV en data/raw/ y verifica sus hashes")
    p_data.add_argument("--src", help="carpeta o zip con los CSV; sin --src solo verifica data/raw/")
    p_data.set_defaults(func=_data)

    p_calidad = sub.add_parser("calidad", help="valida contratos, escribe el panel y el reporte de calidad")
    p_calidad.set_defaults(func=_calidad)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
