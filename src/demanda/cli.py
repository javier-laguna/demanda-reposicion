"""Punto de entrada: `uv run demanda <comando>`. Cada etapa del pipeline agrega aquí su comando."""

import argparse
import sys
from pathlib import Path

from demanda.config import get_config
from demanda.datos import fuente


def _data(args: argparse.Namespace) -> None:
    cfg = get_config()
    manifiesto = cfg.data_dir / "manifest.sha256"
    if args.src:
        hashes = fuente.preparar(Path(args.src), cfg.raw_dir, manifiesto)
    else:
        hashes = fuente.verificar(cfg.raw_dir, manifiesto)
    for nombre, digest in hashes.items():
        print(f"ok  {nombre}  {digest[:12]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="demanda")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_data = sub.add_parser("data", help="coloca los CSV en data/raw/ y verifica sus hashes")
    p_data.add_argument("--src", help="carpeta o zip con los CSV; sin --src solo verifica data/raw/")
    p_data.set_defaults(func=_data)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
