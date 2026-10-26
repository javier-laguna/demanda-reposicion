"""Coloca los CSV de la prueba en `data/raw/` y verifica que sean los originales.

Los datos no se versionan. El repo solo guarda `data/manifest.sha256`; este módulo copia los archivos desde
una carpeta o desde el zip recibido y comprueba cada hash contra ese manifiesto.
"""

import hashlib
import shutil
import zipfile
from pathlib import Path

ARCHIVOS = ("calendar.csv", "stores.csv", "transactions.csv")


def leer_manifiesto(ruta: Path) -> dict[str, str]:
    hashes = {}
    for linea in ruta.read_text().splitlines():
        if linea.strip():
            digest, nombre = linea.split(maxsplit=1)
            hashes[nombre.strip()] = digest
    return hashes


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _copiar_desde_carpeta(src: Path, destino: Path) -> None:
    for nombre in ARCHIVOS:
        encontrados = sorted(src.rglob(nombre))
        if not encontrados:
            raise FileNotFoundError(f"No encontré {nombre} dentro de {src}")
        shutil.copyfile(encontrados[0], destino / nombre)


def _copiar_desde_zip(src: Path, destino: Path) -> None:
    with zipfile.ZipFile(src) as z:
        for nombre in ARCHIVOS:
            miembros = [m for m in z.namelist() if Path(m).name == nombre and not m.startswith("__MACOSX")]
            if not miembros:
                raise FileNotFoundError(f"No encontré {nombre} dentro de {src}")
            with z.open(miembros[0]) as origen, (destino / nombre).open("wb") as salida:
                shutil.copyfileobj(origen, salida)


def preparar(src: Path, destino: Path, manifiesto: Path) -> dict[str, str]:
    """Copia los CSV de `src` (carpeta o zip) a `destino` y devuelve sus hashes, ya verificados."""
    destino.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        _copiar_desde_carpeta(src, destino)
    elif zipfile.is_zipfile(src):
        _copiar_desde_zip(src, destino)
    else:
        raise FileNotFoundError(f"{src} no es una carpeta ni un zip")
    return verificar(destino, manifiesto)


def verificar(carpeta: Path, manifiesto: Path) -> dict[str, str]:
    """Comprueba que cada CSV de `carpeta` tenga el hash del manifiesto."""
    esperados = leer_manifiesto(manifiesto)
    obtenidos = {}
    for nombre in ARCHIVOS:
        ruta = carpeta / nombre
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta}; corre `make data DATA_SRC=...`")
        obtenidos[nombre] = sha256(ruta)
        if obtenidos[nombre] != esperados.get(nombre):
            raise ValueError(f"{nombre} no coincide con el manifiesto: no es el archivo original")
    return obtenidos
