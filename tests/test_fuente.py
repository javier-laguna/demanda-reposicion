import hashlib
import zipfile
from pathlib import Path

import pytest

from demanda.datos import fuente


@pytest.fixture
def material(tmp_path: Path) -> tuple[Path, Path]:
    """Carpeta con CSV sintéticos (anidados como en el zip de la prueba) y su manifiesto."""
    src = tmp_path / "Prueba" / "data"
    src.mkdir(parents=True)
    lineas = []
    for nombre in fuente.ARCHIVOS:
        contenido = f"col\n{nombre}\n".encode()
        (src / nombre).write_bytes(contenido)
        lineas.append(f"{hashlib.sha256(contenido).hexdigest()}  {nombre}")
    manifiesto = tmp_path / "manifest.sha256"
    manifiesto.write_text("\n".join(lineas) + "\n")
    return src, manifiesto


def test_prepara_desde_carpeta(material, tmp_path):
    src, manifiesto = material
    hashes = fuente.preparar(src.parent, tmp_path / "raw", manifiesto)
    assert set(hashes) == set(fuente.ARCHIVOS)


def test_prepara_desde_zip(material, tmp_path):
    src, manifiesto = material
    zip_path = tmp_path / "material.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for nombre in fuente.ARCHIVOS:
            z.write(src / nombre, f"Prueba/data/{nombre}")
    hashes = fuente.preparar(zip_path, tmp_path / "raw", manifiesto)
    assert set(hashes) == set(fuente.ARCHIVOS)


def test_rechaza_archivo_alterado(material, tmp_path):
    src, manifiesto = material
    (src / "stores.csv").write_text("col\nalterado\n")
    with pytest.raises(ValueError, match=r"stores\.csv"):
        fuente.preparar(src, tmp_path / "raw", manifiesto)


def test_falta_archivo(material, tmp_path):
    src, manifiesto = material
    (src / "calendar.csv").unlink()
    with pytest.raises(FileNotFoundError, match=r"calendar\.csv"):
        fuente.preparar(src, tmp_path / "raw", manifiesto)
