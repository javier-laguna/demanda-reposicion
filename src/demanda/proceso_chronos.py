"""Chronos-2 en un proceso propio, con su propio runtime de OpenMP.

torch y LightGBM traen cada uno su `libomp`. En macOS, con los dos cargados en el mismo proceso, el primero
que corre en paralelo después de cargarse el otro termina en segfault, sin importar el orden de importación.
Por eso el pipeline vive en un proceso hijo que carga los pesos una vez y atiende cada predicción.

El hijo es `python -m demanda.proceso_chronos`, no un `multiprocessing` con `spawn`: este reimporta el módulo
principal del padre, y si ese módulo importa `demanda.modelos`, el hijo carga LightGBM. Por la misma razón,
este módulo está fuera de `demanda.modelos`. Padre e hijo intercambian tuplas con pickle por stdin y stdout;
lo que impriman las bibliotecas en el hijo va a stderr.
"""

import argparse
import contextlib
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as Vencido
from typing import IO, Any

MODELO = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
TIMEOUT_S = 600.0  # por llamada, incluida la carga de los pesos; un corte del backtest tarda ~20 s


def _enviar(canal: IO[bytes], mensaje: Any) -> None:
    pickle.dump(mensaje, canal, protocol=pickle.HIGHEST_PROTOCOL)
    canal.flush()


class PipelineAislado:
    """La interfaz `predict_df` de `Chronos2Pipeline`, atendida por un proceso hijo con los pesos cargados.

    Cada llamada espera a lo más `timeout` segundos; si se vence, el hijo se termina. Cualquier falla del hijo
    (un error del pipeline, su muerte o un vencimiento) llega como `RuntimeError` con el motivo.
    """

    def __init__(self, solo_local: bool = False, timeout: float = TIMEOUT_S) -> None:
        self.timeout = timeout
        # -P: el directorio actual no entra al sys.path del hijo
        comando = [sys.executable, "-P", "-m", __name__, *(["--solo-local"] if solo_local else [])]
        self._hijo = subprocess.Popen(comando, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        # Las respuestas se leen en un hilo para poder esperarlas con timeout
        self._lector = ThreadPoolExecutor(max_workers=1)
        try:
            self._recibir()  # espera a que cargue los pesos: una falla de descarga sale aquí
        except RuntimeError:
            self.cerrar(timeout=1)
            raise

    def activo(self) -> bool:
        return self._hijo.poll() is None

    def _murio(self, momento: str) -> RuntimeError:
        try:
            codigo = self._hijo.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._hijo.kill()
            codigo = self._hijo.wait()
        return RuntimeError(f"el proceso de Chronos-2 terminó {momento} (código {codigo}); ver su stderr")

    def _recibir(self) -> Any:
        assert self._hijo.stdout is not None
        respuesta = self._lector.submit(pickle.load, self._hijo.stdout)
        try:
            estado, valor = respuesta.result(timeout=self.timeout)
        except Vencido:
            self._hijo.kill()
            self._hijo.wait()
            raise RuntimeError(
                f"el proceso de Chronos-2 no respondió en {self.timeout:g} s; se terminó"
            ) from None
        except (EOFError, pickle.UnpicklingError, OSError) as e:
            raise self._murio("sin responder") from e
        if estado == "error":
            raise RuntimeError(f"Chronos-2 falló en su proceso: {valor}")
        return valor

    def predict_df(self, df: Any, future_df: Any = None, **kwargs: Any) -> Any:
        assert self._hijo.stdin is not None
        try:
            _enviar(self._hijo.stdin, (df, future_df, kwargs))
        except (BrokenPipeError, OSError, ValueError) as e:  # ValueError: el pipe ya estaba cerrado
            raise self._murio("antes de recibir la petición") from e
        return self._recibir()

    def cerrar(self, timeout: float = 10) -> None:
        """Cierra la entrada del hijo, que entonces termina; si no termina en `timeout` s, lo mata."""
        stdin = self._hijo.stdin
        if stdin is not None and not stdin.closed:
            with contextlib.suppress(OSError):  # el hijo ya había muerto
                stdin.close()
        try:
            self._hijo.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._hijo.kill()
            self._hijo.wait(timeout=timeout)
        self._lector.shutdown(wait=False)

    def __enter__(self) -> "PipelineAislado":
        return self

    def __exit__(self, *_: object) -> None:
        self.cerrar()

    def __del__(self) -> None:
        if hasattr(self, "_lector"):
            # Al apagar el intérprete puede faltar ya cualquier cosa
            with contextlib.suppress(Exception):
                self.cerrar(timeout=1)


def main(argv: list[str] | None = None) -> None:
    """El hijo: carga los pesos, avisa y responde cada `(df, future_df, kwargs)` hasta que se cierra stdin."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--solo-local", action="store_true", help="solo leer los pesos de la caché local")
    args = parser.parse_args(argv)
    salida = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # stdout queda solo para el protocolo
    try:
        from chronos import Chronos2Pipeline

        pipeline = Chronos2Pipeline.from_pretrained(
            MODELO, revision=REVISION, device_map="cpu", local_files_only=args.solo_local
        )
    except Exception as e:
        _enviar(salida, ("error", repr(e)))
        return
    _enviar(salida, ("ok", None))
    while True:
        try:
            df, future_df, kwargs = pickle.load(sys.stdin.buffer)
        except EOFError:
            return
        try:
            _enviar(salida, ("ok", pipeline.predict_df(df, future_df=future_df, **kwargs)))
        except Exception as e:
            _enviar(salida, ("error", repr(e)))


if __name__ == "__main__":
    main()
