"""LightGBM global, directo y por cuantiles, reentrenado en cada corte solo con la historia del corte.

- Usa la matriz de `features.matriz`, una fila por serie × origen × h: un solo modelo cubre todas las series y
  los horizontes, porque h es una feature.
- Entrena con los orígenes T' = T − 1, T − 1 − paso, … y los objetivos d = T' + h ≤ T que tienen real. El paso
  es un costo, no un ajuste; los días de la semana de los orígenes rotan, así que se cubren todos.
- Modela y / nivel con peso nivel, donde nivel = max(mediana de los 28 días que terminan en el origen, 1). La
  pinball es homogénea de grado 1: con ese peso equivale a la pinball en unidades, y pondera por volumen como
  el WAPE. Las features en unidades entran entre el nivel, y log(nivel) entra como feature.
- Un modelo por cuantil. Si se cruzan, se ordenan, y todo se recorta en ≥ 0.

Con peso, la pinball de LightGBM renueva cada hoja con un percentil ponderado que no usa varios hilos y
domina el costo: un cuantil tarda lo mismo con 4 hilos que con 12. Por eso los cuantiles se entrenan a la vez,
con los hilos repartidos. Con `deterministic` y `force_col_wise`, el pronóstico no depende de los hilos.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any, NamedTuple

import lightgbm as lgb
import numpy as np
import polars as pl

from demanda.backtest.metricas import columna_cuantil
from demanda.backtest.protocolo import ultima_fecha
from demanda.datos.contratos import CATEGORIAS, FORMATOS, NSE, REGIONES, SERIE
from demanda.features.matriz import EN_UNIDADES, FEATURES, OBJETIVO, construir
from demanda.modelos.registro import ordenar_cuantiles, registrar

# Los festivos del calendario. Las categorías son fijas para que su código no dependa de qué aparece en cada
# corte; un nombre nuevo rompe la conversión en lugar de quedar como otra categoría en silencio.
FESTIVOS = (
    "Año Nuevo",
    "Día de la Constitución",
    "Natalicio Benito Juárez",
    "Jueves Santo",
    "Viernes Santo",
    "Día del Trabajo",
    "Día de la Independencia",
    "Día de Muertos",
    "Buen Fin",
    "Revolución Mexicana / Buen Fin",
    "Virgen de Guadalupe",
    "Nochebuena",
    "Navidad",
    "Fin de Año",
)
CATEGORICAS = {
    "category": CATEGORIAS,
    "store_format": FORMATOS,
    "region": REGIONES,
    "socioeconomic_level": NSE,
    "holiday_name": FESTIVOS,
}
COLUMNAS = (*FEATURES, "log_nivel")


@dataclass(frozen=True)
class Hiperparametros:
    num_leaves: int
    learning_rate: float
    n_estimators: int
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8


# La ganadora de `make ajuste` en los 8 orígenes anteriores al backtest, con WAPE semanal P50 de 11.18 %
# (las otras tres, de 11.19 a 11.29 %; el sistema actual, 12.58 %): artifacts/modelos/ajuste_lightgbm.json.
# Se copia aquí para que el backtest no dependa de correr el ajuste antes.
AJUSTADOS = Hiperparametros(num_leaves=31, learning_rate=0.05, n_estimators=600)


def nivel() -> pl.Expr:
    """max(mediana de 28 días al origen, 1). La mediana solo es nula si no hay ningún real en los 28 días,
    igual que la media, y entonces el nivel es 1."""
    return pl.max_horizontal(pl.col("mediana_28"), pl.lit(1.0))


def origenes_de_entrenamiento(origen: date, inicio: date, paso: int) -> list[date]:
    """T − 1, T − 1 − paso, … sin bajar de `inicio`, en orden cronológico."""
    ultimo = origen - timedelta(days=1)
    n = (ultimo - inicio).days // paso + 1
    return [ultimo - timedelta(days=paso * k) for k in range(n - 1, -1, -1)]


def entrenamiento(historia: pl.DataFrame, horizonte: int, factor: float, paso: int) -> pl.DataFrame:
    """La matriz de entrenamiento del corte: orígenes T' < T y objetivos d ≤ T que tienen real."""
    origen, inicio = ultima_fecha(historia), historia["date"].min()
    if not isinstance(inicio, date):
        raise ValueError("la historia no tiene fechas")
    origenes = origenes_de_entrenamiento(origen, inicio, paso)
    # Armada solo con la historia, la matriz no tiene reales después de T. El filtro deja explícito ese
    # invariante, para que no dependa de qué tabla reciba `construir`.
    return construir(historia, origenes, horizonte, factor).filter(
        (pl.col("date") <= origen) & pl.col(OBJETIVO).is_not_null()
    )


def del_origen(historia: pl.DataFrame, futuro: pl.DataFrame, horizonte: int, factor: float) -> pl.DataFrame:
    """La matriz del origen T, con la historia y lo conocido del futuro, cuyas medidas quedan en nulo."""
    panel = pl.concat([historia, futuro.drop("h")], how="diagonal")
    return construir(panel, [ultima_fecha(historia)], horizonte, factor)


def a_numpy(matriz: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Las `COLUMNAS` del modelo como float, con las categóricas en su código fijo, y el nivel por fila."""

    def columna(c: str) -> pl.Expr:
        if c in EN_UNIDADES:
            return pl.col(c) / pl.col("nivel")
        if c == "log_nivel":
            return pl.col("nivel").log()
        if c in CATEGORICAS:
            return pl.col(c).cast(pl.Enum(CATEGORICAS[c])).to_physical()
        return pl.col(c)

    con_nivel = matriz.with_columns(nivel().alias("nivel"))
    x = con_nivel.select(columna(c).cast(pl.Float64).alias(c) for c in COLUMNAS).to_numpy()
    return x, con_nivel["nivel"].to_numpy()


class Preparado(NamedTuple):
    """Las matrices de un corte. No dependen de los hiperparámetros: el ajuste las arma una vez por origen."""

    x: np.ndarray  # entrenamiento
    etiqueta: np.ndarray  # y / nivel
    peso: np.ndarray  # nivel
    llave: pl.DataFrame  # serie, date y h de las filas del origen T
    x_origen: np.ndarray
    nivel_origen: np.ndarray


def preparar(historia: pl.DataFrame, futuro: pl.DataFrame, factor: float, paso: int) -> Preparado:
    horizonte = futuro.select(pl.col("h").max()).item()
    matriz = entrenamiento(historia, horizonte, factor, paso)
    if matriz.is_empty():
        raise ValueError("lightgbm: la historia no alcanza para un origen de entrenamiento")
    x, peso = a_numpy(matriz)
    origen = del_origen(historia, futuro, horizonte, factor)
    x_origen, nivel_origen = a_numpy(origen)
    llave = origen.select(*SERIE, "date", "h")
    return Preparado(x, matriz[OBJETIVO].to_numpy() / peso, peso, llave, x_origen, nivel_origen)


@dataclass(frozen=True)
class LightGBMCuantiles:
    """LightGBM global y directo, un modelo por cuantil, reentrenado en cada corte con su historia."""

    factor: float
    hiper: Hiperparametros = AJUSTADOS
    cuantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    paso: int = 3
    semilla: int = 0
    hilos: int | None = None  # por cuantil; None reparte los núcleos entre los cuantiles

    def hilos_por_cuantil(self) -> int:
        return self.hilos or max(1, (os.cpu_count() or 1) // len(self.cuantiles))

    def parametros_lgb(self, tau: float) -> dict[str, Any]:
        arbol = {k: v for k, v in asdict(self.hiper).items() if k != "n_estimators"}
        return {
            "objective": "quantile",
            "alpha": tau,
            **arbol,
            "subsample_freq": 1,  # sin esto LightGBM ignora `subsample`
            "seed": self.semilla,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": self.hilos_por_cuantil(),
            "verbosity": -1,
        }

    def ajustar(self, prep: Preparado) -> dict[float, lgb.Booster]:
        def entrenar(tau: float) -> lgb.Booster:
            # Un Dataset por cuantil: compartir uno entre hilos no es seguro
            datos = lgb.Dataset(
                prep.x,
                label=prep.etiqueta,
                weight=prep.peso,
                feature_name=list(COLUMNAS),
                categorical_feature=list(CATEGORICAS),
                free_raw_data=False,
            )
            return lgb.train(self.parametros_lgb(tau), datos, num_boost_round=self.hiper.n_estimators)

        with ThreadPoolExecutor(len(self.cuantiles)) as hilos:
            return dict(zip(self.cuantiles, hilos.map(entrenar, self.cuantiles), strict=True))

    def predecir(self, prep: Preparado, futuro: pl.DataFrame) -> pl.DataFrame:
        """Entrena con las matrices ya armadas y pronostica las filas de `futuro`."""
        pred = prep.llave.with_columns(
            pl.Series(columna_cuantil(tau), modelo.predict(prep.x_origen) * prep.nivel_origen)
            for tau, modelo in self.ajustar(prep).items()
        )
        llave = [*SERIE, "date", "h"]
        return ordenar_cuantiles(futuro.select(llave).join(pred, on=llave, how="left"))

    def pronosticar(self, historia: pl.DataFrame, futuro: pl.DataFrame) -> pl.DataFrame:
        return self.predecir(preparar(historia, futuro, self.factor, self.paso), futuro)

    def parametros(self) -> dict[str, Any]:
        """Lo que identifica la configuración, para MLflow."""
        return {
            **asdict(self.hiper),
            "paso": self.paso,
            "cuantiles": list(self.cuantiles),
            "semilla": self.semilla,
        }


registrar("lightgbm", lambda ctx: LightGBMCuantiles(ctx.factor_senal, semilla=ctx.semilla))
