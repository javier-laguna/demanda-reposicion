"""PSI y pruebas binomiales del monitor contra casos calculados a mano."""

import numpy as np
import pytest

from demanda.drift.estadisticos import PISO, binomial, cortes, psi


def test_psi_a_mano():
    # Referencia 1..10: los cortes son 1..9 y cada intervalo tiene una décima. La semana pone la mitad en el
    # primer intervalo y la mitad en el último; los otros ocho quedan en el piso
    referencia = np.arange(1, 11, dtype=float)
    assert cortes(referencia).tolist() == list(range(1, 10))
    actual = np.array([1.0] * 5 + [10.0] * 5)
    esperado = 2 * (0.5 - 0.1) * np.log(0.5 / 0.1) + 8 * (PISO - 0.1) * np.log(PISO / 0.1)
    assert psi(referencia, actual) == pytest.approx(esperado)
    # Con 20 valores los cortes caen en los pares: los cuantiles no se interpolan
    assert cortes(np.arange(1, 21, dtype=float)).tolist() == list(range(2, 20, 2))


def test_psi_es_cero_con_distribuciones_iguales_y_no_cambia_con_log():
    rng = np.random.default_rng(0)
    referencia, actual = rng.lognormal(0, 0.3, 500), rng.lognormal(0.1, 0.4, 200)
    assert psi(referencia, referencia) == 0
    valor = psi(referencia, actual)
    assert valor is not None and valor > 0
    assert psi(np.log(referencia), np.log(actual)) == pytest.approx(psi(referencia, actual), abs=1e-12)
    # Una transformación creciente cualquiera tampoco lo cambia
    assert psi(referencia**3, actual**3) == pytest.approx(psi(referencia, actual), abs=1e-12)
    assert psi([], actual) is None and psi(referencia, []) is None


def test_binomiales_a_mano():
    # Una cola: P(X ≥ 3 | n = 3, p = ½) = ⅛; dos colas, por simetría, ¼
    assert binomial(3, 3, 0.5, 1) == pytest.approx(0.125)
    assert binomial(3, 3, 0.5, 2) == pytest.approx(0.25)
    # P(X ≥ 0) = 1
    assert binomial(0, 10, 0.3, 1) == pytest.approx(1.0)
    # Sin observaciones no hay evidencia; con p en 0, un caso es imposible
    assert binomial(0, 0, 0.3, 2) == 1.0
    assert binomial(0, 10, 0.0, 1) == 1.0 and binomial(1, 10, 0.0, 1) == 0.0
    assert binomial(10, 10, 1.0, 2) == 1.0 and binomial(9, 10, 1.0, 2) == 0.0
