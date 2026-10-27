"""El validador: extrae las cifras de los textos, acepta las que salen de las herramientas (con redondeo, en
%, en millones y con el signo invertido) y rechaza las inventadas."""

import math
import random
import re

import pytest

from demanda.agentes.modelos import Evidencia, Insight
from demanda.agentes.validador import Permitidas, analizar, extraer, validar

RESPUESTA = {
    "lightgbm": {"wape_pct": 20.3353, "fraccion_critica": 0.9091},
    "costo": 212717514.08,
    "diferencia_pct": -39.5943,
    "texto": "el orden se mantiene en las 4 fracciones; 91,771 orígenes",
}


def _insight(hallazgo: str, cifra: str = "—", herramienta: str = "get_metrics") -> Insight:
    return Insight(
        titulo="t",
        hallazgo=hallazgo,
        evidencia=[Evidencia(cifra=cifra, que_mide="algo", herramienta=herramienta)],  # pyrefly: ignore[bad-argument-type]
        accion="a",
        confianza="media",
    )


def _valida(hallazgo: str) -> bool:
    return validar([_insight(hallazgo)], Permitidas([RESPUESTA]), ["get_metrics"]).valida


def test_extrae_signo_decimales_miles_y_sufijos():
    cifras = extraer(
        "cae −6.96 pp, sube +9.6 %, cuesta 213 M, 1.2 millones, 5 mil y 91,771 casos; p = 0.166."
    )
    assert [(c.valor, c.decimales, c.sufijo) for c in cifras] == [
        (-6.96, 2, "pp"),
        (9.6, 1, "%"),
        (213.0, 0, "M"),
        (1.2, 1, "M"),
        (5.0, 0, "mil"),
        (91771.0, 0, None),
        (0.166, 3, None),
    ]


def test_ignora_fechas_identificadores_y_anios():
    texto = "El 2023-11-17 y en 2023-12, STR_055, P10, E5, h=7, t+1, t−5 y lag_1 en 2023 y 2024."
    assert extraer(texto) == []


@pytest.mark.parametrize(
    "hallazgo",
    [
        "El WAPE de LightGBM es 20.34 %.",  # redondeo de 20.3353
        "El WAPE de LightGBM es 20.3 %.",
        "La fracción crítica es 90.91 %.",  # fracción a %
        "El costo es de 213 M de pesos.",  # millones
        "El costo es de 212.7 millones.",
        "Cuesta 39.6 % menos.",  # signo invertido de −39.5943
        "Se mantiene en las 4 fracciones de 91,771 orígenes.",  # cifras dentro de textos
    ],
)
def test_acepta_cifras_presentes(hallazgo):
    assert _valida(hallazgo)


@pytest.mark.parametrize("hallazgo", ["El WAPE de LightGBM es 15.7 %.", "Mejora 6.96 pp.", "Cuesta 214 M."])
def test_rechaza_cifras_inventadas_o_derivadas(hallazgo):
    # 6.96 pp sería 27.29 − 20.34: una diferencia que no está en la respuesta
    assert not _valida(hallazgo)


def test_rechazo_dice_cifra_campo_e_insight():
    v = validar(
        [_insight("ok 20.34 %"), _insight("WAPE de 15.7 %", cifra="15.7 %")],
        Permitidas([RESPUESTA]),
        ["get_metrics"],
    )
    assert not v.valida
    assert v.cifras == 3
    assert [(r.cifra, r.campo, r.insight) for r in v.rechazadas] == [
        ("15.7 %", "hallazgo", 2),
        ("15.7 %", "evidencia[0].cifra", 2),
    ]


def test_sin_cifras_es_valido():
    v = validar(
        [_insight("El 2023-11-17 falló STR_055 y P10 quedó corto.")], Permitidas([RESPUESTA]), ["get_metrics"]
    )
    assert v.valida and v.cifras == 0


def test_rechaza_herramienta_fuera_del_dominio():
    v = validar(
        [_insight("20.34 %", herramienta="get_promo_effect")], Permitidas([RESPUESTA]), ["get_metrics"]
    )
    assert [r.motivo for r in v.rechazadas] == ["get_promo_effect no es una herramienta del dominio"]


# --- todo dígito se revisa -----------------------------------------------------------------------------

_FECHA_ISO = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")
_TROZOS = [
    *"0123456789" * 3,
    *".,-−+ %=()[]_xaM",
    " pp",
    " mil",
    " millones",
    "2023-11-17",
    "2024-02",
    "2023",
    "2024",
    "STR_055",
    "IC95",
    "h=7",
    "t+1",
    "1,234.5",
]


def _sin_revisar(texto: str) -> list[int]:
    """Los dígitos que no caen ni en una cifra extraída ni en un tramo ignorado legítimo."""
    cifras, ignorados = analizar(texto)
    cubiertos = {i for c in cifras for i in range(c.inicio, c.fin)}
    for g in ignorados:
        tramo = texto[g.inicio : g.fin]
        antes = texto[g.inicio - 1] if g.inicio > 0 else " "
        antes2 = texto[g.inicio - 2] if g.inicio > 1 else " "
        despues = texto[g.fin] if g.fin < len(texto) else " "
        es_letra = lambda c: c.isalpha() or c == "_"  # noqa: E731
        legitimo = {
            "fecha": _FECHA_ISO.fullmatch(tramo) is not None,
            "año": tramo in ("2023", "2024"),
            "identificador": es_letra(antes) or (antes in "=+-−" and es_letra(antes2)) or es_letra(despues),
        }[g.tipo]
        assert legitimo, (texto, g)
        cubiertos |= set(range(g.inicio, g.fin))
    return [i for i, ch in enumerate(texto) if re.fullmatch(r"\d", ch) and i not in cubiertos]


@pytest.mark.parametrize(
    "texto",
    [
        "15,7 %",
        "IC95 [0.05,98.76]",
        "0.61-98.76 %",
        "de 0.61–2.31",
        "1.2.3 y .5",
        "(11.0,13.4)",
        "1 234 y 12,34 y 1,2345",
        "El 2023-11-17, STR_055, P10, h=7, t+1, t−5 y 2023-2024",
        "10x, 2.65× y 213M",
    ],
)
def test_todo_digito_pertenece_a_una_cifra_o_a_un_tramo_ignorado(texto):
    assert _sin_revisar(texto) == []


def test_invariante_sobre_textos_aleatorios():
    rng = random.Random(0)
    for _ in range(3_000):
        texto = "".join(rng.choice(_TROZOS) for _ in range(rng.randint(1, 25)))
        assert _sin_revisar(texto) == [], texto


@pytest.mark.parametrize("hallazgo", ["El WAPE es 15,7 %.", "IC95 [0.05,98.76]", "IC95 [0.05,20.34]"])
def test_una_coma_que_no_es_de_miles_rechaza_la_cifra(hallazgo):
    # Ni el decimal con coma ni la lista sin espacio se aceptan, aunque sus partes estén permitidas
    p = Permitidas([{"a": 15.0, "b": 7.0, "c": 15.7, "d": 0.05, "e": 20.3353}])
    v = validar([_insight(hallazgo)], p, ["get_metrics"])
    assert not v.valida
    assert "mal escrito" in v.rechazadas[0].motivo


def test_un_rango_revisa_las_dos_cifras():
    p = Permitidas([{"a": 0.61, "b": 2.31}])
    assert validar([_insight("de 0.61-2.31 %")], p, ["get_metrics"]).valida
    assert not validar([_insight("de 0.61-98.76 %")], p, ["get_metrics"]).valida


@pytest.mark.parametrize(
    ("cifra", "valor", "pasa"),
    [
        ("90.91 %", 0.9091, True),  # fracción a %
        ("9.1 pp", 0.091, True),
        ("90.91", 0.9091, False),  # sin % no se multiplica por 100
        ("302.35 %", 3.0235, False),  # 3.0235 no es una fracción
        ("3.02 %", 3.0235, True),
    ],
)
def test_fraccion_a_porcentaje_solo_para_fracciones_citadas_en_porcentaje(cifra, valor, pasa):
    assert Permitidas([valor]).coincide(extraer(cifra)[0]) is pasa


def test_mil_escala_por_mil():
    assert Permitidas([91771]).coincide(extraer("91.8 mil")[0])
    assert not Permitidas([91_800_000]).coincide(extraer("91.8 mil")[0])  # 91.8 millones no son 91.8 mil


def test_acepta_una_cifra_que_solo_esta_en_una_clave():
    assert validar([_insight("Con 4 semanas basta.")], Permitidas([{"4": 2.1}]), ["get_metrics"]).valida
    assert not validar([_insight("Con 5 semanas basta.")], Permitidas([{"4": 2.1}]), ["get_metrics"]).valida


def test_las_cifras_mal_escritas_de_una_respuesta_no_entran_a_las_permitidas():
    p = Permitidas([{"texto": "lista 0.05,98.76 sin espacio"}, 5.0, 10.0, 20.0])
    assert all(math.isfinite(v) for v in p.valores)
    assert p.coincide(extraer("10")[0]) and p.coincide(extraer("20")[0])
    assert not p.coincide(extraer("98.76")[0])


def test_un_punto_antes_de_la_cifra_la_deja_mal_escrita():
    # «.5 %» no es 5 % aunque 5 esté permitido; «0.5 %» sí se revisa como 0.5
    p = Permitidas([{"a": 5.0, "b": 0.5}])
    punto = extraer("subió .5 %")[0]
    assert not punto.bien_escrita and not p.coincide(punto)
    assert p.coincide(extraer("subió 0.5 %")[0]) and p.coincide(extraer("subió 5 %")[0])
