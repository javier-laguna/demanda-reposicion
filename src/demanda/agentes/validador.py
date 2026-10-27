"""El validador determinista: toda cifra de un insight tiene que aparecer en las herramientas que se leyeron.

- **Permitidas:** los números de las respuestas exitosas de las herramientas, tanto los valores numéricos
  del JSON como los que vienen dentro de textos y claves. Los mensajes de error no cuentan: repetirían un
  argumento inventado.
- **Extracción:** de todos los campos de texto del `Insight`, evidencia incluida, se toman los números con
  signo, decimales, separador de miles y los sufijos `%`, `pp`, `M`/`millones` y `mil`. Se ignoran las
  fechas ISO, los números pegados a letras (`STR_055`, `P10`, `E5`, `h=7`, `t+1`) y los años 2023 y 2024.
  Ningún otro dígito queda sin revisar:
  - una coma es separador de miles solo si la siguen exactamente 3 dígitos y el número termina ahí o en su
    parte decimal; cualquier otra coma entre dígitos («15,7», «0.05,98.76») deja la cifra mal escrita, y
    se rechaza;
  - un guion entre dos números es un rango: se revisan los dos («0.61-2.31»).
- **Coincidencia:** una cifra x con d decimales coincide con un valor v si |x| = |round(v', d)|, con v'
  igual a v, 100·v (fracción a %, solo si x trae `%` o `pp` y |v| ≤ 1), v / 10⁶ (si x trae `M`) o v / 10³
  (si trae `mil`), y la tolerancia de redondeo |x − v'| ≤ 0.5·10⁻ᵈ. Se compara el valor absoluto: «cuesta
  39.6 % menos» cita un −39.6.

Las cifras derivadas (diferencias o razones que no están en los artefactos) se rechazan a propósito.
"""

import bisect
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart, ToolReturnPart

from demanda.agentes.modelos import Insight, Rechazo, Validacion

ANIOS = frozenset({"2023", "2024"})
_FECHA = re.compile(r"(?<!\d)\d{4}-\d{2}(?:-\d{2})?(?!\d)")
# Un tramo de dígitos con sus puntos y comas internos; si está bien escrito, es un número
_TRAMO = re.compile(r"\d+(?:[.,]\d+)*")
_BIEN_ESCRITO = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_SUFIJO = re.compile(r"\s?(%|pp|millones|mil|M)(?![^\W\d])")
_SIGNOS = "+-−"
# Holgura numérica sobre la tolerancia de redondeo, para la aritmética de punto flotante
_EPS = 1e-9


@dataclass(frozen=True)
class Cifra:
    texto: str
    valor: float  # nan si la cifra está mal escrita
    decimales: int
    sufijo: str | None  # "%", "pp", "M" (también «millones») o "mil"
    inicio: int = 0
    fin: int = 0
    bien_escrita: bool = True


@dataclass(frozen=True)
class Ignorado:
    """Un tramo de dígitos que no es una cifra: una fecha, un identificador o un año."""

    inicio: int
    fin: int
    tipo: Literal["fecha", "identificador", "año"]


def _es_letra(c: str) -> bool:
    return c.isalpha() or c == "_"


def analizar(texto: str) -> tuple[list[Cifra], list[Ignorado]]:
    """Las cifras de un texto, en orden de aparición, y los tramos de dígitos que se ignoran. Todo dígito
    del texto cae en uno de los dos."""
    ignorados = [Ignorado(m.start(), m.end(), "fecha") for m in _FECHA.finditer(texto)]
    limpio = _FECHA.sub(lambda m: " " * len(m.group()), texto)
    cifras = []
    for m in _TRAMO.finditer(limpio):
        inicio, fin = m.span()
        crudo = m.group()
        antes = limpio[inicio - 1] if inicio > 0 else " "
        antes2 = limpio[inicio - 2] if inicio > 1 else " "
        sufijo = _SUFIJO.match(limpio, fin)
        despues = limpio[fin] if fin < len(limpio) else " "
        # Pegado a una letra (STR_055, P10, 7d) o a un operador pegado a una letra (h=7, t+1, t−5)
        if (
            _es_letra(antes)
            or (antes in "=" + _SIGNOS and _es_letra(antes2))
            or (sufijo is None and _es_letra(despues))
        ):
            ignorados.append(Ignorado(inicio, fin, "identificador"))
            continue
        if sufijo is None and crudo in ANIOS:
            ignorados.append(Ignorado(inicio, fin, "año"))
            continue
        # Entre dos números, el guion es un rango y no un signo: «0.61-2.31» revisa 0.61 y 2.31
        signo = antes if antes in _SIGNOS and not antes2.isdigit() else ""
        desde = inicio - len(signo)
        hasta = fin if sufijo is None else sufijo.end()
        unidad = None if sufijo is None else ("M" if sufijo.group(1) == "millones" else sufijo.group(1))
        # Un punto antes del tramo («.5») o una coma que no es de miles dejan la cifra mal escrita
        if antes == "." or _BIEN_ESCRITO.fullmatch(crudo) is None:
            cifras.append(Cifra(limpio[desde:hasta], math.nan, 0, unidad, desde, hasta, bien_escrita=False))
            continue
        numero = crudo.replace(",", "")
        decimales = len(numero.split(".")[1]) if "." in numero else 0
        valor = -float(numero) if signo in ("-", "−") else float(numero)
        cifras.append(Cifra(limpio[desde:hasta], valor, decimales, unidad, desde, hasta))
    return cifras, sorted(ignorados, key=lambda g: g.inicio)


def extraer(texto: str) -> list[Cifra]:
    """Las cifras de un texto, en orden de aparición."""
    return analizar(texto)[0]


def _numeros(x: Any) -> Iterable[float]:
    if isinstance(x, bool) or x is None:
        return
    if isinstance(x, int | float):
        if math.isfinite(x):
            yield float(x)
    elif isinstance(x, str):
        for c in extraer(x):
            if not c.bien_escrita:
                continue
            yield c.valor
            if c.sufijo == "M":
                yield c.valor * 1e6
            elif c.sufijo == "mil":
                yield c.valor * 1e3
    elif isinstance(x, Mapping):
        for k, v in x.items():
            yield from _numeros(str(k))
            yield from _numeros(v)
    elif isinstance(x, list | tuple):
        for v in x:
            yield from _numeros(v)


class Permitidas:
    """Los valores absolutos de las cifras de unas respuestas, ordenados para buscar por intervalo."""

    def __init__(self, respuestas: Iterable[Any]) -> None:
        self.valores = sorted({abs(v) for r in respuestas for v in _numeros(r)})

    def __len__(self) -> int:
        return len(self.valores)

    def _hay_en(self, centro: float, radio: float, maximo: float = math.inf) -> bool:
        i = bisect.bisect_left(self.valores, centro - radio)
        return i < len(self.valores) and self.valores[i] <= min(centro + radio, maximo)

    def coincide(self, c: Cifra) -> bool:
        """Si |x| sale de algún v permitido, redondeado a los decimales de x (ver el docstring del módulo)."""
        if not c.bien_escrita:
            return False
        x = abs(c.valor)
        radio = 0.5 * 10 ** (-c.decimales)
        # Cada transformación v' = v · f equivale a buscar v en |x| / f con el radio escalado; la de
        # fracción a % solo vale para fracciones (|v| ≤ 1) citadas en % o pp
        factores = [(1.0, math.inf)]
        if c.sufijo in ("%", "pp"):
            factores.append((100.0, 1.0))
        if c.sufijo == "M":
            factores.append((1e-6, math.inf))
        elif c.sufijo == "mil":
            factores.append((1e-3, math.inf))
        return any(self._hay_en(x / f, radio / f + _EPS * max(1.0, x / f), maximo) for f, maximo in factores)


def textos(insight: Insight) -> list[tuple[str, str]]:
    """Todos los campos de texto de un insight, con su nombre."""
    campos = [("titulo", insight.titulo), ("hallazgo", insight.hallazgo), ("accion", insight.accion)]
    for i, e in enumerate(insight.evidencia):
        campos += [(f"evidencia[{i}].cifra", e.cifra), (f"evidencia[{i}].que_mide", e.que_mide)]
    return campos


def validar(insights: Sequence[Insight], permitidas: Permitidas, herramientas: Iterable[str]) -> Validacion:
    """Revisa cada cifra de cada insight y que la evidencia cite herramientas del dominio."""
    validas = set(herramientas)
    rechazadas, revisadas = [], 0
    for n, insight in enumerate(insights, start=1):
        for campo, texto in textos(insight):
            for c in extraer(texto):
                revisadas += 1
                if not permitidas.coincide(c):
                    motivo = (
                        "no aparece en las herramientas"
                        if c.bien_escrita
                        else "número mal escrito: usa punto decimal, coma solo para miles y separa las cifras"
                    )
                    rechazadas.append(Rechazo(cifra=c.texto, campo=campo, insight=n, motivo=motivo))
        for i, e in enumerate(insight.evidencia):
            if e.herramienta not in validas:
                rechazadas.append(
                    Rechazo(
                        cifra=e.cifra,
                        campo=f"evidencia[{i}].herramienta",
                        insight=n,
                        motivo=f"{e.herramienta} no es una herramienta del dominio",
                    )
                )
    return Validacion(valida=not rechazadas, cifras=revisadas, rechazadas=rechazadas)


def respuestas_de(
    mensajes: Sequence[ModelMessage], nombres: Iterable[str]
) -> list[tuple[str, dict[str, Any], Any]]:
    """Las respuestas exitosas de las herramientas `nombres` en un historial de Pydantic AI, con su nombre y
    argumentos. Deja fuera la herramienta de salida, que Pydantic AI también registra como retorno."""
    validas = set(nombres)
    argumentos: dict[str, dict[str, Any]] = {}
    for m in mensajes:
        for p in m.parts:
            if isinstance(p, ToolCallPart):
                argumentos[p.tool_call_id] = p.args_as_dict()
    salida = []
    for m in mensajes:
        if not isinstance(m, ModelRequest):
            continue
        for p in m.parts:
            if isinstance(p, ToolReturnPart) and p.outcome == "success" and p.tool_name in validas:
                salida.append((p.tool_name, argumentos.get(p.tool_call_id, {}), p.content))
    return salida
