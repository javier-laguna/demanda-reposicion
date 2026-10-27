"""La caché de insights (`insights/cache.json`): el único derivado de los datos que se versiona.

- **Contenido, por dominio:** los insights validados, el modelo, `PROMPT_VERSION`, el hash de los prompts,
  las llamadas a herramientas que hizo el agente (únicas y en orden), la huella de los datos y, por
  trazabilidad, el sha256 de cada artefacto leído. La entrada del orquestador guarda además la firma de las
  tres entradas de dominio que resumió.
- **Huella:** el sha256 de las respuestas de esas llamadas, con cada flotante redondeado a 3 cifras
  significativas. No se usa el hash de los archivos porque los artefactos difieren en ~1e-6 entre
  plataformas, y la caché debe valer en el Docker del revisor. La escritura y la revalidación usan la misma
  lista de llamadas únicas, aunque el agente haya repetido alguna en el reintento.
- **Validez:** una entrada vale si coinciden la versión y el hash de los prompts y la huella, si todas sus
  llamadas son herramientas del dominio y si el validador acepta otra vez todas sus cifras contra las
  respuestas actuales. La del orquestador además necesita que valgan las tres de dominio y que sean las
  mismas que resumió.
- **Una corrida por caché:** cada corrida en vivo con algún dominio en vivo reescribe el archivo completo.
  Un dominio que cae a plantilla no deja su entrada de una corrida anterior, y el orquestador solo se guarda
  si los tres dominios salieron en vivo en esa corrida. Si ningún dominio sale en vivo, la caché no cambia.

Solo guarda conclusiones agregadas, nunca filas ni series: `get_forecast` no está en ningún dominio.
"""

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from demanda.agentes import validador
from demanda.agentes.modelos import DOMINIOS, ORDEN, ORQUESTADOR, Insight, Llamada, ResultadoDominio
from demanda.agentes.prompts import PROMPT_VERSION, hash_prompts
from demanda.datos.fuente import sha256
from demanda.mcp_server import herramientas

CIFRAS_HUELLA = 3
DESCRIPCION = (
    "Insights validados de la última corrida en vivo (make insights MODO=vivo). Solo conclusiones agregadas; "
    "make insights los usa sin key mientras la huella de los datos y los prompts coincidan."
)

Respuestas = Sequence[tuple[Llamada, Any]]


def redondear(x: Any, cifras: int = CIFRAS_HUELLA) -> Any:
    """Cada flotante de una estructura JSON redondeado a `cifras` cifras significativas."""
    if isinstance(x, Mapping):
        return {str(k): redondear(v, cifras) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [redondear(v, cifras) for v in x]
    if isinstance(x, float) and math.isfinite(x):
        return float(f"{x:.{cifras}g}")
    return x


def unicas(llamadas: Iterable[Llamada]) -> list[Llamada]:
    """Las llamadas sin repetir, en un orden estable."""
    por_clave = {ll.clave(): ll for ll in llamadas}
    return [por_clave[k] for k in sorted(por_clave)]


def respuestas_unicas(respuestas: Respuestas) -> list[tuple[Llamada, Any]]:
    """Una respuesta por llamada, en el orden de `unicas`: lo que se escribe y lo que se revalida."""
    por_clave = {ll.clave(): (ll, r) for ll, r in respuestas}
    return [por_clave[k] for k in sorted(por_clave)]


def responder(artefactos: Path, llamadas: Sequence[Llamada]) -> list[tuple[Llamada, dict[str, Any]]]:
    """Las respuestas actuales de las herramientas, llamadas directamente."""
    return [(ll, herramientas.llamar(artefactos, ll.herramienta, ll.argumentos)) for ll in unicas(llamadas)]


def huella(respuestas: Respuestas) -> str:
    contenido = [
        {"herramienta": ll.herramienta, "argumentos": ll.argumentos, "respuesta": redondear(r)}
        for ll, r in respuestas_unicas(respuestas)
    ]
    texto = json.dumps(contenido, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(texto.encode()).hexdigest()


def firma(e: Mapping[str, Any]) -> str:
    """El sha256 de una entrada de caché: el orquestador lo guarda de cada dominio que resumió."""
    return hashlib.sha256(json.dumps(e, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def artefactos_sha256(artefactos: Path, llamadas: Iterable[Llamada]) -> dict[str, str]:
    rutas = sorted({f for ll in llamadas for f in herramientas.FUENTES[ll.herramienta]})
    return {r: sha256(artefactos / r) for r in rutas}


def entrada(
    artefactos: Path,
    resultado: ResultadoDominio,
    respuestas: Respuestas,
    entradas_de_dominio: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """La entrada de caché de un dominio que salió en vivo; la del orquestador lleva la firma de las
    entradas de dominio que resumió."""
    llamadas = [ll for ll, _ in respuestas_unicas(respuestas)]
    e: dict[str, Any] = {
        "insights": [i.model_dump() for i in resultado.insights],
        "modelo": resultado.modelo,
        "prompt_version": PROMPT_VERSION,
        "prompts_sha256": hash_prompts(),
        "llamadas": [ll.model_dump() for ll in llamadas],
        "huella": huella(respuestas),
        "artefactos_sha256": artefactos_sha256(artefactos, llamadas),
    }
    if entradas_de_dominio is not None:
        e["entradas_de_dominio"] = {d: firma(x) for d, x in entradas_de_dominio.items()}
    return e


def leer(ruta: Path) -> dict[str, Any]:
    if not ruta.exists():
        return {}
    return json.loads(ruta.read_text()).get("dominios", {})


def escribir(ruta: Path, entradas: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Reescribe la caché con las entradas de una corrida: no conserva las de corridas anteriores. JSON
    determinista, en el orden de los dominios."""
    contenido = {
        "descripcion": DESCRIPCION,
        "dominios": {d: entradas[d] for d in ORDEN if d in entradas},
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False) + "\n")
    return contenido


def herramientas_del(dominio: str) -> tuple[str, ...]:
    """Las herramientas que puede citar un dominio; el orquestador cita las de los tres."""
    if dominio == ORQUESTADOR:
        return tuple(h for hs in DOMINIOS.values() for h in hs)
    return DOMINIOS[dominio]


def vigente(
    artefactos: Path,
    dominio: str,
    e: Mapping[str, Any] | None,
    firmas: Mapping[str, str] | None = None,
) -> tuple[ResultadoDominio | None, str | None, list[tuple[Llamada, dict[str, Any]]]]:
    """La entrada de caché de un dominio si vale, o el motivo por el que no; con las respuestas actuales de
    sus llamadas, que el orquestador reúne. Para el orquestador, `firmas` son las de las entradas de dominio
    vigentes."""
    if e is None:
        return None, "no hay entrada en la caché", []
    if e.get("prompt_version") != PROMPT_VERSION:
        version = e.get("prompt_version")
        return None, f"la caché es de la versión de prompts {version}, no de {PROMPT_VERSION}", []
    if e.get("prompts_sha256") != hash_prompts():
        return None, "el hash de los prompts de la caché no coincide con el de los prompts actuales", []
    if dominio == ORQUESTADOR and (firmas is None or e.get("entradas_de_dominio") != dict(firmas)):
        return None, "el resumen es de otras entradas de dominio", []
    llamadas = [Llamada.model_validate(ll) for ll in e.get("llamadas", [])]
    ajenas = sorted({ll.herramienta for ll in llamadas} - set(herramientas_del(dominio)))
    if ajenas:
        return None, f"la entrada trae llamadas fuera del dominio: {', '.join(ajenas)}", []
    respuestas = responder(artefactos, llamadas)
    if huella(respuestas) != e.get("huella"):
        return None, "la huella de los datos cambió", respuestas
    insights = [Insight.model_validate(i) for i in e["insights"]]
    permitidas = validador.Permitidas(r for _, r in respuestas)
    v = validador.validar(insights, permitidas, herramientas_del(dominio))
    if not v.valida:
        return None, "el validador rechaza cifras de la caché contra las herramientas actuales", respuestas
    resultado = ResultadoDominio(
        origen="cache", insights=insights, validacion=v, modelo=e.get("modelo"), llamadas=llamadas
    )
    return resultado, None, respuestas
