"""El modelo de Pydantic AI desde la config, sus ajustes y la tabla de precios para estimar el gasto.

`LLM_MODEL` va en notación `proveedor:modelo`. Hay tres proveedores:
- `anthropic` (por omisión, `claude-opus-5-5`), con `ANTHROPIC_API_KEY`;
- `openrouter:<modelo>`, con `OPENROUTER_API_KEY`;
- `openai:<modelo>`, con `OPENAI_API_KEY` y, si se define, `LLM_BASE_URL`: cualquier gateway compatible con la
  API de OpenAI (corporativo, vLLM).

Las keys salen de `config.py` como `SecretStr` y se pasan al proveedor de forma explícita; nunca se imprimen.

**Salida estructurada con Claude Opus 5.5** (leído en Pydantic AI 2.61, `profiles/anthropic.py`,
`models/__init__.py` y `models/anthropic.py`):
- su perfil marca `supports_forced_tool_choice=False` (el `tool_choice` forzado da 400),
  `thinking_always_enabled=True` (pensamiento adaptativo, que no se puede apagar) y
  `supports_json_schema_output=True`;
- con el modo `auto`, `AnthropicModel._default_structured_output_mode` ya resolvería `native`: el
  pensamiento es adaptativo y no `enabled`, así que decide `Model._default_structured_output_mode`, que
  cambia `tool` por `native` porque forzar la herramienta de salida apagaría el pensamiento;
- aquí no se deja a esa inferencia: `tipo_salida` pide `NativeOutput` de forma explícita a todo modelo que
  rechaza el `tool_choice` forzado y soporta esquemas JSON (`output_config.format` en Anthropic, con
  `strict`), `PromptedOutput` si no los soporta, y deja `auto` a los demás.

**Ajustes de Anthropic:** esfuerzo `medium` explícito (`output_config.effort`), `max_tokens` fijo y el
respaldo del servidor ante rechazos (`fallbacks: "default"` en el cuerpo, beta
`server-side-fallback-2026-07-01`). Nunca se mandan `temperature`, `top_p` ni `top_k`: dan 400.

Limitación de Pydantic AI 2.61: no reconoce el bloque `fallback` que trae la respuesta cuando el respaldo
rescata una solicitud (`_process_response` acaba en un `assert`). Si eso pasa, la corrida de ese dominio
falla y cae a la plantilla, igual que con un rechazo sin respaldo; los tokens de esa solicitud no se ven en
`usage` y el gasto de esa corrida queda subestimado.
"""

from typing import Any, Literal

from pydantic import SecretStr
from pydantic_ai import NativeOutput, PromptedOutput
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from demanda.config import Config

PROVEEDORES = ("anthropic", "openrouter", "openai")
ESFUERZO = "medium"
MAX_TOKENS = 16_000
BETA_RESPALDO = "server-side-fallback-2026-07-01"

# USD por millón de tokens de entrada y de salida (precios de la API de Anthropic, 2026-10). El freno de
# gasto usa esta tabla; un modelo que no está usa el precio más alto.
PRECIOS: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.1, 0.5),
}
PRECIO_MAXIMO = max(PRECIOS.values(), key=lambda p: p[0] + p[1])


def partes(llm_model: str) -> tuple[str, str]:
    """`proveedor:modelo` → (proveedor, modelo). Sin proveedor se supone Anthropic."""
    proveedor, _, nombre = llm_model.partition(":")
    return (proveedor, nombre) if nombre else ("anthropic", proveedor)


def precio(llm_model: str) -> tuple[float, float]:
    """El precio por millón de tokens (entrada, salida) del modelo, sin el prefijo del proveedor ni el de la
    ruta de OpenRouter (`anthropic/claude-opus-5.5`)."""
    nombre = partes(llm_model)[1].rsplit("/", 1)[-1].replace(".", "-")
    return PRECIOS.get(nombre, PRECIO_MAXIMO)


def _key(cfg: Config) -> SecretStr | None:
    proveedor = partes(cfg.llm_model)[0]
    return {
        "anthropic": cfg.anthropic_api_key,
        "openrouter": cfg.openrouter_api_key,
        "openai": cfg.openai_api_key,
    }.get(proveedor)


def hay_key(cfg: Config) -> bool:
    """Si la config trae una key no vacía para el proveedor de `LLM_MODEL`."""
    key = _key(cfg)
    return key is not None and bool(key.get_secret_value().strip())


def crear_modelo(cfg: Config) -> Model:
    """El modelo de Pydantic AI de `LLM_MODEL`, con su key. No hace solicitudes."""
    proveedor, nombre = partes(cfg.llm_model)
    if proveedor not in PROVEEDORES:
        raise ValueError(f"proveedor no soportado en LLM_MODEL: {proveedor}; usa {', '.join(PROVEEDORES)}")
    if not hay_key(cfg):
        raise ValueError(f"falta la key de {proveedor} para LLM_MODEL={cfg.llm_model}")
    key = _key(cfg)
    assert key is not None
    secreto = key.get_secret_value()
    if proveedor == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(nombre, provider=AnthropicProvider(api_key=secreto))
    if proveedor == "openrouter":
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        return OpenRouterModel(nombre, provider=OpenRouterProvider(api_key=secreto))
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(nombre, provider=OpenAIProvider(base_url=cfg.llm_base_url, api_key=secreto))


def ajustes(llm_model: str) -> ModelSettings:
    """Los ajustes de las solicitudes: los de Anthropic solo van a Anthropic, porque el cuerpo extra del
    respaldo rompería a otros proveedores."""
    if partes(llm_model)[0] == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModelSettings

        return AnthropicModelSettings(
            max_tokens=MAX_TOKENS,
            anthropic_effort=ESFUERZO,
            anthropic_betas=[BETA_RESPALDO],
            extra_body={"fallbacks": "default"},
        )
    return ModelSettings(max_tokens=MAX_TOKENS)


def modo_salida(modelo: Model) -> Literal["native", "prompted", "auto"]:
    """El modo de salida estructurada para el modelo, según su perfil (ver el docstring del módulo)."""
    perfil = modelo.profile
    if perfil.get("supports_forced_tool_choice", True):
        return "auto"
    return "native" if perfil.get("supports_json_schema_output", False) else "prompted"


def tipo_salida(modelo: Model, tipo: type) -> Any:
    """El `output_type` del agente: el tipo envuelto en el modo que corresponde al modelo."""
    modo = modo_salida(modelo)
    if modo == "native":
        return NativeOutput(tipo)
    if modo == "prompted":
        return PromptedOutput(tipo)
    return tipo
