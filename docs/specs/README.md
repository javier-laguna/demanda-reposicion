# Plan de ejecución y estado

Cada etapa tiene una spec (`E<n>.md`) con objetivo, entregables, criterios de aceptación y, al cerrar, su
resultado. Las etapas están acotadas para que cada una quepa en una ventana de contexto de la sesión de desarrollo.

| Etapa | Qué | Estado |
|---|---|---|
| [E0](E0.md) | Base del repo y gobierno | cerrada |
| [E1](E1.md) | Datos: contratos y calidad | cerrada |
| [E2](E2.md) | EDA reproducible (marimo) | cerrada |
| [E3](E3.md) | Features, backtest y baselines | cerrada |
| [E4](E4.md) | Modelos (LightGBM, Chronos-2) y validación robusta | cerrada |
| [E5](E5.md) | Monitor de drift y reglas de decisión | cerrada |
| [E6](E6.md) | Experimento de promociones | cerrada |
| [E7](E7.md) | Servidor MCP, agentes y validador | cerrada |
| [E8](E8.md) | Tablero | pendiente |
| [E9](E9.md) | Empaquetado y documentación | pendiente |
| E10 | Ensayo en limpio y publicación | pendiente |
| E11 | `PROCESS.md` y contenido de la entrega (lo decide el usuario) | cerrada |

Las specs de E2 en adelante se escriben al empezar cada etapa, a partir del plan.
El mínimo entregable es E0–E6, E9 y E11 (la prueba pide `PROCESS.md`). Si hay atraso, se recorta en este orden:
Chronos-2 en todos los cortes → heterogeneidad causal → páginas secundarias del tablero → exponer el MCP a clientes
externos.

## Acuerdos para etapas siguientes

Se incorporan a la spec de su etapa cuando se escriba.

- **E7:** el servidor MCP expone también `get_eda_summary`, que lee `artifacts/eda/resumen.json` para el agente
  de datos.
- **E9:** el README aclara que `.claude/` es opcional. Son los subagentes de desarrollo, solo funcionan en Claude
  Code y quedan como evidencia del proceso; reproducir la solución solo requiere uv/make o Docker.
- **E11:** el usuario decide qué lleva `PROCESS.md` y qué archivos entran a la entrega. Hasta entonces, la
  bitácora de `PROCESS.md` se sigue llenando al cerrar cada etapa, como borrador del que el usuario elige.

## Protocolo de cierre de etapa

1. `make validate` y `pre-commit run --all-files` en verde. Si fallan, no hay commit.
2. En las etapas con código sustancial (E3–E8), el subagente `reviewer` revisa el diff.
3. Se documenta la sección *Resultado* de la spec, esta tabla y la bitácora de `PROCESS.md`.
4. Se propone el commit (diff + mensaje de una línea) y se hace solo con aprobación. Lo mismo para el push.
5. Se compacta el contexto de la sesión. Al reanudar se leen `CLAUDE.md`, este archivo y la spec siguiente.
