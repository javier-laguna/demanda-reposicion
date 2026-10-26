# Plan de ejecución y estado

Cada etapa tiene una spec (`E<n>.md`) con objetivo, entregables, criterios de aceptación y, al cerrar, su
resultado. Las etapas están acotadas para que cada una quepa en una ventana de contexto de la sesión de desarrollo.

| Etapa | Qué | Estado |
|---|---|---|
| [E0](E0.md) | Base del repo y gobierno | cerrada |
| [E1](E1.md) | Datos: contratos y calidad | pendiente |
| E2 | EDA reproducible (marimo) | pendiente |
| E3 | Features, backtest y baselines | pendiente |
| E4 | Modelos (LightGBM, Chronos-2) y validación robusta | pendiente |
| E5 | Monitor de drift y reglas de decisión | pendiente |
| E6 | Experimento de promociones | pendiente |
| E7 | Servidor MCP, agentes y validador | pendiente |
| E8 | Tablero | pendiente |
| E9 | Empaquetado y documentación | pendiente |
| E10 | Ensayo en limpio y publicación | pendiente |

Las specs de E2 en adelante se escriben al empezar cada etapa, a partir del plan.
El mínimo entregable es E0–E6 más E9. Si hay atraso, se recorta en este orden: Chronos-2 en todos los cortes →
heterogeneidad causal → páginas secundarias del tablero → exponer el MCP a clientes externos.

## Protocolo de cierre de etapa

1. `make validate` y `pre-commit run --all-files` en verde. Si fallan, no hay commit.
2. En las etapas con código sustancial (E3–E8), el subagente `reviewer` revisa el diff.
3. Se documenta la sección *Resultado* de la spec, esta tabla y la bitácora de `PROCESS.md`.
4. Se propone el commit (diff + mensaje de una línea) y se hace solo con aprobación. Lo mismo para el push.
5. Se compacta el contexto de la sesión. Al reanudar se leen `CLAUDE.md`, este archivo y la spec siguiente.
