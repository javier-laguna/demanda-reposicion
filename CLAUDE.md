# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es

Solución a una prueba técnica de Data Science (retail con tiendas en varias regiones de México). El candidato define
el problema; el entregable es este repo, que debe ser reproducible siguiendo su documentación, más un `PROCESS.md`
que documente el proceso y el uso de IA.

**Solución elegida:**
- un pipeline determinista que pronostica `units_sold` por tienda × categoría a 7 días y lo compara contra el sistema
  de reposición actual;
- un experimento natural con las promociones;
- un monitor de drift y reglas de decisión;
- una capa de agentes (Pydantic AI, agnóstica de proveedor) que interpreta los resultados vía un servidor MCP de
  solo lectura;
- un tablero en Streamlit.

El diagrama está en `.archify/` (el HTML de la carpeta más reciente).

**Plan y estado:** `docs/specs/README.md` tiene la tabla de etapas (E0–E11) y el protocolo de cierre. Cada etapa
tiene su spec en `docs/specs/E<n>.md`. **Al empezar o reanudar, leer esos dos archivos antes de tocar código.**

## Comandos

```bash
make setup                          # uv sync --locked + hooks de pre-commit
make data DATA_SRC="<zip o carpeta>" # copia los 3 CSV a data/raw/ y verifica data/manifest.sha256
make data                           # solo verifica los hashes de data/raw/
make calidad                        # contratos + data/interim/panel.parquet + artifacts/calidad/reporte.json
make eda                            # notebook marimo → reports/eda.html + artifacts/eda/resumen.json
make backtest                       # 17 cortes × 6 modelos → artifacts/backtest/ + MLflow (~19 min; LightGBM ~11)
uv run demanda backtest --modelos sistema_actual,lightgbm   # subconjunto, para desarrollar
make seleccion                      # Diebold-Mariano + regla de campeón → artifacts/modelos/seleccion.json (segundos)
make ajuste                         # rejilla de LightGBM en orígenes previos al backtest (~4 min; no hace falta para correr)
make calibracion                    # campeón y retador en 8 orígenes previos → artifacts/backtest/calibracion.parquet (~4 min)
make decision                       # newsvendor: 3 políticas, impacto y sensibilidad → artifacts/decision/impacto.json (segundos)
make drift                          # monitor semana a semana y drift inyectado → artifacts/drift/monitor.json (~15 s)
make causal                         # experimento de promociones → artifacts/causal/promociones.json (~20 s)
make insights                       # agentes: vivo con key, si no caché o plantillas → artifacts/insights/insights.json
make insights MODO=vivo             # fuerza el modo vivo (~USD 0.4, ~2 min); actualiza insights/cache.json
make mcp                            # servidor MCP de solo lectura por stdio (demanda mcp --http: 127.0.0.1:8765)
uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db   # ver los runs
make validate                       # ruff (lint + formato) + pyrefly + pytest; requisito antes de cada commit
uv run pytest tests/test_fuente.py::test_prepara_desde_zip   # un test
uv run demanda --help               # CLI del pipeline; cada etapa agrega su comando
```

Python 3.13 fijado (`.python-version`; `requires-python` < 3.14). Las dependencias se agregan con `uv add` en la etapa
que las usa.

## Reglas del repo

- **Los datos nunca se versionan.** `data/` está en `.gitignore`, salvo `data/manifest.sha256`, y un hook de
  pre-commit bloquea `*.csv`, `*.parquet` y `*.zip`. Lo mismo para `artifacts/` y `reports/`. Lo único derivado de
  los datos que se versiona es `insights/cache.json`: conclusiones agregadas por segmento, nunca filas ni series.
- **Secretos:** solo en `.env` (gitignored; plantilla en `.env.example`) y leídos como `SecretStr` en
  `src/demanda/config.py`. gitleaks corre en pre-commit y en CI. CI corre sin key.
- **Tests sin datos reales:** usan fixtures sintéticos generados en el test.
- **Agentes:** interpretan y proponen. No calculan ni eligen el modelo (eso son reglas deterministas), solo leen vía
  MCP, y cada cifra pasa por el validador.
- **Contexto:** el trabajo pesado o exploratorio va a los subagentes de `.claude/agents/`. Las salidas largas se
  escriben a archivo.

## Datos

`data/raw/`: `transactions.csv` (~204k filas), `stores.csv`, `calendar.csv`.

- 80 tiendas × 6 categorías × 425 días (2023-01-01 a 2024-02-29) = 480 series diarias.
- Llaves: `transactions` ↔ `stores` por `store_id`; `transactions` ↔ `calendar` por `date`.
  Grano de `transactions`: `(date, store_id, category)`, sin duplicados.
- Contratos en `src/demanda/datos/contratos.py`: lo que debe cumplirse siempre (tipos, dominios, llaves, montos que
  cuadran). Los problemas conocidos no rompen el contrato: se miden en el reporte de calidad y se marcan en el panel.
- **El panel es la malla completa** (204,000 filas): las 42 filas que faltan aparecen con medidas nulas y
  `flag_fila_faltante`, así un `shift` por serie siempre equivale a días de calendario. Nada se imputa.

### Hallazgos de calidad verificados (el diccionario no los menciona o los describe mal)

- **`replenishment_signal` es fuga de información y además mira un día al futuro.** Es ≈ 0.53 × la media de
  `units_sold` de **t−5 a t+1**: el valor fechado en t ya contiene las ventas de t y de t+1. La regresión intraserie
  sobre los desfases −7..+7 da pesos de ~0.06 justo en −5..+1 y ~0 fuera. **Con origen T solo se pueden usar
  valores fechados en T−1 o antes.** Sus nulos están en **2023-01-01 y 2023-01-02**, no "en los últimos días" como
  dice el diccionario. Con el protocolo de la exploración (suma de T+1 a T+7, orígenes desde 2023-07-01), el WAPE
  del sistema actual es 18.6 % con la señal de T y **20.1 % con la de T−1**, contra 21.6 % de la MA7. En las
  semanas con Buen Fin llega a 44 %. Los resultados del backtest de E3 están en «Backtest y modelos».
- **Conteos inconsistentes en ~20 % de las filas, que son exactamente los días con promoción** (lo encontró E6):
  - en esas filas, `cash_transactions + card_transactions < total_transactions`. `total_transactions` vale en
    promedio 1.22 × (cash + card), de 1.08 a 1.35; en los demás días vale exactamente lo mismo;
  - la promoción no mueve ni las transacciones pagadas, ni las unidades, ni el monto, así que el exceso son registros
    sin pago;
  - por eso `avg_ticket = amount_total / (cash + card)` es el ticket por transacción pagada, y
    `amount_total / total_transactions` lo subestima ~18 % en los días con promoción;
  - los montos sí cuadran: `amount_cash + amount_card == amount_total`.
- **Huecos de la malla:** solo 42 filas: `STR_045` (2023-02-01/02), `STR_064` (2023-02-22/23) y `STR_055`
  (2024-02-18 a 20; cae en la ventana de test).
- **Nulos dispersos:** `cash_transactions` y `amount_cash` ~6 % (siempre juntos), `units_sold` ~3 %, `avg_ticket`
  ~2.4 %. Todos los nulos de `avg_ticket` caen en filas sin efectivo, porque se calcula con `cash + card`.
- **Promociones asignadas al azar** (20 % uniforme por formato, día, mes y nivel previo; sin autocorrelación),
  con lift ingenuo pequeño: +1.8 % en Bebidas (IC95 de 0.05 a 3.6 %) y entre −0.6 % y +0.9 % en el resto. E6
  confirma la asignación con pruebas de balance y estima el efecto (ver «Promociones»).
- **Calendario multiplicativo y casi idéntico por categoría** (rango ≤ 4 %, salvo Semana Santa con 9.5 % en dos
  días). Contra la misma serie, mes y día de la semana en días limpios, los efectos son:
  - Buen Fin ×2.65, quincena ×1.38, fin de semana ×1.53 y Semana Santa ×0.66.
  - **Navidad depende del método:** ×1.09 sobre el nivel del mes, que ya es alto, y ~×1.3 sin controlar el mes.
    Con un solo diciembre no se pueden separar.
  - El perfil semanal no es binario: lunes 0.75, viernes 1.10, sábado 1.40.
  - Cada festivo tiene su propio efecto, de ×0.43 (Año Nuevo) a ×2.8 (Nochebuena).
  - 2024-01-01 es el único día atípico. El total cae al 29 % de la mediana; el 60 % de las tiendas queda por debajo
    del 30 %.
- **Nivel con estacionalidad anual y tendencia:** aun en días limpios, diciembre vende 1.9× lo de enero de 2023 y el
  crecimiento interanual es +9.6 %. Las series difieren 38× en nivel, y el tamaño de la tienda correlaciona 0.83 con
  su demanda.
- La proporción de pago en efectivo es 41.6 % en todos los segmentos: no hay señal ahí.
- `calendar.week_of_year` es ISO: 2023-01-01 aparece como semana 52.

### Implicaciones para validar

- Buen Fin (2023-11-17 a 20) y Semana Santa (2023-04-06/07) **ocurren una sola vez**, así que su efecto no se puede
  validar fuera de muestra. La temporada navideña aparece parcialmente dos veces, y el nivel alto de noviembre y
  diciembre solo una.
- Por los nulos de `units_sold`, solo el 81 % de los orígenes tiene los 7 días siguientes completos. Si se evalúa la
  suma semanal, hay que reportar la cobertura; los errores diarios no pierden esos orígenes.

## Backtest y modelos

- **Protocolo** (`src/demanda/backtest/protocolo.py`):
  - 17 orígenes semanales, los jueves del 2023-11-02 al 2024-02-22, con h = 1..7;
  - `separar` le da a cada modelo solo dos tablas: `historia`, con fecha ≤ T y la señal de T en nulo, y `futuro`,
    con las columnas `CONOCIDAS` (calendario, tienda y `has_promotion`).
- **Modelos:** un modelo nuevo se registra en `src/demanda/modelos/` con `registrar(nombre, fábrica)`. Con eso entra
  al backtest y a la prueba de fugas de `tests/test_modelos.py`, que altera el futuro y exige el mismo pronóstico.
  Las features están en `src/demanda/features/matriz.py`, y todo lo que entra en `FEATURES` queda cubierto por la
  prueba de fugas de la matriz.
- **LightGBM** (`modelos/gbm.py`):
  - es global y directo, con un modelo por cuantil;
  - el objetivo es y / nivel, con peso nivel, donde nivel es la mediana de 28 días;
  - se reentrena en cada corte;
  - los hiperparámetros salen de `make ajuste` sobre orígenes previos al backtest y viven como constante en
    `AJUSTADOS`. Nunca se ajusta con los cortes del backtest.
- **Chronos-2** (`modelos/chronos2.py`):
  - corre en un proceso hijo (`proceso_chronos.py`), porque en macOS el libomp de LightGBM y el de torch tumban el
    proceso si conviven;
  - los pesos (`amazon/chronos-2`, revisión fijada) se bajan de Hugging Face la primera vez;
  - en los tests se usa un doble y CI no descarga nada.
- **Torch** viene del índice de CPU de PyTorch en Linux (`[tool.uv.sources]`). En macOS viene de PyPI.
- **Métrica principal:** el WAPE de la suma semanal por serie y corte, solo con los pares que tienen los 7 días
  observados (80.5 %). Lo diario es diagnóstico. Hay un nivel de evento `post_buen_fin` para la semana siguiente
  al Buen Fin, así `resto` son las semanas normales.
- **La vara de E3:**
  - el sistema actual tiene un WAPE semanal de 27.3 % en los 17 cortes y de **11.0 % en semanas normales**, donde es
    el mejor baseline;
  - falla en el Buen Fin (sesgo de −50 %), en la semana siguiente (+101 %, porque su señal promedia el pico) y en
    fin de año;
  - AutoETS da 24.7 % en el global, pero 13.4 % en semanas normales.
- **Resultado de E4** (`docs/specs/E4.md`). La regla de campeón (`seleccion/campeon.py`) se fijó antes de ver
  resultados, y mantiene a `sistema_actual`.
  - LightGBM da 20.3 % en el global y 11.3 % en semanas normales, pero la Diebold-Mariano por corte da p = 0.17 con
    17 cortes.
  - Chronos-2 sin entrenar da 23.3 %, con un sesgo de −14 %.
  - Los intervalos del 80 % de los dos cubren solo el 67–69 %.

## Decisión y drift

- **Calibración** (`backtest/calibracion.py`): el campeón y el retador de `seleccion.json` en los 8 orígenes previos
  al backtest (los del ajuste). El retador es el de menor WAPE semanal global que no es el campeón.
  - `calibracion.json` guarda los hashes de su fuente;
  - `decision` y `drift` fallan si la calibración es de otra corrida.
- **Newsvendor** (`decision/`):
  - el nivel objetivo semanal es S = Σŷ × q, donde q es el cuantil ponderado (peso Σŷ) de Σy / Σŷ en la fracción
    crítica de la categoría;
  - la ventana es expansiva: las 8 semanas de calibración más los cortes previos del backtest **con evento `resto`**.
    Nunca mira el corte que decide;
  - los márgenes y costos de mantener son supuestos ilustrativos en `decision/supuestos.py`; el precio sale de los
    datos.
- **Monitor** (`drift/`):
  - es una función pura de (panel, pronósticos, semana); la inyección de drift solo cambia el panel;
  - la referencia son los 56 días que terminan en T, sin días especiales;
  - tiene tres capas: calidad, entradas y desempeño. Las alertas de nivel, ticket y desempeño son «esperadas» si la
    semana tiene un día especial o un evento del backtest.
- **Resultado de E5** (`docs/specs/E5.md`):
  - el newsvendor sobre el pronóstico del sistema actual cuesta 40 % menos que reponer al pronóstico puntual
    (p = 0.019);
  - LightGBM baja otro 9 %, sin significancia (p = 0.20), y empata en semanas normales;
  - el costo está en los eventos (Buen Fin: 114 de 213 M);
  - el monitor detecta los tres escenarios inyectados y la falla real de STR_055, pero su chequeo de nivel confunde
    la estacionalidad anual con drift: en noviembre y enero alerta en todos los segmentos;
  - el chequeo de ticket usa `amount_total / total_transactions`, así que también se mueve con la tasa de promoción
    (ver el hallazgo de conteos).

## Promociones

- **Método** (`causal/`):
  - log-OLS con efectos fijos de serie y de fecha, absorbidos por proyecciones alternadas;
  - errores agrupados por tienda, con t de 79 gl;
  - inferencia por aleatorización y calibración A/A;
  - las Wald conjuntas llevan la corrección de Hotelling.
- **Balance:**
  - la regla pre-registrada falla solo por `flag_conteo_inconsistente`, que es un resultado de la promoción;
  - las pruebas de la asignación pasan todas (`asignacion_se_sostiene`, que excluye `balance.MEDICION`; la separación
    se decidió después de ver datos).
- **Resultado de E6** (`docs/specs/E6.md`; el diseño del próximo A/B, en `docs/causal_promos.md`):
  - la promoción no mueve las unidades: +0.05 %, IC95 de −0.25 a +0.34 %;
  - la excepción es Bebidas, con +1.45 %;
  - no hay adelanto de compras ni efecto sobre otras categorías;
  - con márgenes supuestos, ningún descuento de 5 a 20 % se paga.

## Capa de agentes

- **Servidor** (`mcp_server/`):
  - FastMCP con 7 herramientas de solo lectura; las funciones puras están en `herramientas.py`;
  - los agentes lo usan en proceso, con `MCPToolset` sobre la instancia.
  - `get_forecast` no la usa ningún agente de `make insights`, porque devuelve series de una tienda.
- **Agentes** (`agentes/`):
  - `datos`, `pronostico` y `causal`, más un orquestador sin herramientas que recibe sus insights ya validados;
  - el modelo es `LLM_MODEL`, por omisión `anthropic:claude-opus-5-5`, con `NativeOutput`, esfuerzo `medium` y sin
    parámetros de muestreo (Opus 5.5 rechaza `temperature` y el `tool_choice` forzado).
- **Validador:** una cifra solo pasa si aparece, con redondeo, en las respuestas de herramientas que el agente leyó
  en esa corrida. Si falla, hay un reintento; si vuelve a fallar, se usa la plantilla.
- **Caché:**
  - `insights/cache.json`, el único derivado de los datos que se versiona;
  - vale si coinciden la huella (respuestas redondeadas a 3 cifras significativas) y `PROMPT_VERSION`, y si sus
    cifras se revalidan;
  - un cambio de prompt exige subir `PROMPT_VERSION` y correr `make insights MODO=vivo`.
- **Presupuesto:**
  - un tope de USD 1 por corrida y uno acumulado de USD 20, en `artifacts/insights/gasto.json`;
  - el límite duro va en la consola del proveedor.
- **Tests:** nunca salen a la red (`ALLOW_MODEL_REQUESTS = False`), vacían las keys del entorno y apuntan
  `DEMANDA_ENV_FILE` a un archivo inexistente, así que nunca leen el `.env` real.
