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

**Plan y estado:** `docs/specs/README.md` tiene la tabla de etapas (E0–E10) y el protocolo de cierre. Cada etapa
tiene su spec en `docs/specs/E<n>.md`. **Al empezar o reanudar, leer esos dos archivos antes de tocar código.**

## Comandos

```bash
make setup                          # uv sync --locked + hooks de pre-commit
make data DATA_SRC="<zip o carpeta>" # copia los 3 CSV a data/raw/ y verifica data/manifest.sha256
make data                           # solo verifica los hashes de data/raw/
make calidad                        # contratos + data/interim/panel.parquet + artifacts/calidad/reporte.json
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
  dice el diccionario. El WAPE de ~18.6 % (~36 % en Buen Fin y Navidad) que se midió en la exploración usaba el
  valor de T, así que es optimista; E3 lo recalcula con el de T−1.
- **Conteos inconsistentes en ~20 % de las filas**, uniformes por tienda, categoría y mes:
  `cash_transactions + card_transactions < total_transactions`, y en esas filas
  `avg_ticket = amount_total / (cash + card)`, que sobreestima el ticket real en ~22 %. Los montos sí cuadran:
  `amount_cash + amount_card == amount_total`.
- **Huecos de la malla:** solo 42 filas: `STR_045` (2023-02-01/02), `STR_064` (2023-02-22/23) y `STR_055`
  (2024-02-18 a 20; cae en la ventana de test).
- **Nulos dispersos:** `cash_transactions` y `amount_cash` ~6 % (siempre juntos), `units_sold` ~3 %, `avg_ticket`
  ~2.4 %. Todos los nulos de `avg_ticket` caen en filas sin efectivo, porque se calcula con `cash + card`.
- **Promociones asignadas al azar** (20 % uniforme por formato, día, mes y nivel previo; sin autocorrelación),
  con lift pequeño: +2.7 % en Bebidas y ~0 en Cuidado Personal y Abarrotes.
- **Calendario multiplicativo e idéntico por categoría:** Buen Fin ×2.6, quincena ×1.4, fin de semana ×1.5,
  Navidad ×1.16 y Semana Santa ×0.64. 2024-01-01 cae por debajo del 30 % de la mediana en todas las tiendas.
- La proporción de pago en efectivo es 41.6 % en todos los segmentos: no hay señal ahí.
- `calendar.week_of_year` es ISO: 2023-01-01 aparece como semana 52.

### Implicaciones para validar

- Buen Fin (2023-11-17 a 20) y Semana Santa (2023-04-06/07) **ocurren una sola vez**, así que su efecto no se puede
  validar fuera de muestra. La temporada navideña aparece parcialmente dos veces.
