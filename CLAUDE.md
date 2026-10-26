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
make eda                            # notebook marimo → reports/eda.html + artifacts/eda/resumen.json
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
  semanas con Buen Fin llega a 44 %. El backtest formal es de E3.
- **Conteos inconsistentes en ~20 % de las filas**, uniformes por tienda, categoría y mes:
  `cash_transactions + card_transactions < total_transactions`, y en esas filas
  `avg_ticket = amount_total / (cash + card)`, que sobreestima el ticket real en ~22 %. Los montos sí cuadran:
  `amount_cash + amount_card == amount_total`.
- **Huecos de la malla:** solo 42 filas: `STR_045` (2023-02-01/02), `STR_064` (2023-02-22/23) y `STR_055`
  (2024-02-18 a 20; cae en la ventana de test).
- **Nulos dispersos:** `cash_transactions` y `amount_cash` ~6 % (siempre juntos), `units_sold` ~3 %, `avg_ticket`
  ~2.4 %. Todos los nulos de `avg_ticket` caen en filas sin efectivo, porque se calcula con `cash + card`.
- **Promociones asignadas al azar** (20 % uniforme por formato, día, mes y nivel previo; sin autocorrelación),
  con lift ingenuo pequeño: +1.8 % en Bebidas (IC95 de 0.05 a 3.6 %) y entre −0.6 % y +0.9 % en el resto. E6 lo
  estima formalmente.
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
