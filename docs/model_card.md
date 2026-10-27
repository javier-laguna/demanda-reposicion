# Model card · Pronóstico semanal de demanda para reposición

Las cifras salen de los artefactos del pipeline (`artifacts/`) y de las specs de cada etapa (`docs/specs/`).

## Qué resuelve

Pronostica las unidades vendidas por tienda × categoría para los 7 días siguientes a un origen semanal (el jueves).
Esa cifra alimenta la decisión de reposición de la semana. La solución tiene cuatro piezas:
- **Pronóstico:** compara el sistema de reposición actual contra cuatro baselines y dos modelos nuevos.
- **Decisión:** convierte el pronóstico en un nivel objetivo de inventario (newsvendor).
- **Monitor de drift:** vigila semana a semana si los datos, las entradas o el desempeño cambiaron.
- **Interpretación:** una capa de agentes redacta conclusiones de negocio, con cada cifra validada contra los
  artefactos.

**Uso previsto:** la planeación semanal de inventario por tienda × categoría.

**Fuera de alcance:**
- el pronóstico a nivel SKU;
- horizontes de más de una semana;
- la planeación de eventos únicos como el Buen Fin, que necesita revisión humana (ver *Limitaciones*).

## Datos

- 80 tiendas × 6 categorías × 425 días (2023-01-01 a 2024-02-29): 480 series diarias y 204,000 filas en la malla
  completa.
- Las tablas son transacciones, tiendas y calendario. Sus contratos están en `src/demanda/datos/contratos.py`.
- **Hallazgos que condicionan el modelo:**
  - `replenishment_signal` es fuga de información: el valor fechado en t ya contiene las ventas de t y de t+1. Con
    origen T solo se usa la señal fechada en T−1 o antes;
  - el 3 % de `units_sold` es nulo y no se imputa, así que el 80.5 % de los pares serie × corte tiene la semana
    completa;
  - los «conteos inconsistentes» del ~20 % de las filas son los días con promoción: `total_transactions` cuenta
    registros sin pago.
- Los datos parecen sintéticos: promociones al azar por fila, efectos de calendario casi idénticos entre categorías y
  una proporción de efectivo de 41.6 % en todos los segmentos.

## Modelos evaluados

| Modelo | Qué es |
|---|---|
| `sistema_actual` | el sistema de reposición de hoy, reconstruido desde `replenishment_signal` con la señal de T−1 |
| `naive_estacional`, `ma7`, `autoets` | baselines |
| `lightgbm` | modelo global y directo, por cuantiles (P10, P50, P90), con objetivo y / nivel; hiperparámetros ajustados en orígenes previos al backtest |
| `chronos2` | modelo fundacional de series de tiempo, sin entrenar |

## Evaluación

- **Protocolo:** 17 orígenes semanales, los jueves del 2023-11-02 al 2024-02-22, con h = 1..7. Cada modelo recibe
  solo la historia hasta T y las variables conocidas a futuro (calendario, tienda y promoción planeada).
- **Métrica principal:** el WAPE de la suma semanal por serie y corte.

### Global y por evento (WAPE semanal, %)

| Segmento | sistema_actual | lightgbm | chronos2 |
|---|---|---|---|
| Global (17 cortes) | 27.3 | **20.3** | 23.3 |
| Semanas normales (11) | **11.0** | 11.3 | 14.7 |
| Buen Fin | 49.9 | 48.9 | 54.9 |
| Semana posterior al Buen Fin | 100.9 | 20.8 | **12.6** |
| Navidad y Año Nuevo (4) | 33.5 | **24.9** | 28.0 |

### Por subgrupo (WAPE / sesgo semanal, %)

| Subgrupo | sistema_actual | lightgbm |
|---|---|---|
| Bodega | 27.7 / +2.1 | 20.0 / −3.7 |
| Express | 23.9 / +2.7 | 18.0 / −0.4 |
| Supercenter | 28.0 / +0.9 | 21.5 / −4.7 |
| Centro | 27.8 / +1.6 | 20.2 / −4.1 |
| Norte | 26.9 / +0.7 | 20.5 / −4.1 |
| Occidente | 26.6 / +1.7 | 20.2 / −3.4 |
| Oriente | 27.8 / +2.3 | 20.1 / −3.0 |
| Sur | 27.7 / +2.2 | 20.7 / −3.6 |
| NSE A/B | 25.4 / +0.9 | 19.1 / −4.3 |
| NSE B | 27.4 / +1.7 | 20.6 / −3.8 |
| NSE C+ | 27.6 / +1.4 | 20.8 / −3.4 |
| NSE C | 27.2 / +2.0 | 19.9 / −3.5 |
| Abarrotes | 27.0 / +1.1 | 20.6 / −4.7 |
| Bebidas | 28.2 / +2.2 | 20.6 / −3.2 |
| Cuidado_Personal | 26.5 / +1.2 | 19.7 / −3.7 |
| Electronica | 28.0 / +2.1 | 20.6 / −2.7 |
| Hogar | 27.4 / +2.9 | 19.9 / −2.8 |
| Ropa | 26.8 / +1.5 | 20.2 / −2.4 |

- **Ningún subgrupo queda atrás.** El WAPE varía menos de 5 pp entre formatos, regiones, NSE y categorías, en los
  dos modelos. Ningún sesgo por subgrupo pasa del umbral de 5 % de la regla de campeón.
- **Sesgo:** LightGBM subpronostica un poco en todos los subgrupos (de −0.4 a −4.7 %) y el sistema actual
  sobrepronostica un poco (de +0.7 a +2.9 %).
- **Intervalos:** el P10–P90 de LightGBM cubre el 69 % de los días contra el 80 % nominal (Chronos-2, el 67 %).
  - En semanas normales llega al 76 %, y entre subgrupos va de 67 a 71 %;
  - en los días del Buen Fin cae al 5 %.

  Los intervalos son demasiado angostos en los eventos. Por eso la decisión no los usa directamente: calibra su
  propio colchón (ver *Decisión*).

## Elección del modelo

- **La regla de campeón se fijó antes de ver resultados** (`seleccion/campeon.py`). Un retador reemplaza al sistema
  actual si cumple las cuatro condiciones:
  - gana en el global;
  - la Diebold-Mariano por corte es significativa (p < 0.05);
  - no pierde más de 0.5 pp en las semanas normales;
  - ningún subgrupo tiene un sesgo de más de 5 %.
- **Resultado: el campeón sigue siendo `sistema_actual`.** LightGBM baja el WAPE global de 27.3 a 20.3 %, pero la
  diferencia no es significativa con 17 cortes (p = 0.17), y su ventaja está en los eventos.
- **LightGBM queda como retador en vigilancia.** El monitor estima que harían falta unos 34 cortes para que la
  diferencia, si se mantiene, sea significativa.

## Decisión de reposición

El nivel objetivo semanal es S = Σŷ × q, donde q es el cuantil de servicio de la categoría (newsvendor). Se calibra
con los errores de semanas pasadas, sin mirar la semana que se decide.

| Política | Costo (M de pesos) | Quiebre | Sobreinventario | Fill rate |
|---|---|---|---|---|
| Pronóstico puntual del sistema actual | 352 | 12.8 % | 14.5 % | 87.2 % |
| Newsvendor sobre el sistema actual | 213 | 5.7 % | 34.1 % | 94.4 % |
| Newsvendor sobre LightGBM | 193 | 5.6 % | 25.8 % | 94.4 % |

- **La regla de decisión vale más que el modelo:**
  - el colchón baja el costo 40 % (Diebold-Mariano p = 0.019);
  - cambiar el pronóstico por LightGBM baja otro 9 %, sin significancia (p = 0.20), y empata en semanas normales.
- **Los márgenes y costos de mantener son supuestos ilustrativos** (`decision/supuestos.py`); el precio sale de los
  datos. El orden de las políticas se mantiene con fracciones críticas de 0.80 a 0.98.
- **El costo se concentra en los eventos:** el Buen Fin pesa 114 de los 213 M.

## Monitor de drift

- **Capas:** calidad (tiendas sin datos, tasas de nulos), entradas (plan de promoción, nivel de demanda, ticket) y
  desempeño (WAPE contra su banda, sesgo localizado).
- **Clasificación:** cada alerta es esperada (por calendario o evento) o inesperada, y lleva una recomendación.
- **Validación:**
  - detecta los tres escenarios de drift inyectado: falla de punto de venta, salto de nivel de +30 % en una región
    y cambio en la tasa de promoción;
  - también la falla real de captura de STR_055;
  - en las 35 semanas de calibración da 3 alertas inesperadas.
- **Limitación conocida:** el chequeo de nivel confunde la estacionalidad anual con drift. En noviembre y enero alerta
  en todos los segmentos. Con un año de historia no hay una referencia interanual.

## Promociones

La promoción se asignó al azar, lo que permitió un experimento natural (`docs/causal_promos.md`):
- **Unidades:** no las mueve (+0.05 %, IC95 de −0.25 a +0.34 %). La excepción es Bebidas (+1.45 %).
- **Rentabilidad:** con los márgenes supuestos, ningún descuento de 5 a 20 % se paga.
- **Efecto sobre el pronóstico:** la promoción entra como variable conocida a futuro, pero su efecto es tan pequeño
  que aporta poco.

## Capa de agentes

- Tres agentes de dominio y un orquestador (Pydantic AI, `anthropic:claude-opus-5-5` por defecto) leen los artefactos
  por un servidor MCP de solo lectura.
- No calculan ni deciden. Cada cifra que escriben pasa por un validador que exige que aparezca en lo que leyeron.
- Sin key, se usa la caché versionada o plantillas deterministas.
- **Límite:** el validador comprueba que la cifra exista, no que la frase la use bien.

## Limitaciones

- **Eventos únicos.** El Buen Fin y la Semana Santa ocurren una sola vez en los datos, así que su efecto no se valida
  fuera de muestra. Los dos modelos fallan cerca de 50 % en la semana del Buen Fin. Ese evento necesita un plan
  propio.
- **Un solo año.** No hay referencia interanual para el monitor ni para separar la estacionalidad anual de la
  tendencia (+9.6 % anual).
- **La señal de reposición tiene fuga.** El sistema actual se reconstruyó con la señal de T−1, que es lo honesto, pero
  no es exactamente el sistema en producción.
- **Ventas, no demanda.** En un quiebre real la venta queda censurada, y aquí se supone que no lo está.
- **Costos supuestos.** Los márgenes, costos de mantener y descuentos son ilustrativos; las conclusiones de orden se
  sostienen en la sensibilidad, los montos no.
- **Datos aparentemente sintéticos.** Algunas regularidades (promoción al azar por fila, calendario común) no se
  repetirían en datos reales.
- **Reproducibilidad entre plataformas.** Dos corridas en la misma máquina dan artefactos idénticos. Entre macOS y
  Linux, los pronósticos difieren en ~1e-6 relativo, sin cambiar ninguna conclusión.
