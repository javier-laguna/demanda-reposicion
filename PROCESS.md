# Proceso de trabajo

Este documento registra cómo se construyó la solución: las decisiones, las herramientas y, en particular, cómo se
usó la IA. Se escribió al cerrar cada etapa del plan (`docs/specs/README.md`).

## Herramientas

| Ámbito | Herramienta |
|---|---|
| Ambiente y calidad | uv (Python 3.13 fijado, `uv.lock`), ruff, pyrefly, pytest, pre-commit, gitleaks, Make |
| Datos | Polars, Pandera _(a partir de E1)_ |
| Modelado | LightGBM (sobre una matriz de features propia), statsforecast, Chronos-2, SciPy, statsmodels _(E3–E6)_ |
| Seguimiento | MLflow local con SQLite _(E3)_ |
| Capa de IA | Pydantic AI, con Anthropic por defecto y proveedor configurable; servidor MCP con FastMCP _(E7)_ |
| Entrega | Streamlit _(E8)_, Docker Compose, GitHub Actions |
| Asistente de desarrollo | Claude Code (Claude Opus 5.5) con subagentes, skills y servidores MCP de documentación |

## Cómo se usó la IA

La IA aparece en dos lugares distintos, y conviene no confundirlos:

1. **En el desarrollo.** Claude Code construyó el repo conmigo:
   - exploración de datos y lectura de documentación;
   - código y tests;
   - revisión de cada etapa con subagentes de `.claude/agents/`;
   - diagramas con la skill archify.

   Las decisiones de enfoque y alcance fueron mías: la IA propuso opciones y yo elegí o corregí. Los criterios de
   validación los escribió Claude Code en la spec de cada etapa, a partir del plan que aprobé. Nada del producto
   depende de Claude Code.
2. **En el producto.** Una capa de agentes (Pydantic AI) interpreta los resultados del pipeline y redacta
   conclusiones de negocio. Tiene tres límites: no calcula ni decide, solo lee vía un servidor MCP, y un validador
   determinista comprueba cada cifra que cita contra los artefactos. Funciona con cualquier proveedor y también sin
   key: en ese caso usa la caché versionada o conclusiones generadas con plantillas.

## Bitácora

### Antes de E0 · Exploración, investigación y plan

- **Perfilado de los datos con scripts desechables**, antes de proponer nada. De ahí salieron los hallazgos que
  definieron el enfoque, y que hoy están en `CLAUDE.md`:
  - la fuga de `replenishment_signal`;
  - el 20 % de conteos inconsistentes;
  - que los huecos se reducen a 42 filas;
  - la aleatorización de las promociones;
  - los efectos de calendario;
  - la falta de señal en el pago en efectivo.
- **Investigación de herramientas 2025–2026 con tres agentes en paralelo:**
  - servidores MCP del ecosistema de ciencia de datos;
  - modelos y librerías de forecasting, tabular y causal;
  - herramientas de trabajo.

  Las versiones de 19 paquetes clave se verificaron después contra PyPI. De ahí salió el stack de la tabla de
  arriba. El documento de la investigación fue material de trabajo y no forma parte de la entrega.
- **Definición del problema.** La IA propuso varios problemas de negocio y se eligió el pronóstico de demanda para
  reposición, por tres razones:
  - es el único con un sistema actual contra el cual medir valor;
  - `total_transactions` tiene conteos inconsistentes;
  - el efectivo no tiene señal.

  El experimento de promociones se sumó porque la asignación aleatoria permite medir causalidad.
- **Correcciones mías a la propuesta inicial:**
  - pedí que los agentes por etapa fueran parte del sistema replicable, no solo una herramienta de desarrollo;
  - exigí que la capa de IA fuera agnóstica de proveedor;
  - pedí un monitor de drift explícito.
- **Diagrama de arquitectura con archify.** La versión final, con los resultados, se ve en la página Arquitectura
  del tablero.

### E0 · Base del repo y gobierno

- **Qué se hizo:**
  - ambiente uv con Python 3.13 fijado;
  - Makefile como punto de entrada;
  - pre-commit con ruff, pyrefly, gitleaks y un bloqueo de archivos de datos;
  - configuración con `SecretStr`;
  - `demanda data`, que coloca los CSV y verifica sus hashes (los datos no se versionan);
  - esqueletos de Docker y CI;
  - specs por etapa.
- **Decisión:** los datos de la prueba no se suben al repo. Quien lo reproduzca los coloca con
  `make data DATA_SRC=...`, y el manifiesto SHA-256 garantiza que son los originales.

### E1 · Datos: contratos y calidad

- **Qué se hizo:**
  - contratos Pandera (backend Polars) para las 3 tablas y sus uniones;
  - reporte de calidad en JSON;
  - panel sobre la malla completa con banderas de calidad;
  - `make calidad`;
  - 29 tests sobre datos sintéticos.
- **Criterio de diseño:** el contrato solo exige lo que debe cumplirse siempre (tipos, dominios, llaves, montos que
  cuadran). Los problemas conocidos, como los conteos inconsistentes o los nulos, no lo rompen: se miden y se marcan.
  Si un archivo nuevo rompe el contrato, la etapa no escribe nada.
- **Qué aportó la IA:** al diseñar la prueba de fuga no se quedó con la cifra de la exploración (MA7 que incluye el
  día, correlación 0.98). Probó ventanas que terminan después del día y una regresión por desfase, y así descubrió
  que la señal incluye las ventas de t+1. Eso cerró el pendiente "falta verificar si usa información futura" y
  endureció la regla de uso de la señal: con origen T, solo valores fechados en T−1 o antes. También implica que el
  WAPE del sistema actual medido en la exploración era optimista.
- **Decisión:** el panel incluye las filas faltantes, con medidas nulas y una bandera, en vez de omitirlas. Así los
  lags por `shift` corresponden a días de calendario y el backtest sabe qué fechas no evaluar.

### E2 · EDA reproducible

- **Qué se hizo:**
  - notebook marimo con seis secciones, exportado sin interfaz con `make eda`;
  - `artifacts/eda/resumen.json` con las cifras clave y su definición, para los agentes de E7;
  - una prueba de humo que corre el notebook sobre datos sintéticos.
- **Cómo se usó la IA:**
  - El notebook lo escribió un subagente con las instrucciones de `.claude/agents/eda-analyst.md`. No se invocó como
    subagente registrado, porque Claude Code carga esos archivos al iniciar la sesión y se habían creado después; se
    usó un agente general con el mismo prompt.
  - Claude Code recalculó las cifras centrales con un script independiente antes de aceptarlas. Coinciden al centésimo.
- **Qué corrigió el EDA:**
  - El sistema actual sin fuga tiene un WAPE de 20.1 %, no 18.6 %, y le gana a una media móvil de 7 días por solo
    1.5 pp. Esa es la vara honesta para los modelos.
  - Tres cifras de la exploración no resistieron un método más cuidadoso, y `CLAUDE.md` se corrigió:
    - el efecto de Navidad;
    - el lift de Bebidas;
    - "2024-01-01 en todas las tiendas".
- **Decisión:** el efecto de Navidad queda reportado con sus dos lecturas (×1.09 sobre el mes, ~×1.3 sin controlarlo)
  en vez de elegir una. Con un solo diciembre los datos no permiten separarlas, y el modelo usa mes y nivel reciente
  como features en lugar de depender de ese multiplicador.

### E3 · Features, backtest y baselines

- **Qué se hizo:**
  - un protocolo con 17 cortes semanales que cubren Buen Fin, Navidad y enero–febrero;
  - cuatro baselines: el sistema actual sin fuga, naive estacional, MA7 y AutoETS;
  - métricas semanales y diarias por segmento;
  - la conciliación con E2;
  - MLflow local;
  - la matriz de features que usará E4.
- **Decisiones de diseño de la spec:**
  - **El futuro se separa por construcción.** Cada modelo recibe solo la historia hasta T y lo que se conoce de
    antemano, así no depende de que cada modelo "se porte bien". Encima de eso, una prueba altera el futuro y exige
    el mismo pronóstico. Se aplica a todo modelo registrado, incluidos los de E4.
  - **La métrica principal es la suma semanal.** Es la granularidad de la reposición y la única en la que el sistema
    actual, que es una media de 7 días, compite sin desventaja. Medido por día, su perfil plano lo castigaría
    (45.7 %), y la comparación sería injusta a favor de los modelos nuevos.
  - **17 cortes en lugar de los ~8 del plan,** porque es lo que cabe con frecuencia semanal en ese periodo, y más
    cortes dan más potencia a las pruebas de E4.
- **Cómo se usó la IA:**
  - Claude Code escribió la spec a partir del plan aprobado, con las decisiones de arriba. Antes midió el tiempo de
    AutoETS en un corte (~12 s) para fijar el presupuesto de la etapa.
  - La implementación la hizo un agente general con las instrucciones de `.claude/agents/forecast-engineer.md`. El
    subagente registrado seguía sin estar disponible, por la misma razón que en E2.
  - Las cifras se verificaron sin usar el código del agente. Los tres baselines cerrados se recalcularon con un
    script aparte y coinciden al centésimo; AutoETS se reajustó por separado en un corte.
  - La revisión del diff la hizo otro agente general, con las instrucciones de `reviewer`. No encontró fugas, pero
    sí defectos reales, y se corrigieron todos:
    - `MLFLOW_TRACKING_URI` del shell podía mandar los runs a un servidor remoto;
    - un modelo con pronósticos nulos podía sacar puntos de su propio denominador;
    - las métricas por segmento no tenían ninguna prueba contra valores calculados a mano.

    La revisión lo comprobó con mutaciones: el código roto seguía pasando las pruebas. Tras corregirlo, cinco de
    seis mutaciones fallan.
- **Qué cambió al ver la primera corrida:**
  - En la tabla por corte, la semana siguiente al Buen Fin tenía ~100 % de error y caía en "resto". Eso hacía parecer
    que AutoETS ganaba en general.
  - Se agregó el nivel `post_buen_fin`. Con él se ve que en semanas normales el sistema actual es el mejor modelo
    (11.0 %, contra 13.4 % de AutoETS), y que su falla está en los picos y en la semana que sigue.
  - Ese hallazgo cambia lo que E4 tiene que demostrar: no basta con ganar en el global; también hay que no perder en
    las semanas normales.

### E4 · Modelos y validación robusta

- **Qué se hizo:**
  - LightGBM global por cuantiles, sobre la matriz de E3 con 8 features nuevas de nivel y de eventos;
  - Chronos-2 sin entrenar, con el calendario y la promoción como covariables;
  - un ajuste de hiperparámetros con orígenes previos al backtest;
  - la prueba Diebold-Mariano por corte;
  - una regla de campeón determinista (`make seleccion`).
- **Resultado:** LightGBM baja el WAPE semanal de 27.3 a 20.3 % y empata en semanas normales. Pero con 17 cortes la
  diferencia no es significativa (p = 0.17), así que la regla mantiene al sistema actual como campeón.
- **Cómo se usó la IA:**
  - **Claude Code escribió la spec**, con el diseño de los modelos y la regla de campeón con sus umbrales, **antes de
    correr ningún modelo nuevo**.
    - La regla, los umbrales y la sensibilidad quedaron fijados por escrito antes de ver resultados. Al implementar no
      se tocaron.
    - Antes de escribirla, midió el costo de LightGBM y de Chronos-2 en cortes reales. Esas mediciones imprimieron un
      WAPE preliminar de tres cortes del backtest; la spec lo declara, y ninguna decisión sale de esas cifras.
  - **La implementación la hizo un agente general** con las instrucciones de `forecast-engineer`. Se desvió de la spec
    en tres puntos, documentados en ella:
    - Chronos-2 corre en un proceso aparte, porque en macOS LightGBM y torch no pueden convivir en un proceso;
    - LightGBM entrena sus tres cuantiles en paralelo;
    - se agregó un parámetro que la spec olvidó, sin el cual el submuestreo no tenía efecto.
  - **La verificación no usó el código del agente.** Con scripts propios se recalcularon las métricas, las 16
    pruebas DM y la regla; Chronos-2 se reajustó en otro ambiente; y una segunda corrida completa salió idéntica byte
    a byte.
  - **La revisión la hizo otro agente** con las instrucciones de `reviewer`. No encontró fugas ni errores en la regla,
    pero sí pruebas que pasaban con el código roto: 6 de 13 mutaciones sobrevivían. El implementador las corrigió y
    ahora fallan las seis.
- **Qué enseñó:**
  - Fijar la regla antes de ver resultados importó. Un 7 pp de mejora es tentador, pero viene de tres semanas
    excepcionales, y con 17 cortes no se distingue del ruido.
  - El resultado honesto es dejar a LightGBM como retador y medir el valor del cambio en E5, no declararlo ganador.

### E5 · Monitor de drift y reglas de decisión

- **Qué se hizo:**
  - una calibración del campeón y el retador en 8 semanas previas al backtest;
  - un newsvendor por categoría, con cuantiles de servicio tomados de los errores semanales de semanas pasadas, y
    tres políticas: puntual, newsvendor con el sistema actual y newsvendor con LightGBM;
  - un monitor de drift en tres capas (calidad, entradas y desempeño), que separa lo esperado por calendario de lo
    inesperado, vigila al retador y traduce cada alerta en una recomendación;
  - la validación del monitor con tres escenarios de drift inyectado y una prueba de potencia.
- **Resultado:**
  - el newsvendor con el pronóstico del sistema actual cuesta 40 % menos que reponer al pronóstico puntual
    (p = 0.019). Cambiar el pronóstico por LightGBM baja otro 9 %, pero sin significancia (p = 0.20);
  - el monitor detecta los tres escenarios y la falla real de captura de una tienda;
  - su chequeo de nivel confunde la estacionalidad anual con drift: alerta en todos los segmentos en noviembre y
    enero.
- **Cómo se usó la IA:**
  - **Claude Code escribió la spec**, con los umbrales, las reglas, los supuestos de costo y los escenarios,
    **antes de correr nada sobre las semanas del backtest**.
    - Para elegir los umbrales prototipó el monitor solo sobre las 35 semanas anteriores. La spec lo declara, junto
      con lo que mostró.
    - La spec también anticipó una limitación: alertas de nivel en enero por la estacionalidad.
  - **La implementación la hizo un agente general** con las instrucciones de `forecast-engineer`. No cambió
    umbrales ni reglas; las precisiones quedaron en la spec.
  - **La verificación no usó el código del agente.** Con dos scripts propios, escritos desde la spec, se
    recalcularon el newsvendor completo y las alertas del monitor. Coinciden.
  - **La revisión la hizo otro agente** con las instrucciones de `reviewer`. No encontró errores de cálculo.
    - Sí encontró que 16 de 42 mutaciones sobrevivían. Entre ellas, una vigilancia del retador que podía mirar
      cortes posteriores sin que ninguna prueba fallara.
    - El implementador agregó las pruebas. Ahora mueren 48 de 50 mutaciones, y las 2 restantes son equivalentes.
- **Qué enseñó:**
  - La decisión pesa más que el modelo. El colchón de servicio vale 40 % del costo con el pronóstico de hoy; el mejor
    modelo agrega 9 %, y solo en las semanas con evento.
  - Pre-registrar también expone los errores de diseño de la spec. La referencia móvil de 56 días no distingue la
    estacionalidad anual, y alertó en noviembre, cosa que la spec no anticipó. Queda documentado y no se ajustó
    mirando esas semanas.

### E6 · Experimento de promociones

- **Qué se hizo:**
  - la verificación formal de que `has_promotion` se asignó al azar, con 19 pruebas de balance;
  - el efecto causal sobre las unidades con efectos fijos de serie y de fecha, errores agrupados por tienda,
    inferencia por aleatorización y una calibración A/A de los errores;
  - el efecto por canal (tráfico, ingreso, precio, canasta), por categoría y formato con Holm, la dinámica (adelanto
    de compras) y el efecto sobre las otras categorías;
  - la potencia del diseño actual y de uno semanal;
  - la traducción a utilidad con márgenes y descuentos como parámetros;
  - el diseño del próximo A/B en `docs/causal_promos.md`.
- **Resultado:**
  - la promoción no mueve las unidades (+0.05 %, IC95 de −0.25 a +0.34 %), salvo en Bebidas (+1.45 %). El nulo es
    preciso: el diseño detecta efectos de 0.42 %;
  - con cualquier descuento de 5 a 20 %, ninguna categoría se paga;
  - **los «conteos inconsistentes» de E1 son exactamente los días con promoción.** Ese día `total_transactions` sube
    22 % sin que cambien los pagos, las unidades ni el monto.
- **Cómo se usó la IA:**
  - **Claude Code escribió la spec con el método, las familias de Holm y las reglas de conclusión antes de estimar el
    efecto.** Lo único que corrió antes fue una prueba de factibilidad con una promoción placebo, declarada en la
    spec.
  - **La implementación la hizo un agente general** con las instrucciones de `causal-analyst`. Cambió una cosa de la
    spec y la justificó con datos: la Wald conjunta sin corregir rechazaba el 46 % de los placebos, así que agregó
    la corrección de Hotelling.
  - **La verificación usó otra librería.** Un script propio con pyfixest, escrito desde la spec, recalculó todos los
    efectos. Coinciden.
    - Ese mismo script encontró el hallazgo de registro antes que el implementador: la bandera de conteos dependía
      por completo de la promoción.
  - **La regla de balance pre-registrada falló por esa bandera**, y se reporta tal cual.
    - La separación entre pruebas de asignación y de medición se agregó después, declarada como tal en el código, la
      spec y el JSON.
    - Fue un error de diseño de la spec: metió una variable medida el mismo día en una regla de asignación.
  - **La revisión la hizo otro agente** con las instrucciones de `reviewer`, con pruebas de mutación. No encontró
    errores en los cálculos, pero sí tres cosas:
    - `make validate` fallaba por tres líneas que Claude Code agregó al separar asignación y medición;
    - la potencia del diseño semanal, extrapolada con 1/√semanas, daba cifras imposibles: una semana para detectar
      5 %, cuando con una semana el efecto no está identificado;
    - sobrevivían 22 de 41 mutaciones.

    Tras los arreglos, la potencia se mide con placebos en ventanas reales y hay 15 pruebas más. Ahora mueren 41 de
    44 mutaciones: 2 de las restantes son equivalentes y la otra solo mueve un grado de libertad.
- **Qué enseñó:**
  - Un experimento natural no solo mide un efecto: también audita los datos. La bandera de calidad más extraña del
    dataset resultó ser la huella de la promoción en el registro.
  - El efecto nulo no basta como respuesta de negocio. Lo que decide es que el IC completo queda lejos del lift de
    equilibrio.
