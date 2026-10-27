---
theme: default
title: Pronóstico de demanda para reposición
info: Prueba técnica de Data Science · qué se hizo, qué se encontró y cómo se verificó
author: Javier Laguna
colorSchema: auto
aspectRatio: 16/9
canvasWidth: 980
transition: fade
mdc: true
drawings:
  enabled: false
fonts:
  provider: none
  sans: system-ui
layout: cover
class: text-left
---

<div class="kicker">Prueba técnica · Data Science · retail en México</div>

# Pronóstico de demanda para reposición, con capa de agentes

<p class="muted" style="max-width: 640px">¿Cuánto reponer por tienda y categoría cada semana, y con qué evidencia?</p>

<div class="small muted" style="margin-top: 3rem">80 tiendas · 6 categorías · 425 días · 480 series diarias</div>

---

# El problema que elegí

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

**La pregunta de negocio**

¿Cuánto reponer por tienda × categoría cada semana?

**Por qué esa y no otra**

- Es la única con un **sistema actual** contra el cual medir valor (`replenishment_signal`).
- Las promociones se asignaron **al azar**: eso permite medir causalidad, no solo correlación.
- Otras opciones (tráfico, efectivo) no tenían señal o tenían datos inconsistentes.

</div>
<div>

**Los datos**

| Tabla | Grano |
|---|---|
| `transactions` | fecha × tienda × categoría (~204 k filas) |
| `stores` | 80 tiendas: formato, región, NSE, tamaño |
| `calendar` | 425 días: festivos, Buen Fin, quincenas |

<p class="small muted">Los datos nunca se suben al repo: se colocan con <code>make data</code> y se verifican contra un manifiesto SHA-256.</p>

</div>
</div>

---

# Seis preguntas, seis respuestas

| Pregunta | Respuesta |
|---|---|
| ¿El sistema actual es honesto? | **No del todo:** su señal mira un día al futuro. Sin la fuga, su WAPE semanal sube de 18.6 a **20.1 %** |
| ¿Un modelo nuevo es mejor? | **Todavía no se puede afirmar:** LightGBM baja el WAPE de 27.3 a 20.3 %, pero p = 0.17 con 17 cortes |
| ¿Dónde está el valor? | **En la regla de decisión:** el newsvendor cuesta **40 % menos** (p = 0.019); LightGBM agrega 9 %, sin significancia |
| ¿El monitor de drift funciona? | **Sí, con un límite:** detecta los 3 escenarios inyectados y una falla real, pero confunde la estacionalidad anual |
| ¿Las promociones venden más? | **No:** +0.05 % (IC95 −0.25 a +0.34 %). Solo Bebidas responde, y ningún descuento se paga |
| ¿Hay algo raro en los datos? | **Sí:** el 20 % de «conteos inconsistentes» son exactamente los días con promoción |

---

# La solución, de punta a punta

<img src="/arquitectura.png" class="shot" style="width: 100%" />

<div class="grid grid-cols-3 gap-4 mt-3 small">
<div><b>Pipeline determinista:</b> todo lo que calcula o decide es código con pruebas. Un comando: <code>make pipeline</code>.</div>
<div><b>Capa de IA:</b> los agentes leen por MCP, no calculan ni eligen el modelo, y cada cifra pasa por un validador.</div>
<div><b>Desarrollo:</b> Claude Code construyó el repo; nada del producto depende de él.</div>
</div>

---
layout: section
---

<div class="kicker">1 · Datos</div>

# Antes de modelar, auditar

---

# La señal del sistema actual mira al futuro

<div class="fuente mt-1">Tablero › Datos</div>
<img src="/tab_fuga.png" class="shot" style="width: 84%" />

<div class="grid grid-cols-2 gap-6 mt-3">
<div class="callout">

**Consecuencia:** con origen T solo se pueden usar valores de la señal fechados en **T−1 o antes**. Medido así, el sistema actual tiene un WAPE semanal de **20.1 %**, no 18.6 %. Esa es la vara honesta.

</div>
<div class="small muted">

- El diccionario no lo menciona. Lo descubrió la prueba de fuga de E1: una regresión de la señal sobre las ventas de t−7 a t+7 concentra el peso de t−5 a t+1 (correlación intraserie 0.95).
- El backtest lo impone por construcción: cada modelo recibe solo `historia` (≤ T, con la señal de T en nulo) y `futuro` (calendario, tienda y promoción).

</div>
</div>

---

# Otros hallazgos de calidad

<div class="grid grid-cols-2 gap-5 mt-4">
<div class="card">
<h3>Conteos inconsistentes = días de promoción</h3>
<p>En ~20 % de las filas, <code>cash + card &lt; total_transactions</code>. Son exactamente los días con promoción: <code>total_transactions</code> vale 1.22 × los pagos, y el exceso son registros sin pago. Lo encontró el experimento de E6.</p>
</div>
<div class="card">
<h3>Solo 42 filas faltan</h3>
<p>El panel es la malla completa (204,000 filas) con las faltantes marcadas. Así un <code>shift</code> por serie siempre equivale a días de calendario. Nada se imputa.</p>
</div>
<div class="card">
<h3>Calendario fuerte y multiplicativo</h3>
<p>Buen Fin ×2.65, fin de semana ×1.53, quincena ×1.38 y Semana Santa ×0.66, casi igual en todas las categorías. Buen Fin y Semana Santa ocurren una sola vez.</p>
</div>
<div class="card">
<h3>Contratos que no mienten</h3>
<p>El contrato exige solo lo que siempre debe cumplirse (tipos, llaves, montos que cuadran). Los problemas conocidos se miden y se marcan, no rompen el pipeline.</p>
</div>
</div>

---
layout: section
---

<div class="kicker">2 · Pronóstico</div>

# ¿Un modelo nuevo le gana al sistema actual?

---

# El protocolo, fijado antes de ver resultados

<div class="grid grid-cols-3 gap-5 mt-6">
<div class="card">
<h3>17 cortes semanales</h3>
<p>Los jueves del 2023-11-02 al 2024-02-22, con horizonte de 7 días. Cubren Buen Fin, Navidad y enero–febrero.</p>
</div>
<div class="card">
<h3>Métrica: WAPE de la suma semanal</h3>
<p>Es la granularidad de la reposición. Medido por día, el sistema actual (una media de 7 días) quedaría castigado injustamente: 45.7 %.</p>
</div>
<div class="card">
<h3>Prueba de fugas</h3>
<p>Todo modelo registrado pasa una prueba que altera el futuro y exige el mismo pronóstico.</p>
</div>
</div>

<div class="grid grid-cols-3 gap-5 mt-5">
<div class="card">
<h3>6 modelos</h3>
<p>El sistema actual sin fuga, naive estacional, MA7, AutoETS, LightGBM global por cuantiles y Chronos-2 sin entrenar.</p>
</div>
<div class="card">
<h3>Regla de campeón pre-registrada</h3>
<p>Para cambiar de modelo: menor WAPE, Diebold-Mariano significativo y sin sesgo excesivo en ningún segmento.</p>
</div>
<div class="card">
<h3>Ajuste fuera del backtest</h3>
<p>Los hiperparámetros de LightGBM salen de orígenes previos al primer corte. Nunca se ajusta con los cortes que se evalúan.</p>
</div>
</div>

---

# LightGBM gana en el global; en semanas normales, empate

<WapeModelos class="mt-2" />

<div class="callout mt-3">

La ventaja de LightGBM viene de **tres semanas excepcionales** (la posterior al Buen Fin y las dos de fin de año). Con 17 cortes, la Diebold-Mariano da **p = 0.17**: la regla mantiene al **sistema actual como campeón** y deja a LightGBM como retador en vigilancia.

</div>

---

# El sistema actual falla en los picos, no en lo normal

<div class="grid grid-cols-3 gap-5 mt-6">
<div class="card">
<div class="stat">11.0 %</div>
<div class="stat-label">WAPE del sistema actual en semanas normales: el mejor de los seis</div>
</div>
<div class="card">
<div class="stat">−50 %</div>
<div class="stat-label">su sesgo en el Buen Fin: pide la mitad</div>
</div>
<div class="card">
<div class="stat">+101 %</div>
<div class="stat-label">su sesgo la semana siguiente: su señal promedia el pico y pide el doble</div>
</div>
</div>

<div class="grid grid-cols-2 gap-6 mt-6 small">
<div>

**Chronos-2** (sin entrenar, con covariables): 23.3 % global, con un sesgo de −14 %.

**Los intervalos del 80 %** de LightGBM y Chronos-2 cubren solo el 67–69 %. Por eso la decisión no se apoya en ellos, sino en los errores empíricos.

</div>
<div class="callout">

**Lo que enseñó:** fijar la regla antes de ver resultados importó. Siete puntos de mejora son tentadores, pero no se distinguen del ruido con 17 semanas. Harían falta ~34.

</div>
</div>

---

# Una tienda, los 17 cortes

<div class="fuente">Tablero › Pronóstico</div>
<img src="/tab_pronostico.png" class="shot" style="width: 74%" />

<p class="small mt-3">Abarrotes en <code>STR_001</code>. El campeón (azul) es plano dentro de cada semana: no ve el Buen Fin, y la semana siguiente su pronóstico sube a más del doble cuando la venta ya bajó. LightGBM (naranja) sigue el perfil semanal, pero tampoco anticipa el pico. En el tablero se elige cualquier tienda y categoría.</p>

---

# Dónde falla cada modelo

<div class="fuente">Tablero › Modelos</div>
<img src="/tab_evento.png" class="shot" style="width: 70%" />

<div class="grid grid-cols-2 gap-6 mt-3 small">
<div>

**Buen Fin:** los seis modelos piden la mitad (sesgo de −49 a −55 %). Ninguno puede aprenderlo: ocurre una sola vez en los datos.

</div>
<div>

**La semana siguiente:** los que promedian lo reciente (sistema actual, naive y MA7) piden casi el doble, +95 a +101 %. LightGBM falla 21 % y Chronos-2, 13 %.

</div>
</div>

---

# El retador queda en vigilancia

<div class="fuente">Tablero › Decisión y drift</div>
<img src="/tab_retador.png" class="shot" style="width: 80%" />

<p class="small mt-3">WAPE semanal acumulado al cerrar cada corte. La brecha se abre la semana después del Buen Fin y ya no se cierra, pero la Diebold-Mariano acumulada termina en <b>p = 0.166</b>. Como se revisa cada semana, el umbral lleva Bonferroni (0.05 / 17), y se estiman <b>34 cortes</b> para poder decidir. El monitor lo repite en cada corte, sin intervención.</p>

---
layout: section
---

<div class="kicker">3 · Decisión y drift</div>

# Del pronóstico a cuánto pedir

---

# La decisión pesa más que el modelo

<CostoPoliticas class="mt-1" style="max-width: 820px" />

<div class="grid grid-cols-3 gap-5 mt-3 small">
<div>

**Newsvendor por categoría:** el nivel objetivo es S = Σŷ × q, con q el cuantil de los errores pasados en la fracción crítica.

</div>
<div>

**Los quiebres bajan de 12.8 a 5.6 %** de la demanda. El costo está en los eventos: el Buen Fin pesa **114 de los 213 M**.

</div>
<div>

**En semanas normales, LightGBM y el sistema actual empatan.** Los márgenes y costos de mantener son supuestos ilustrativos; el precio sale de los datos.

</div>
</div>

---

# Un monitor de drift en tres capas

<div class="grid grid-cols-3 gap-5 mt-5">
<div class="card">
<h3>Calidad</h3>
<p>Nulos, filas faltantes y conteos. Detectó la falla real de captura de <code>STR_055</code>.</p>
</div>
<div class="card">
<h3>Entradas</h3>
<p>Nivel de demanda, tasa de promoción, ticket y mezcla de calendario. Separa lo <b>esperado por calendario</b> de lo inesperado.</p>
</div>
<div class="card">
<h3>Desempeño</h3>
<p>El WAPE semanal del campeón contra su banda, y la vigilancia del retador. Cada alerta trae una recomendación.</p>
</div>
</div>

<div class="grid grid-cols-2 gap-6 mt-6">
<div class="small">

**Validado con drift inyectado:** una falla de POS en una tienda, un salto de +30 % en una región y un cambio en la tasa de promoción. Detecta los tres, y en un panel sintético limpio no da alertas inesperadas.

</div>
<div class="callout">

**Límite declarado:** su referencia de 56 días no distingue la estacionalidad anual, y alerta en todos los segmentos en noviembre y enero. Se documentó; no se ajustó mirando esas semanas.

</div>
</div>

---

# Las alertas, semana a semana

<div class="grid grid-cols-4 gap-5">
<div class="col-span-3">
<div class="fuente">Tablero › Decisión y drift</div>
<img src="/tab_drift.png" class="shot" style="width: 100%" />
</div>
<div class="small col-span-1 mt-6">

**Entradas:** desde el primer corte alertan casi todos los segmentos. En las semanas con evento son esperadas; en noviembre y enero salen inesperadas. Es el límite de la referencia de 56 días.

**Desempeño:** 15 alertas en el backtest, 14 esperadas por evento. La única inesperada es del retador, la semana del 11 de enero.

<p class="muted">Arriba de estas dos va la capa de calidad: 5 alertas en 52 semanas.</p>

</div>
</div>

---
layout: section
---

<div class="kicker">4 · Promociones</div>

# Un experimento natural

---

# Las promociones no mueven las unidades

<EfectoPromo class="mt-2" />

<div class="grid grid-cols-2 gap-6 mt-2 small">
<div>

**Método:** log-OLS con efectos fijos de serie y de fecha, errores agrupados por tienda, inferencia por aleatorización (2000 permutaciones), calibración A/A y Holm por familias.

</div>
<div class="callout">

**Un nulo preciso:** el diseño detecta efectos de 0.42 %. Con márgenes supuestos, **ningún descuento de 5 a 20 % se paga**, ni siquiera en Bebidas.

</div>
</div>

---

# ¿Alguna categoría se paga?

<div class="fuente">Tablero › Promociones</div>
<img src="/tab_lift.png" class="shot" style="width: 70%" />

<div class="grid grid-cols-2 gap-6 mt-3 small">
<div>

El **lift de equilibrio** (barra clara) es el aumento de unidades con el que la promoción apenas se paga, con el margen y el descuento supuestos. El selector cambia el descuento.

</div>
<div class="callout">

Con el descuento implícito en el precio (0.3 %), Bebidas y Ropa quedan indeterminadas y el resto no se paga. Con 5 a 20 %, **ninguna se paga**.

</div>
</div>

---

# El experimento también auditó los datos

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

**La bandera más rara del dataset era la huella de la promoción.**

- En los días con promoción, `total_transactions` sube **22 %**…
- …pero no cambian los pagos, ni las unidades, ni el monto.
- El exceso son **registros sin pago**.
- Consecuencia: el ticket correcto es `amount_total / (cash + card)`. Usar `total_transactions` lo subestima ~18 % esos días.

</div>
<div>

**Lo honesto: una regla pre-registrada que falló**

<div class="callout">

La prueba de balance pre-registrada falló, solo por esa bandera, que es un **resultado** de la promoción y no una covariable. Las pruebas de la asignación pasan todas.

La separación entre asignación y medición se decidió **después** de ver los datos, y está declarada así en el código, la spec y el JSON.

</div>

<p class="small muted mt-3">El próximo A/B propuesto: 4 brazos por tienda × categoría × semana, en bloques de formato × región, durante 8 semanas.</p>

</div>
</div>

---
layout: section
---

<div class="kicker">5 · Capa de agentes</div>

# Agentes que interpretan, no deciden

---

# Arquitectura de la capa de agentes

<div class="grid grid-cols-5 gap-6 mt-2">
<div class="col-span-3">
<img src="/capa_ia.png" class="shot" style="width: 100%" />
</div>
<div class="col-span-2 small">

- **Pydantic AI**, agnóstico de proveedor. Por omisión Claude Opus 5.5; con `LLM_MODEL`, OpenRouter o cualquier gateway compatible con OpenAI.
- **El servidor MCP** también sirve a clientes externos, por stdio o HTTP.

</div>
</div>

<div class="grid grid-cols-3 gap-4 mt-4">
<div class="card"><h3>Validador de cifras</h3><p>Una cifra solo pasa si aparece en lo que el agente leyó de las herramientas <b>en esa corrida</b>. Si falla, reintenta; si vuelve a fallar, usa una plantilla.</p></div>
<div class="card"><h3>Tres modos</h3><p>En vivo con key. Sin key, una <b>caché versionada</b>, válida solo si coinciden la huella de los artefactos y la versión del prompt. Si no, plantillas deterministas.</p></div>
<div class="card"><h3>Costo acotado</h3><p>~USD 0.44 y ~2 min por corrida, con un freno de USD 1 por corrida y uno de USD 20 acumulados.</p></div>
</div>

---

# Lo que concluyen los agentes

<div class="grid grid-cols-2 gap-5 mt-5">
<div class="card"><h3>1 · Mantener el sistema actual</h3><p>Como pronóstico oficial, y medirlo contra la referencia correcta (sin la fuga).</p></div>
<div class="card"><h3>2 · El beneficio está en cómo se pide</h3><p>No en cambiar de modelo: el colchón calibrado baja el costo y los quiebres.</p></div>
<div class="card"><h3>3 · Las promociones no venden más</h3><p>El aumento de tráfico es un error de registro, no demanda.</p></div>
<div class="card"><h3>4 · Buen Fin necesita plan manual</h3><p>Y el monitor no debe disparar reentrenamientos por estacionalidad.</p></div>
</div>

<div class="grid grid-cols-2 gap-6 mt-6 small">
<div>

**183 cifras** revisadas por el validador, ninguna rechazada. Un script independiente encontró en los artefactos los 161 números del texto.

</div>
<div class="callout">

**Límite medido:** el validador comprueba que la cifra exista, no que la frase la use bien. Una cifra inventada con un decimal pasa el 13–15 % de las veces por coincidencia. Por eso también se lee el texto.

</div>
</div>

---

# El tablero

<div class="grid grid-cols-2 gap-5 mt-3">
<div>
<img src="/resumen.png" class="shot" />
<p class="small muted mt-2">Resumen: indicadores y conclusiones del orquestador, con su estado de validación.</p>
</div>
<div>
<img src="/pronostico.png" class="shot" />
<p class="small muted mt-2">Pronóstico por tienda × categoría: real, campeón y retador con su banda P10–P90.</p>
</div>
</div>

<p class="small mt-3">Streamlit, 7 páginas: resumen, datos, modelos, pronóstico, decisión y drift, promociones y arquitectura. <b>Solo lee artefactos</b>: no recalcula nada. Las capturas con la etiqueta «Tablero» salen de él. Se levanta con <code>make app</code> o <code>docker compose up app</code>.</p>

---
layout: section
---

<div class="kicker">6 · Proceso</div>

# Cómo se trabajó con IA

---

# Un ciclo por etapa, con verificación independiente

<div class="grid grid-cols-4 gap-4 mt-6">
<div class="card"><div class="kicker">1</div><h3>Spec pre-registrada</h3><p>Método, umbrales y reglas de conclusión, escritos <b>antes</b> de correr nada.</p></div>
<div class="card"><div class="kicker">2</div><h3>Implementación</h3><p>Un agente con instrucciones de rol (<code>.claude/agents/</code>). Los desvíos quedan en la spec.</p></div>
<div class="card"><div class="kicker">3</div><h3>Verificación</h3><p>Scripts propios escritos desde la spec, sin el código del agente. En E6, con otra librería (pyfixest).</p></div>
<div class="card"><div class="kicker">4</div><h3>Revisión con mutaciones</h3><p>Otro agente rompe el código a propósito. Si las pruebas siguen pasando, faltan pruebas.</p></div>
</div>

<div class="grid grid-cols-2 gap-6 mt-6">
<div class="small">

| Etapa | Mutaciones que sobrevivían | Después de los arreglos |
|---|---|---|
| E5 · decisión y drift | 16 de 42 | mueren 48 de 50 |
| E6 · promociones | 22 de 41 | mueren 41 de 44 |
| E7 · agentes | 18 de 45 | mueren 49 de 51 |
| E8 · tablero | 19 de 35 | mueren las 34 que aplican |

</div>
<div class="small">

**Bugs reales que atrapó la revisión:**
- el validador dejaba pasar «15,7 %» como «15»;
- la potencia del A/B semanal, extrapolada, daba cifras imposibles;
- la imagen Docker copiaba un caché con salidas del EDA;
- `MLFLOW_TRACKING_URI` del shell podía mandar runs a un servidor remoto.

</div>
</div>

---

# Reproducible por dos rutas

<div class="grid grid-cols-2 gap-8 mt-5">
<div>

```bash
make setup
make data DATA_SRC="ruta/al/zip"
make pipeline        # ~25–35 min
make app             # :8501
```

```bash
docker compose build
docker compose run --rm pipeline   # ~40 min
docker compose up app mcp mlflow
```

</div>
<div class="small">

**Ensayo en limpio** con dos clones nuevos, siguiendo el README al pie de la letra:

- **macOS (uv):** los 9 artefactos salen **idénticos byte a byte** a los del repo de desarrollo.
- **Linux (Docker):** las mismas cifras, y la caché de insights sigue valiendo entre plataformas.

**CI:** lint, tipos, 451 pruebas sin datos reales, gitleaks sobre el historial, `docker build` y una prueba de humo del pipeline completo sobre 40 tiendas sintéticas (~1 min).

</div>
</div>

---

# Límites, dichos en voz alta

<div class="grid grid-cols-2 gap-x-8 gap-y-2 mt-5" style="font-size: 0.92rem">
<div>

- **Buen Fin y Semana Santa ocurren una sola vez**: su efecto no se valida fuera de muestra.
- **17 cortes** no bastan para afirmar que LightGBM es mejor; harían falta ~34.
- **Los intervalos** del 80 % cubren solo el 67–69 %.
- **Los costos** del newsvendor son supuestos, no datos.
- **Ventas ≠ demanda**: los quiebres no se observan.

</div>
<div>

- **Los datos parecen sintéticos**: aleatorización perfecta y un calendario casi idéntico por categoría.
- **El monitor** confunde la estacionalidad anual con drift.
- **El validador** comprueba la cifra, no la interpretación.
- **Chronos-2** necesita red la primera vez.
- **La imagen** pesa 3.4 GB, sobre todo por torch.

</div>
</div>

---

# Para llevarlo a producción

<div class="grid grid-cols-4 gap-4 mt-6">
<div class="card"><h3>Secretos</h3><p>Secret Manager con Workload Identity, sin llaves JSON; GitHub → GCP por OIDC; el LLM vía un gateway corporativo.</p></div>
<div class="card"><h3>Datos</h3><p>GCS con versionado de objetos y el mismo manifiesto de hashes; DVC como opción.</p></div>
<div class="card"><h3>Escala</h3><p>PySpark en Dataproc o BigQuery para llegar a SKU; entrenamiento distribuido del modelo global.</p></div>
<div class="card"><h3>Operación</h3><p>Corrida semanal agendada, registro de modelos en MLflow, el monitor como job con alertas y el MCP detrás de autenticación.</p></div>
</div>

<div class="callout mt-8">

**Siguiente paso de negocio:** correr el A/B de promociones de 8 semanas y acumular cortes. Con ~34 semanas, la misma regla, sin tocarla, vuelve a decidir con más evidencia.

</div>

---
layout: center
class: text-center
---

# Gracias

<p class="muted">El detalle de cada etapa (lo pre-registrado, los cambios al implementar y el resultado) está en <code>docs/specs/</code>.</p>

<p class="small" style="margin-top: 2rem"><a href="https://github.com/javier-laguna/demanda-reposicion">github.com/javier-laguna/demanda-reposicion</a></p>
