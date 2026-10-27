# Promociones: qué dice el experimento natural y cómo diseñar el siguiente A/B

Las cifras salen de `artifacts/causal/promociones.json` (`make causal`), y el método y las pruebas están en
`docs/specs/E6.md`. Los márgenes son los mismos supuestos ilustrativos del newsvendor (`decision/supuestos.py`).

## Lo que ya se sabe

`has_promotion` se asignó al azar, por día, tienda y categoría, al 20 %. Las pruebas de la asignación pasan todas:
el ómnibus, la homogeneidad en 13 dimensiones, las ventas previas y la independencia serial. Eso convierte la
historia en un experimento de 197,840 observaciones.

| Pregunta | Respuesta | Cifra |
|---|---|---|
| ¿Suben las unidades? | No | +0.05 %, IC95 de −0.25 a +0.34 % (p = 0.75; por aleatorización, p = 0.77) |
| ¿En alguna categoría? | Solo en Bebidas | +1.45 %, IC95 de 0.61 a 2.31 % (p de Holm = 0.009); las otras cinco, entre −0.55 y +0.29 %, sin significancia |
| ¿En algún formato? | No | Bodega −0.22 %, Express +0.41 %, Supercenter +0.07 %, todas sin significancia |
| ¿Adelantan compras? | No se detecta | suma de los efectos de los 7 días siguientes: −0.55 %, IC95 de −1.49 a +0.40 % |
| ¿Mueven otras categorías? | No | +0.06 % por cada otra categoría en promoción, IC95 de −0.08 a +0.20 % |
| ¿Bajan el precio? | Apenas | −0.30 % por unidad, IC95 de −0.60 a −0.003 % |
| ¿Atraen tráfico? | No: es un artefacto de registro | `total_transactions` sube 21.9 %, pero las transacciones pagadas (efectivo + tarjeta) no cambian (−0.12 %, IC95 de −0.28 a +0.04 %) |

**El resultado nulo es preciso, no falta de potencia.** El diseño actual detecta efectos de 0.42 % en el global y de
0.9 a 1.2 % por categoría, con α = 0.05 y potencia 0.8.

**El hallazgo de registro.** En los días con promoción, `total_transactions` vale en promedio 1.22 veces efectivo +
tarjeta (rango de 1.08 a 1.35). En los demás días vale exactamente 1. Los «conteos inconsistentes» del ~20 % de las
filas que encontró la etapa de calidad son exactamente los días con promoción. El sistema cuenta como transacción
algo que no trae pago, ni unidades, ni monto. Por ejemplo, la lectura de un cupón o un ticket en cero.

## ¿Se pagan?

Con un descuento d sobre el precio y margen bruto m, una promoción se paga si su lift supera d / (m − d).

| Categoría | Margen (supuesto) | Lift estimado (IC95) | Lift de equilibrio con 5 % | con 10 % |
|---|---|---|---|---|
| Abarrotes | 20 % | −0.31 % (−1.17 a 0.55) | 33 % | 100 % |
| Bebidas | 25 % | +1.45 % (0.61 a 2.31) | 25 % | 67 % |
| Cuidado_Personal | 30 % | −0.50 % (−1.35 a 0.36) | 20 % | 50 % |
| Electronica | 12 % | −0.55 % (−1.30 a 0.21) | 71 % | 500 % |
| Hogar | 30 % | −0.08 % (−0.93 a 0.77) | 20 % | 50 % |
| Ropa | 45 % | +0.29 % (−0.35 a 0.93) | 12.5 % | 29 % |

- **Con descuento:** ninguna categoría se paga con descuentos de 5 a 20 %. El IC95 completo del lift queda muy por
  debajo del equilibrio.
- **Sin descuento** (exhibición o folleto): en Bebidas, el valor de las unidades incrementales es de 479 pesos por
  día-promoción en una tienda (IC95 de 202 a 754). Es lo más que podría costar esa promoción para pagarse.

## Qué hacer ya

1. **No ampliar las promociones como están.** En cinco de las seis categorías no mueven la venta.
2. **Revisar cómo registra el punto de venta las transacciones en los días con promoción.** Mientras tanto, usar
   efectivo + tarjeta como medida de tráfico, nunca `total_transactions`. Esto también afecta al `avg_ticket` y al
   chequeo de ticket del monitor de drift.
3. **En Bebidas,** comparar el costo real de la mecánica contra los 479 pesos por día-promoción.

## Diseño del próximo A/B

**Pregunta.** ¿Qué mecánica y qué profundidad de promoción generan un lift mayor que su lift de equilibrio? Los datos
actuales no lo pueden responder: la promoción es una bandera, sin profundidad ni costo.

**Brazos** por tienda × categoría × semana:
- control, sin promoción;
- la mecánica actual;
- descuento del 10 %;
- descuento del 20 %.

El 10 y el 20 % dan dos puntos de la curva de respuesta al precio. Con margen bajo (Electrónica, 12 %), el brazo de
20 % se reemplaza por exhibición sin descuento, porque d ≥ m nunca se paga.

**Unidad y asignación:**
- **la semana completa como unidad,** no el día. Así el efecto incluye lo que se adelanta dentro de la semana, y la
  operación lo puede ejecutar;
- **asignación por bloques de formato × región,** con proporciones iguales por brazo. Los bloques aseguran el balance
  en las dimensiones donde el negocio querrá cortar el resultado;
- **por categoría, no por tienda completa.** E6 no detectó efecto sobre las otras categorías (+0.06 % por cada una).
  Aun así, se mide como métrica de control.

**Duración.** El efecto mínimo detectable del diseño semanal (480 series, la mitad en promoción, α = 0.05 y potencia
0.8) se midió con asignaciones placebo sobre ventanas reales de W semanas consecutivas:

| Semanas | Ventanas medidas | Efecto mínimo detectable |
|---|---|---|
| 2 | 30 | 3.62 % |
| 3 | 20 | 2.40 % |
| 4 | 15 | 1.93 % |
| 6 | 10 | 1.46 % |
| 8 | 7 | 1.24 % |
| 12 | 5 | 0.98 % |

- Con dos brazos, detectar un 2 % pide 4 semanas, y un 5 %, 2 semanas.
- Con cuatro brazos, cada comparación contra control usa la mitad de las series, y el efecto detectable sube cerca
  de √2. Es un cálculo aparte, no medido: en 8 semanas queda alrededor de 1.8 %.
- **Se recomiendan 8 semanas**, con dos quincenas por mes, fuera del Buen Fin y de la temporada navideña. Detectan
  un 2 % en cada brazo y además miden la semana siguiente.

**Métricas:**
- **primaria:** log de las unidades semanales por serie;
- **secundarias:** el ingreso, el margen bruto (con el costo real del producto y de la mecánica) y las transacciones
  pagadas;
- **de control:**
  - las unidades de la semana siguiente (adelanto);
  - las de las otras categorías de la tienda (canibalización);
  - el fill rate (quiebres: una venta censurada por falta de inventario subestima el lift).

**Análisis, pre-registrado con el mismo estimador de E6:**
- efectos fijos de serie y de semana, errores agrupados por tienda y t de G − 1;
- Holm sobre las comparaciones de cada brazo contra control.
- **Regla de adopción:** una mecánica se adopta en una categoría si el extremo inferior del IC95 de su lift supera
  su lift de equilibrio, calculado con el margen real.

**Datos que hay que registrar desde el primer día:**
- la profundidad del descuento y el costo de la mecánica;
- el margen real por categoría;
- el inventario disponible;
- un conteo de transacciones que no mezcle eventos sin pago.

## Límites

- **Los márgenes son supuestos.** Las reglas de «se paga» cambian con márgenes reales, pero no hay margen razonable
  que alcance un lift de equilibrio de 12.5 % o más.
- **Se observan ventas, no demanda.** Si una promoción agotara el producto, su efecto real sería mayor.
- **El Buen Fin y la Navidad ocurren una vez.** El efecto de las promociones en esos eventos no se puede estimar con un
  solo año.
- **El efecto detectable con cuatro brazos es una aproximación** (×√2 sobre el de dos brazos medido), no una
  simulación del diseño completo.
