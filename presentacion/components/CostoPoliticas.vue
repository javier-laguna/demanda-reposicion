<script setup lang="ts">
// Costo total de reposición en los 17 cortes (millones de pesos, con costos supuestos).
// Cifras de artifacts/decision/impacto.json, commit fbea251.
const filas = [
  { politica: 'Reponer al pronóstico', sub: 'puntual · sistema actual', costo: 352.1, nota: 'referencia', ref: true },
  { politica: 'Newsvendor', sub: 'sobre el sistema actual', costo: 212.7, nota: '−40 % · p = 0.019' },
  { politica: 'Newsvendor', sub: 'sobre LightGBM', costo: 193.0, nota: '−9 % adicional · p = 0.20' },
]
const W = 760
const izq = 200
const der = 170
const arriba = 10
const alto = 62
const grosor = 24
const H = arriba + filas.length * alto + 28
const max = 400
const x = (v: number) => izq + (v / max) * (W - izq - der)
const ticks = [0, 100, 200, 300, 400]
// barra con extremo de datos redondeado (4px) y base recta
const barra = (y: number, v: number) => {
  const x0 = izq
  const x1 = x(v)
  const r = 4
  return `M${x0},${y} H${x1 - r} Q${x1},${y} ${x1},${y + r} V${y + grosor - r} Q${x1},${y + grosor} ${x1 - r},${y + grosor} H${x0} Z`
}
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full" role="img"
    aria-label="Costo total por política de reposición">
    <line v-for="t in ticks" :key="t" :x1="x(t)" :x2="x(t)" :y1="arriba" :y2="H - 24"
      stroke="var(--grid)" stroke-width="1" />
    <text v-for="t in ticks" :key="'t' + t" :x="x(t)" :y="H - 8" text-anchor="middle" :style="{ fontSize: '11px' }"
      fill="var(--ink-muted)" style="font-variant-numeric: tabular-nums">{{ t }} M</text>
    <g v-for="(f, i) in filas" :key="i">
      <text :x="izq - 14" :y="arriba + i * alto + 18" text-anchor="end" :style="{ fontSize: '13.5px' }" font-weight="600"
        fill="var(--ink-1)">{{ f.politica }}</text>
      <text :x="izq - 14" :y="arriba + i * alto + 34" text-anchor="end" :style="{ fontSize: '11px' }"
        fill="var(--ink-2)">{{ f.sub }}</text>
      <path :d="barra(arriba + i * alto + 6, f.costo)" :fill="f.ref ? 'var(--neutral-mark)' : 'var(--series-1)'">
        <title>{{ f.politica }} {{ f.sub }}: {{ f.costo.toFixed(1) }} M de pesos</title>
      </path>
      <text :x="x(f.costo) + 10" :y="arriba + i * alto + 18" :style="{ fontSize: '13px' }" font-weight="650"
        fill="var(--ink-1)">{{ f.costo.toFixed(0) }} M</text>
      <text :x="x(f.costo) + 10" :y="arriba + i * alto + 33" :style="{ fontSize: '11px' }"
        fill="var(--ink-2)">{{ f.nota }}</text>
    </g>
    <line :x1="izq" :x2="izq" :y1="arriba" :y2="H - 24" stroke="var(--axis)" stroke-width="1" />
  </svg>
</template>
