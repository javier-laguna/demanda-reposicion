<script setup lang="ts">
// Efecto de la promoción sobre las unidades, con IC95 (errores agrupados por tienda).
// Cifras de artifacts/causal/promociones.json, commit fbea251.
const filas = [
  { nombre: 'Todas', efecto: 0.05, ic: [-0.25, 0.34], global: true },
  { nombre: 'Bebidas', efecto: 1.45, ic: [0.61, 2.31], sig: true },
  { nombre: 'Ropa', efecto: 0.29, ic: [-0.35, 0.93] },
  { nombre: 'Hogar', efecto: -0.08, ic: [-0.93, 0.77] },
  { nombre: 'Abarrotes', efecto: -0.31, ic: [-1.17, 0.55] },
  { nombre: 'Cuidado personal', efecto: -0.5, ic: [-1.35, 0.36] },
  { nombre: 'Electrónica', efecto: -0.55, ic: [-1.3, 0.21] },
]
const W = 760
const izq = 150
const der = 30
const arriba = 10
const alto = 34
const H = arriba + filas.length * alto + 30
const min = -2
const max = 4
const x = (v: number) => izq + ((v - min) / (max - min)) * (W - izq - der)
const ticks = [-2, -1, 0, 1, 2, 3, 4]
const color = (f: { sig?: boolean; global?: boolean }) =>
  f.sig ? 'var(--series-1)' : f.global ? 'var(--ink-1)' : 'var(--neutral-mark)'
const fmt = (v: number) => (v > 0 ? '+' : '') + v.toFixed(2).replace('-', '−')
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full" role="img"
    aria-label="Efecto de la promoción en unidades por categoría, con intervalo de confianza">
    <line v-for="t in ticks" :key="t" :x1="x(t)" :x2="x(t)" :y1="arriba" :y2="H - 24"
      :stroke="t === 0 ? 'var(--axis)' : 'var(--grid)'" :stroke-width="t === 0 ? 1.5 : 1" />
    <text v-for="t in ticks" :key="'t' + t" :x="x(t)" :y="H - 8" text-anchor="middle" :style="{ fontSize: '11px' }"
      fill="var(--ink-muted)" style="font-variant-numeric: tabular-nums">{{ fmt(t).replace('.00', '') }} %</text>
    <g v-for="(f, i) in filas" :key="f.nombre">
      <line v-if="f.global" :x1="8" :x2="W - 8" :y1="arriba + alto + 1" :y2="arriba + alto + 1"
        stroke="var(--grid)" stroke-width="1" />
      <text :x="izq - 12" :y="arriba + i * alto + alto / 2 + 4" text-anchor="end" :style="{ fontSize: '13px' }"
        :font-weight="f.global || f.sig ? 650 : 400" fill="var(--ink-1)">{{ f.nombre }}</text>
      <line :x1="x(f.ic[0])" :x2="x(f.ic[1])" :y1="arriba + i * alto + alto / 2"
        :y2="arriba + i * alto + alto / 2" :stroke="color(f)" stroke-width="2" stroke-linecap="round" />
      <circle :cx="x(f.efecto)" :cy="arriba + i * alto + alto / 2" r="6" :fill="color(f)"
        stroke="var(--surface-1)" stroke-width="2">
        <title>{{ f.nombre }}: {{ fmt(f.efecto) }} % (IC95 {{ fmt(f.ic[0]) }} a {{ fmt(f.ic[1]) }} %)</title>
      </circle>
      <text v-if="f.global || f.sig" :x="x(f.ic[1]) + 10" :y="arriba + i * alto + alto / 2 + 4"
        :style="{ fontSize: '11.5px' }" fill="var(--ink-2)">{{ fmt(f.efecto) }} %<template v-if="f.global"> · IC95 {{ fmt(f.ic[0]) }} a {{ fmt(f.ic[1]) }}</template><tspan v-if="f.sig" font-weight="600" fill="var(--ink-1)"> · significativo (Holm)</tspan></text>
    </g>
  </svg>
</template>
