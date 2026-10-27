<script setup lang="ts">
// WAPE semanal por modelo: global (17 cortes) y semanas normales (evento «resto").
// Cifras de artifacts/modelos/seleccion.json, commit fbea251.
const filas = [
  { modelo: 'lightgbm', global: 20.34, resto: 11.3 },
  { modelo: 'chronos2', global: 23.31, resto: 14.72 },
  { modelo: 'autoets', global: 24.71, resto: 13.38 },
  { modelo: 'sistema_actual', global: 27.29, resto: 10.98, campeon: true },
  { modelo: 'naive_estacional', global: 28.29, resto: 13.02 },
  { modelo: 'ma7', global: 28.36, resto: 12.95 },
]
const W = 760
const izq = 150
const der = 40
const arriba = 34
const alto = 34
const H = arriba + filas.length * alto + 30
const max = 30
const x = (v: number) => izq + (v / max) * (W - izq - der)
const ticks = [0, 5, 10, 15, 20, 25, 30]
const etiquetar = new Set(['lightgbm', 'sistema_actual'])
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full" role="img"
    aria-label="WAPE semanal por modelo, global y en semanas normales">
    <g :style="{ fontSize: '12px' }" fill="var(--ink-2)">
      <circle :cx="izq" cy="12" r="5" fill="var(--series-1)" />
      <text :x="izq + 10" y="16">Global · 17 cortes</text>
      <circle :cx="izq + 150" cy="12" r="5" fill="var(--series-2)" />
      <text :x="izq + 160" y="16">Semanas normales · 11 cortes</text>
    </g>
    <g>
      <line v-for="t in ticks" :key="t" :x1="x(t)" :x2="x(t)" :y1="arriba - 6"
        :y2="H - 26" stroke="var(--grid)" stroke-width="1" />
      <text v-for="t in ticks" :key="'l' + t" :x="x(t)" :y="H - 10" text-anchor="middle"
        :style="{ fontSize: '11px' }" fill="var(--ink-muted)" style="font-variant-numeric: tabular-nums">{{ t }} %</text>
    </g>
    <g v-for="(f, i) in filas" :key="f.modelo">
      <rect v-if="f.campeon" :x="4" :y="arriba + i * alto + 2" :width="W - 8" :height="alto - 4"
        rx="6" fill="var(--accent-wash)" />
      <text :x="izq - 12" :y="arriba + i * alto + alto / 2 + 4" text-anchor="end" :style="{ fontSize: '13px' }"
        :font-weight="f.campeon ? 650 : 400" fill="var(--ink-1)">{{ f.modelo }}</text>
      <text v-if="f.campeon" :x="izq - 12" :y="arriba + i * alto + alto / 2 + 16" text-anchor="end"
        :style="{ fontSize: '9.5px' }" fill="var(--ink-muted)">campeón</text>
      <line :x1="x(f.resto)" :x2="x(f.global)" :y1="arriba + i * alto + alto / 2"
        :y2="arriba + i * alto + alto / 2" stroke="var(--axis)" stroke-width="2" stroke-linecap="round" />
      <circle :cx="x(f.resto)" :cy="arriba + i * alto + alto / 2" r="6" fill="var(--series-2)"
        stroke="var(--surface-1)" stroke-width="2">
        <title>{{ f.modelo }} · semanas normales: {{ f.resto.toFixed(1) }} %</title>
      </circle>
      <circle :cx="x(f.global)" :cy="arriba + i * alto + alto / 2" r="6" fill="var(--series-1)"
        stroke="var(--surface-1)" stroke-width="2">
        <title>{{ f.modelo }} · global: {{ f.global.toFixed(1) }} %</title>
      </circle>
      <template v-if="etiquetar.has(f.modelo)">
        <text :x="x(f.resto) - 10" :y="arriba + i * alto + alto / 2 + 4" text-anchor="end"
          :style="{ fontSize: '11.5px' }" fill="var(--ink-2)">{{ f.resto.toFixed(1) }}</text>
        <text :x="x(f.global) + 10" :y="arriba + i * alto + alto / 2 + 4" :style="{ fontSize: '11.5px' }"
          font-weight="600" fill="var(--ink-1)">{{ f.global.toFixed(1) }}</text>
      </template>
    </g>
  </svg>
</template>
