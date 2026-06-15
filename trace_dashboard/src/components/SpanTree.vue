<script setup>
defineOptions({ name: "SpanTree" });

defineProps({
  steps: {
    type: Array,
    required: true
  },
  selectedSpanId: {
    type: String,
    default: ""
  }
});

const emit = defineEmits(["select"]);

function statusLabel(status) {
  const normalized = String(status || "").toLowerCase();
  if (normalized === "success") return "成功";
  if (normalized === "error" || normalized === "failed") return "失败";
  if (normalized === "blocked") return "已拦截";
  if (normalized === "running") return "运行中";
  return status || "未知";
}

function formatLatency(ms) {
  if (ms === null || ms === undefined) return "-";
  const value = Number(ms);
  if (!Number.isFinite(value)) return "-";
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${Math.round(value)}ms`;
}
</script>

<template>
  <ol class="span-flow-list">
    <li v-for="step in steps" :key="step.id" class="span-flow-step">
      <button
        type="button"
        class="span-node span-flow-node"
        :class="[
          `span-kind-${step.kind || 'plumbing'}`,
          `span-status-${step.status || 'unknown'}`,
          { selected: step.id === selectedSpanId, 'is-plumbing': step.kind === 'plumbing' && !step.hasError }
        ]"
        :aria-current="step.id === selectedSpanId ? 'true' : undefined"
        @click="emit('select', step)"
      >
        <span class="span-rail" aria-hidden="true"></span>
        <span class="span-main">
          <span class="span-title">
            <span class="span-kind-dot" :class="`kind-${step.kind || 'plumbing'}`" aria-hidden="true"></span>
            {{ step.title }}
            <span v-if="step.repeat_index > 1" class="span-repeat">第 {{ step.repeat_index }} 次</span>
          </span>
          <span class="span-meta">
            <span v-if="step.signal" class="span-signal">{{ step.signal }}</span>
            <span class="span-state">{{ statusLabel(step.status) }}</span>
            <span class="span-duration">{{ formatLatency(step.latency_ms) }}</span>
          </span>
        </span>
      </button>
    </li>
  </ol>
</template>
