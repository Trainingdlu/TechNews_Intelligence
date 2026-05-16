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
    <li v-for="step in steps" :key="step.step_id" class="span-flow-step">
      <button
        type="button"
        class="span-node span-flow-node"
        :class="[
          `span-status-${step.span.status || 'unknown'}`,
          { selected: step.span.span_id === selectedSpanId }
        ]"
        :aria-current="step.span.span_id === selectedSpanId ? 'true' : undefined"
        @click="emit('select', step.span)"
      >
        <span class="span-rail" aria-hidden="true"></span>
        <span class="span-main">
          <span class="span-title">{{ step.title }}</span>
          <span class="span-meta">
            <span class="span-kind">{{ step.span.span_type_label }}</span>
            <span class="span-state">{{ statusLabel(step.span.status) }}</span>
            <span class="span-duration">{{ formatLatency(step.span.latency_ms) }}</span>
            <span v-if="step.repeat_index > 1" class="span-repeat">第 {{ step.repeat_index }} 次</span>
          </span>
        </span>
      </button>

      <ol v-if="step.items.length" class="span-detail-list">
        <li v-for="item in step.items" :key="item.span_id" class="span-detail-item">
          <button
            type="button"
            class="span-node span-detail-node"
            :class="[`span-status-${item.status || 'unknown'}`, { selected: item.span_id === selectedSpanId }]"
            :aria-current="item.span_id === selectedSpanId ? 'true' : undefined"
            @click="emit('select', item)"
          >
            <span class="span-rail" aria-hidden="true"></span>
            <span class="span-main">
              <span class="span-title">{{ item.display_name }}</span>
              <span class="span-meta">
                <span class="span-kind">{{ item.span_type_label }}</span>
                <span class="span-state">{{ statusLabel(item.status) }}</span>
                <span class="span-duration">{{ formatLatency(item.latency_ms) }}</span>
              </span>
            </span>
          </button>
        </li>
      </ol>
    </li>
  </ol>
</template>
