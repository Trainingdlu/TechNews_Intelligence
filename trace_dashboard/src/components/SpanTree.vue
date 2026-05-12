<script setup>
defineOptions({ name: "SpanTree" });

defineProps({
  nodes: {
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
  <ul class="span-tree-list">
    <li v-for="node in nodes" :key="node.span_id" class="span-tree-item">
      <button
        type="button"
        class="span-node"
        :class="[`span-status-${node.status || 'unknown'}`, { selected: node.span_id === selectedSpanId }]"
        @click="emit('select', node)"
      >
        <span class="span-rail" aria-hidden="true"></span>
        <span class="span-main">
          <span class="span-title">{{ node.display_name }}</span>
          <span class="span-meta">
            <span>{{ node.span_type_label }}</span>
            <span>{{ statusLabel(node.status) }}</span>
            <span>{{ formatLatency(node.latency_ms) }}</span>
          </span>
        </span>
      </button>
      <SpanTree
        v-if="node.children && node.children.length"
        :nodes="node.children"
        :selected-span-id="selectedSpanId"
        @select="emit('select', $event)"
      />
    </li>
  </ul>
</template>
