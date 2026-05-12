<script setup>
import { computed, ref } from "vue";

const props = defineProps({
  title: {
    type: String,
    required: true
  },
  value: {
    type: null,
    required: true
  },
  open: {
    type: Boolean,
    default: true
  }
});

const copied = ref(false);

function providerInternalPlaceholder(key, value) {
  const length = String(value ?? "").length;
  return `[provider_internal_${key}_omitted chars=${length}]`;
}

function sanitizeDisplayValue(value) {
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeDisplayValue(item));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        key === "thought_signature" ? providerInternalPlaceholder(key, item) : sanitizeDisplayValue(item)
      ])
    );
  }
  return value;
}

const displayValue = computed(() => sanitizeDisplayValue(props.value ?? null));
const text = computed(() => JSON.stringify(displayValue.value, null, 2));

async function copyValue() {
  await navigator.clipboard.writeText(text.value);
  copied.value = true;
  window.setTimeout(() => {
    copied.value = false;
  }, 1200);
}
</script>

<template>
  <details class="json-block" :open="open">
    <summary>
      <span>{{ title }}</span>
      <button type="button" class="ghost-button" @click.prevent="copyValue">
        {{ copied ? "已复制" : "复制" }}
      </button>
    </summary>
    <pre>{{ text }}</pre>
  </details>
</template>
