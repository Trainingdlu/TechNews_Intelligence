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

const text = computed(() => JSON.stringify(props.value ?? null, null, 2));

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
