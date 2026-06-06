<template>
  <div class="json-tree" :class="{ 'json-tree--nested': level > 0 }">
    <div class="json-tree__row">
      <button
        v-if="isExpandable"
        class="json-tree__toggle"
        type="button"
        :aria-label="collapsed ? '展开字段' : '收起字段'"
        @click="toggleCollapsed"
      >
        {{ collapsed ? '+' : '-' }}
      </button>
      <span v-else class="json-tree__spacer"></span>

      <span v-if="label" class="json-tree__label">{{ label }}</span>
      <span class="json-tree__type">{{ valueType }}</span>
      <span v-if="!isExpandable || collapsed" class="json-tree__preview">{{ previewText }}</span>
    </div>

    <div v-if="isExpandable && !collapsed" class="json-tree__children">
      <JsonTree
        v-for="entry in entries"
        :key="entry.key"
        :label="entry.key"
        :value="entry.value"
        :level="level + 1"
      />
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

defineOptions({
  name: 'JsonTree',
})

// props：JSON 树节点的输入数据
const props = defineProps({
  value: {
    type: null,
    required: true,
  },
  label: {
    type: String,
    default: '',
  },
  level: {
    type: Number,
    default: 0,
  },
})

// collapsed：节点折叠状态，深层节点默认折叠以减少页面噪音
const collapsed = ref(props.level > 1)

// valueType：当前值的可读类型名称
const valueType = computed(() => {
  // 空值判断：null 需要单独展示，避免被 typeof 识别成 object
  if (props.value === null) {
    return 'null'
  }
  // 数组判断：数组展示长度，方便快速扫描复杂字段
  if (Array.isArray(props.value)) {
    return `array(${props.value.length})`
  }
  return typeof props.value
})

// isExpandable：对象和数组可以展开查看子字段
const isExpandable = computed(() => {
  return props.value !== null && typeof props.value === 'object'
})

// entries：当前节点的子字段列表，数组用索引作为字段名
const entries = computed(() => {
  // 非展开节点判断：基础值没有子字段
  if (!isExpandable.value) {
    return []
  }
  // 数组处理：把数组索引转成可显示的 key
  if (Array.isArray(props.value)) {
    return props.value.map((item, index) => ({
      key: `[${index}]`,
      value: item,
    }))
  }
  return Object.entries(props.value).map(([key, value]) => ({
    key,
    value,
  }))
})

// previewText：折叠时显示的摘要文本
const previewText = computed(() => {
  // 空值判断：保留 null 的显式语义
  if (props.value === null) {
    return 'null'
  }
  // 对象摘要：展示字段数量，避免大对象挤占空间
  if (isExpandable.value) {
    return Array.isArray(props.value) ? `${props.value.length} 项` : `${Object.keys(props.value).length} 个字段`
  }
  // 字符串摘要：保留原文并限制长度，防止长日志撑破布局
  if (typeof props.value === 'string') {
    return props.value.length > 120 ? `${props.value.slice(0, 120)}...` : props.value
  }
  return String(props.value)
})

/**
 * 切换 JSON 节点折叠状态。
 *
 * 参数说明：
 * - 无
 *
 * 返回值说明：
 * - 无
 *
 * 异常说明：
 * - 无
 */
function toggleCollapsed() {
  collapsed.value = !collapsed.value
}
</script>
