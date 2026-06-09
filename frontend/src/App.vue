<template>
  <div class="app-shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">LangGraph Multi-Agent</p>
        <h1>AI 工单工作台</h1>
      </div>
      <div class="topbar__actions">
        <span class="status-pill" :class="healthClass">{{ healthText }}</span>
        <button class="icon-button" type="button" aria-label="刷新" @click="refreshAll">刷新</button>
      </div>
    </header>

    <main class="workspace">
      <aside class="sidebar">
        <section class="panel">
          <div class="panel__header">
            <h2>新建工单</h2>
          </div>
          <form class="ticket-form" @submit.prevent="submitTicket">
            <label>
              <span>工单 ID</span>
              <input v-model.trim="createForm.ticket_id" type="text" placeholder="TKT-001" />
            </label>
            <label>
              <span>故障现象</span>
              <textarea v-model.trim="createForm.symptom" rows="5" placeholder="描述故障表现、影响范围和关键日志"></textarea>
            </label>
            <button class="primary-button" type="submit" :disabled="loading.submit || !createForm.ticket_id || !createForm.symptom">
              {{ loading.submit ? '提交中' : '提交工单' }}
            </button>
            <p v-if="errors.submit" class="error-text">{{ errors.submit }}</p>
          </form>
        </section>

        <section class="panel panel--list">
          <div class="panel__header">
            <h2>工单列表</h2>
            <span>{{ filteredTickets.length }}/{{ tickets.length }}</span>
          </div>
          <div class="filters">
            <input v-model.trim="filters.keyword" type="search" placeholder="搜索 ID、症状、诊断" />
            <select v-model="filters.status">
              <option value="all">全部状态</option>
              <option v-for="option in statusOptions" :key="option.value" :value="option.value">
                {{ option.label }}
              </option>
            </select>
          </div>
          <div class="ticket-list">
            <button
              v-for="ticket in filteredTickets"
              :key="ticket.ticket_id"
              class="ticket-row"
              :class="{ 'ticket-row--active': ticket.ticket_id === selectedTicketId }"
              type="button"
              @click="selectTicket(ticket.ticket_id)"
            >
              <span class="ticket-row__main">
                <strong>{{ ticket.ticket_id }}</strong>
                <small>{{ ticket.symptom || '无故障描述' }}</small>
              </span>
              <span class="status-dot" :class="statusClass(ticket.status)">{{ statusLabel(ticket.status) }}</span>
            </button>
            <p v-if="!loading.tickets && filteredTickets.length === 0" class="empty-text">暂无工单</p>
            <p v-if="errors.tickets" class="error-text">{{ errors.tickets }}</p>
          </div>
        </section>
      </aside>

      <section class="content">
        <div v-if="!selectedTicket" class="empty-state">
          <h2>暂无详情</h2>
          <p>等待工单数据</p>
        </div>

        <template v-else>
          <section class="detail-header">
            <div>
              <p class="eyebrow">{{ selectedTicket.ticket_id }}</p>
              <h2>{{ selectedTicket.symptom }}</h2>
            </div>
            <div class="detail-header__meta">
              <span class="status-pill" :class="statusClass(selectedTicket.status)">{{ statusLabel(selectedTicket.status) }}</span>
              <span class="status-pill" :class="urgencyClass(selectedTicket.urgency)">{{ urgencyLabel(selectedTicket.urgency) }}</span>
              <span class="status-pill">{{ diagnosisTypeLabel(selectedTicket.diagnosis_type) }}</span>
            </div>
          </section>

          <section class="metrics-grid">
            <div class="metric">
              <span>证据</span>
              <strong>{{ evidenceItems.length }}</strong>
            </div>
            <div class="metric">
              <span>协作消息</span>
              <strong>{{ communicationMessages.length }}</strong>
            </div>
            <div class="metric">
              <span>修复步骤</span>
              <strong>{{ fixSteps.length }}</strong>
            </div>
            <div class="metric">
              <span>Trace</span>
              <strong>{{ traceEvents.length }}</strong>
            </div>
            <div class="metric">
              <span>Agent 操作</span>
              <strong>{{ flowSteps.length }}</strong>
            </div>
          </section>

          <nav class="tabs" aria-label="详情分组">
            <button
              v-for="tab in detailTabs"
              :key="tab.value"
              type="button"
              :class="{ 'tabs__button--active': activeTab === tab.value }"
              @click="activeTab = tab.value"
            >
              {{ tab.label }}
            </button>
          </nav>

          <section v-if="activeTab === 'process'" class="tab-panel">
            <div class="process-layout">
              <article class="panel process-hero">
                <div class="panel__header">
                  <h3>当前进展</h3>
                  <span class="status-pill" :class="currentPhase.className">{{ currentPhase.label }}</span>
                </div>
                <p class="process-hero__diagnosis">{{ primaryDiagnosisText }}</p>
                <dl class="field-grid field-grid--compact">
                  <div>
                    <dt>故障类型</dt>
                    <dd>{{ primaryFaultType }}</dd>
                  </div>
                  <div>
                    <dt>置信度</dt>
                    <dd>{{ primaryConfidence }}</dd>
                  </div>
                  <div>
                    <dt>下一步</dt>
                    <dd>{{ nextActionText }}</dd>
                  </div>
                </dl>
              </article>

              <article class="panel process-action">
                <div class="panel__header">
                  <h3>处理动作</h3>
                  <span>{{ approvalLabel(selectedTicket.approval_status) }}</span>
                </div>
                <template v-if="canApproveSelectedTicket">
                  <form class="approval-form approval-form--stacked" @submit.prevent="approveSelectedTicket">
                    <select v-model="approvalForm.approved">
                      <option :value="true">批准</option>
                      <option :value="false">拒绝</option>
                    </select>
                    <input v-model.trim="approvalForm.comments" type="text" placeholder="审批意见" />
                    <button class="primary-button" type="submit" :disabled="loading.approval">
                      {{ loading.approval ? '处理中' : '提交审批' }}
                    </button>
                  </form>
                  <p v-if="errors.approval" class="error-text">{{ errors.approval }}</p>
                </template>
                <p v-else class="empty-text">{{ actionPanelText }}</p>
              </article>
            </div>

            <article class="panel">
              <div class="panel__header">
                <h3>主流程</h3>
                <span>{{ completedStageCount }}/{{ processStages.length }}</span>
              </div>
              <div class="stage-rail">
                <div
                  v-for="stage in processStages"
                  :key="stage.key"
                  class="stage-card"
                  :class="stage.className"
                >
                  <div class="stage-card__index">{{ stage.index }}</div>
                  <div class="stage-card__body">
                    <strong>{{ stage.title }}</strong>
                    <span>{{ stage.summary }}</span>
                    <small>{{ stage.meta }}</small>
                  </div>
                </div>
              </div>
            </article>

            <div class="process-layout">
              <article class="panel">
                <div class="panel__header">
                  <h3>关键协作</h3>
                  <span>{{ communicationChains.length }}</span>
                </div>
                <div v-if="primaryCommunicationChain" class="compact-chain">
                  <div
                    v-for="message in primaryCommunicationChain.messages"
                    :key="message._key"
                    class="compact-message"
                    :class="messageTypeClass(message.msg_type)"
                  >
                    <strong>{{ messageTypeLabel(message.msg_type) }}</strong>
                    <span>{{ message.sender || 'unknown' }} -> {{ message.receiver || 'broadcast' }}</span>
                    <p>{{ message.content || message.hypothesis || '无消息内容' }}</p>
                  </div>
                </div>
                <p v-else class="empty-text">暂无协作链路</p>
              </article>

              <article class="panel">
                <div class="panel__header">
                  <h3>修复摘要</h3>
                  <span>{{ fixSteps.length }} 步</span>
                </div>
                <dl class="field-grid">
                  <div>
                    <dt>方案</dt>
                    <dd>{{ fixPlan.description || fixPlan.plan_id || '暂无方案' }}</dd>
                  </div>
                  <div>
                    <dt>风险</dt>
                    <dd>{{ fixPlan.risk_level || '未知' }}</dd>
                  </div>
                  <div>
                    <dt>预计耗时</dt>
                    <dd>{{ fixPlan.estimated_time || '未知' }}</dd>
                  </div>
                  <div>
                    <dt>验证</dt>
                    <dd>{{ verificationSummary }}</dd>
                  </div>
                </dl>
              </article>
            </div>
          </section>

          <section v-if="activeTab === 'overview'" class="tab-panel">
            <div class="two-column">
              <article class="panel">
                <div class="panel__header">
                  <h3>诊断总览</h3>
                </div>
                <dl class="field-grid">
                  <div v-for="field in diagnosisFields" :key="field.key">
                    <dt>{{ field.label }}</dt>
                    <dd>{{ field.value }}</dd>
                  </div>
                </dl>
              </article>

              <article class="panel">
                <div class="panel__header">
                  <h3>审批</h3>
                </div>
                <dl class="field-grid">
                  <div>
                    <dt>审批状态</dt>
                    <dd>{{ approvalLabel(selectedTicket.approval_status) }}</dd>
                  </div>
                  <div>
                    <dt>审批意见</dt>
                    <dd>{{ selectedTicket.approver_comments || '无' }}</dd>
                  </div>
                </dl>
                <form class="approval-form" @submit.prevent="approveSelectedTicket">
                  <select v-model="approvalForm.approved">
                    <option :value="true">批准</option>
                    <option :value="false">拒绝</option>
                  </select>
                  <input v-model.trim="approvalForm.comments" type="text" placeholder="审批意见" />
                  <button class="primary-button" type="submit" :disabled="loading.approval">
                    {{ loading.approval ? '处理中' : '提交审批' }}
                  </button>
                </form>
                <p v-if="errors.approval" class="error-text">{{ errors.approval }}</p>
              </article>
            </div>

            <article class="panel">
              <div class="panel__header">
                <h3>证据列表</h3>
                <span>{{ evidenceItems.length }}</span>
              </div>
              <div class="evidence-grid">
                <div v-for="item in evidenceItems" :key="item._key" class="evidence-card">
                  <div class="evidence-card__top">
                    <strong>{{ item.tool_name || item.source_agent || 'unknown' }}</strong>
                    <span class="status-dot" :class="statusClass(item.status)">{{ item.status || 'unknown' }}</span>
                  </div>
                  <p>{{ item.observed || item.content || '无观测内容' }}</p>
                  <dl class="mini-grid">
                    <div>
                      <dt>目标</dt>
                      <dd>{{ item.target || '无' }}</dd>
                    </div>
                    <div>
                      <dt>置信度</dt>
                      <dd>{{ confidenceText(item.confidence) }}</dd>
                    </div>
                    <div>
                      <dt>支持假设</dt>
                      <dd>{{ booleanText(item.supports_hypothesis) }}</dd>
                    </div>
                  </dl>
                </div>
                <p v-if="evidenceItems.length === 0" class="empty-text">暂无证据</p>
              </div>
            </article>

          </section>

          <section v-if="activeTab === 'collaboration'" class="tab-panel">
            <article class="panel">
              <div class="panel__header">
                <h3>Agent 协作链路</h3>
                <span>{{ communicationChains.length }}</span>
              </div>
              <div class="collaboration-summary-grid">
                <div class="metric metric--compact">
                  <span>链路</span>
                  <strong>{{ communicationChains.length }}</strong>
                </div>
                <div class="metric metric--compact">
                  <span>假设</span>
                  <strong>{{ communicationStats.hypothesis }}</strong>
                </div>
                <div class="metric metric--compact">
                  <span>证据请求</span>
                  <strong>{{ communicationStats.requests }}</strong>
                </div>
                <div class="metric metric--compact">
                  <span>证据回复</span>
                  <strong>{{ communicationStats.responses }}</strong>
                </div>
              </div>
            </article>

            <article v-for="(chain, chainIndex) in communicationChains" :key="chain.chain_id" class="panel">
              <div class="panel__header">
                <h3>{{ chain.title }}</h3>
                <span>链路 {{ chainIndex + 1 }}</span>
              </div>
              <dl class="chain-meta">
                <div>
                  <dt>关联 ID</dt>
                  <dd><code>{{ chain.chain_id }}</code></dd>
                </div>
                <div>
                  <dt>消息数量</dt>
                  <dd>{{ chain.messages.length }}</dd>
                </div>
                <div>
                  <dt>参与 Agent</dt>
                  <dd>{{ listText(chain.agents) }}</dd>
                </div>
              </dl>
              <div class="communication-line">
                <div
                  v-for="(message, messageIndex) in chain.messages"
                  :key="message._key"
                  class="message-card"
                  :class="messageTypeClass(message.msg_type)"
                >
                  <div class="message-card__marker">{{ messageIndex + 1 }}</div>
                  <div class="message-card__body">
                    <div class="message-card__top">
                      <span class="status-pill" :class="messageTypeClass(message.msg_type)">
                        {{ messageTypeLabel(message.msg_type) }}
                      </span>
                      <strong>{{ message.sender || 'unknown' }} -> {{ message.receiver || 'broadcast' }}</strong>
                      <small>{{ formatDate(message.timestamp) }}</small>
                    </div>
                    <p class="message-card__content">{{ message.content || message.hypothesis || '无消息内容' }}</p>
                    <dl class="field-grid field-grid--compact">
                      <div>
                        <dt>消息 ID</dt>
                        <dd><code>{{ message.message_id || message._key }}</code></dd>
                      </div>
                      <div>
                        <dt>回应对象</dt>
                        <dd><code>{{ message.related_to || '无' }}</code></dd>
                      </div>
                      <div>
                        <dt>故障类型</dt>
                        <dd>{{ message.fault_type || '无' }}</dd>
                      </div>
                      <div>
                        <dt>请求证据</dt>
                        <dd>{{ listText(message.required_evidence) }}</dd>
                      </div>
                      <div>
                        <dt>建议工具</dt>
                        <dd>{{ listText(message.suggested_tools) }}</dd>
                      </div>
                      <div>
                        <dt>支持假设</dt>
                        <dd>{{ booleanText(message.supports_hypothesis) }}</dd>
                      </div>
                    </dl>
                    <div v-if="message.evidence?.length" class="message-evidence-list">
                      <div v-for="item in message.evidence" :key="item._key" class="message-evidence">
                        <strong>{{ item.tool_name || item.source_agent || 'evidence' }}</strong>
                        <span class="status-dot" :class="statusClass(item.status)">{{ item.status || 'unknown' }}</span>
                        <p>{{ item.observed || item.content || '无观测内容' }}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </article>

            <article v-if="communicationChains.length === 0" class="panel">
              <p class="empty-text">暂无 Agent 协作消息</p>
            </article>
          </section>

          <section v-if="activeTab === 'fix'" class="tab-panel">
            <article class="panel">
              <div class="panel__header">
                <h3>修复方案</h3>
                <span>{{ fixPlan.risk_level || 'unknown' }}</span>
              </div>
              <dl class="field-grid">
                <div>
                  <dt>方案 ID</dt>
                  <dd>{{ fixPlan.plan_id || '无' }}</dd>
                </div>
                <div>
                  <dt>说明</dt>
                  <dd>{{ fixPlan.description || '无' }}</dd>
                </div>
                <div>
                  <dt>预计耗时</dt>
                  <dd>{{ fixPlan.estimated_time || '无' }}</dd>
                </div>
                <div>
                  <dt>前置条件</dt>
                  <dd>{{ listText(fixPlan.prerequisites) }}</dd>
                </div>
              </dl>
            </article>

            <article class="panel">
              <div class="panel__header">
                <h3>修复步骤</h3>
                <span>{{ fixSteps.length }}</span>
              </div>
              <div class="step-list">
                <div v-for="step in fixSteps" :key="step.step_id || step.action" class="step-card">
                  <div class="step-card__index">{{ step.step_id || '-' }}</div>
                  <div class="step-card__body">
                    <div class="step-card__top">
                      <h4>{{ step.action || '未命名步骤' }}</h4>
                      <span class="status-pill" :class="riskClass(step.risk_level)">{{ step.risk_level || 'unknown' }}</span>
                    </div>
                    <dl class="field-grid field-grid--compact">
                      <div>
                        <dt>动作类型</dt>
                        <dd>{{ step.action_type || '无' }}</dd>
                      </div>
                      <div>
                        <dt>目标</dt>
                        <dd>{{ step.target || '无' }}</dd>
                      </div>
                      <div>
                        <dt>命令</dt>
                        <dd><code>{{ step.command || '无' }}</code></dd>
                      </div>
                      <div>
                        <dt>预期输出</dt>
                        <dd>{{ step.expected_output || '无' }}</dd>
                      </div>
                      <div>
                        <dt>失败处理</dt>
                        <dd>{{ step.on_failure || '无' }}</dd>
                      </div>
                      <div>
                        <dt>回滚</dt>
                        <dd>{{ step.rollback_action_type || step.rollback_command || '无' }}</dd>
                      </div>
                    </dl>
                    <JsonTree :value="step.parameters || {}" label="parameters" />
                  </div>
                </div>
                <p v-if="fixSteps.length === 0" class="empty-text">暂无修复步骤</p>
              </div>
            </article>

            <article class="panel">
              <div class="panel__header">
                <h3>验证与完整 JSON</h3>
              </div>
              <JsonTree :value="fixPlan" label="fix_plan" />
            </article>
          </section>

          <section v-if="activeTab === 'flow'" class="tab-panel">
            <article class="panel">
              <div class="panel__header">
                <h3>Agent 汇总</h3>
                <span>{{ agentNames.length }}</span>
              </div>
              <div class="agent-grid">
                <div v-for="name in agentNames" :key="name" class="agent-card">
                  <strong>{{ name }}</strong>
                  <span>{{ flowAgentSummary[name].actions?.length || 0 }} 次操作</span>
                  <small>轮次：{{ listText(flowAgentSummary[name].dispatch_rounds) }}</small>
                </div>
                <p v-if="agentNames.length === 0" class="empty-text">暂无 Agent 汇总</p>
              </div>
            </article>

            <article class="panel">
              <div class="panel__header">
                <h3>审计流程</h3>
                <span>{{ flowSteps.length }}</span>
              </div>
              <div class="timeline">
                <div v-for="(step, index) in flowSteps" :key="`${step.agent_name}-${index}`" class="timeline-item">
                  <div class="timeline-item__marker">{{ index + 1 }}</div>
                  <div class="timeline-item__body">
                    <div class="timeline-item__top">
                      <strong>{{ step.agent_name }}</strong>
                      <span>{{ step.action_type }}</span>
                      <small>{{ formatDate(step.timestamp) }}</small>
                    </div>
                    <JsonTree :value="step" :label="`flow_steps[${index}]`" />
                  </div>
                </div>
                <p v-if="flowSteps.length === 0" class="empty-text">暂无审计流程</p>
              </div>
            </article>
          </section>

          <section v-if="activeTab === 'trace'" class="tab-panel">
            <article class="panel">
              <div class="panel__header">
                <h3>标准 Trace</h3>
                <span>{{ traceEvents.length }}</span>
              </div>
              <div class="trace-list">
                <div v-for="(event, index) in traceEvents" :key="`${event.event_type}-${index}`" class="trace-card">
                  <div class="trace-card__top">
                    <strong>{{ traceEventLabel(event.event_type) }}</strong>
                    <span class="status-dot" :class="statusClass(event.status)">{{ event.status }}</span>
                    <small>{{ event.agent_name }}</small>
                  </div>
                  <JsonTree :value="event" :label="`standard_trace[${index}]`" />
                </div>
                <p v-if="traceEvents.length === 0" class="empty-text">暂无 Trace</p>
              </div>
            </article>
          </section>

          <section v-if="activeTab === 'raw'" class="tab-panel">
            <article class="panel">
              <div class="panel__header">
                <h3>工单完整 JSON</h3>
              </div>
              <JsonTree :value="selectedTicket" label="ticket" />
            </article>
            <article class="panel">
              <div class="panel__header">
                <h3>流程完整 JSON</h3>
              </div>
              <JsonTree :value="agentFlow || {}" label="agent_flow" />
            </article>
            <article class="panel">
              <div class="panel__header">
                <h3>限流器</h3>
              </div>
              <JsonTree :value="rateStats || {}" label="rate_limiter" />
            </article>
          </section>
        </template>
      </section>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import JsonTree from './components/JsonTree.vue'

// API_BASE_URL：前端访问后端业务接口的基础路径
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'
// HEALTH_URL：健康检查接口路径
const HEALTH_URL = import.meta.env.VITE_HEALTH_URL || '/health'
// MAX_COLLECTED_ITEMS：递归提取复杂字段时的最大数量，避免异常大对象拖慢页面
const MAX_COLLECTED_ITEMS = 200
// statusOptions：工单状态筛选项
const statusOptions = [
  { value: 'pending', label: '待处理' },
  { value: 'approved', label: '已批准' },
  { value: 'rejected', label: '已拒绝' },
  { value: 'completed', label: '已完成' },
]
// detailTabs：详情区域页签
const detailTabs = [
  { value: 'process', label: '流程总览' },
  { value: 'overview', label: '诊断证据' },
  { value: 'collaboration', label: '协作链' },
  { value: 'fix', label: '修复审批' },
  { value: 'flow', label: '审计流程' },
  { value: 'trace', label: 'Trace' },
  { value: 'raw', label: '原始 JSON' },
]
// statusLabels：状态码到中文文案的映射
const statusLabels = {
  pending: '待处理',
  pending_approval: '待审批',
  approved: '已批准',
  rejected: '已拒绝',
  completed: '已完成',
  success: '成功',
  failure: '失败',
  failed: '失败',
  skipped: '跳过',
  ok: '正常',
  unknown: '未知',
}
// urgencyLabels：紧急程度到中文文案的映射
const urgencyLabels = {
  low: '低',
  medium: '中',
  high: '高',
  critical: '紧急',
}
// diagnosisTypeLabels：诊断类型到中文文案的映射
const diagnosisTypeLabels = {
  app: '应用',
  db: '数据库',
  net: '网络',
  other: '其他',
}
// traceEventLabels：标准 Trace 事件到中文文案的映射
const traceEventLabels = {
  agent_started: 'Agent 开始',
  tool_called: '工具调用',
  observation_received: '观测结果',
  diagnosis_generated: '诊断生成',
  handoff_requested: '协作请求',
  plan_generated: '方案生成',
  policy_checked: '策略检查',
  approval_received: '审批接收',
  action_executed: '动作执行',
  verification_passed: '验证通过',
}
// messageTypeLabels：Agent 通信消息类型到中文文案的映射
const messageTypeLabels = {
  hypothesis: '提出假设',
  evidence_request: '请求证据',
  evidence_response: '回复证据',
  challenge: '质疑',
  support: '支持',
  diagnosis: '诊断结论',
  info: '普通消息',
}

// tickets：工单列表数据
const tickets = ref([])
// selectedTicketId：当前选中的工单 ID
const selectedTicketId = ref('')
// selectedTicket：当前工单详情
const selectedTicket = ref(null)
// agentFlow：当前工单的 Agent 流程详情
const agentFlow = ref(null)
// health：后端健康检查结果
const health = ref(null)
// rateStats：限流器统计信息
const rateStats = ref(null)
// activeTab：详情区域当前页签
const activeTab = ref('process')
// loading：页面各异步动作的加载状态
const loading = reactive({
  tickets: false,
  detail: false,
  flow: false,
  submit: false,
  approval: false,
  stats: false,
})
// errors：页面各异步动作的错误信息
const errors = reactive({
  tickets: '',
  detail: '',
  flow: '',
  submit: '',
  approval: '',
  stats: '',
})
// filters：工单列表筛选条件
const filters = reactive({
  keyword: '',
  status: 'all',
})
// createForm：创建工单表单数据
const createForm = reactive({
  ticket_id: '',
  symptom: '',
})
// approvalForm：审批表单数据
const approvalForm = reactive({
  approved: true,
  comments: '',
})

// filteredTickets：按关键字和状态过滤后的工单列表
const filteredTickets = computed(() => {
  // keyword：统一小写后的搜索关键字
  const keyword = filters.keyword.toLowerCase()
  return tickets.value.filter((ticket) => {
    // matchesStatus：状态筛选是否命中
    const matchesStatus = filters.status === 'all' || ticket.status === filters.status
    // searchable：参与搜索的主要字段，包含症状和诊断类型
    const searchable = `${ticket.ticket_id || ''} ${ticket.symptom || ''} ${ticket.diagnosis_type || ''}`.toLowerCase()
    // matchesKeyword：关键字筛选是否命中
    const matchesKeyword = !keyword || searchable.includes(keyword)
    return matchesStatus && matchesKeyword
  })
})

// fixPlan：当前工单修复方案，缺省为空对象
const fixPlan = computed(() => selectedTicket.value?.fix_plan || {})
// fixSteps：当前工单修复步骤列表
const fixSteps = computed(() => Array.isArray(fixPlan.value.steps) ? fixPlan.value.steps : [])
// flowSteps：Agent 审计流程步骤列表
const flowSteps = computed(() => Array.isArray(agentFlow.value?.flow_steps) ? agentFlow.value.flow_steps : [])
// flowAgentSummary：Agent 汇总结构
const flowAgentSummary = computed(() => agentFlow.value?.agent_summary || {})
// agentNames：Agent 名称列表
const agentNames = computed(() => Object.keys(flowAgentSummary.value))
// traceEvents：标准 Trace 事件列表，优先使用 Agent Flow 接口返回的数据
const traceEvents = computed(() => {
  // flowTrace：Agent Flow 接口中的标准 Trace
  const flowTrace = agentFlow.value?.standard_trace
  // ticketTrace：工单 execution_result 中持久化的标准 Trace
  const ticketTrace = selectedTicket.value?.execution_result?.trace_events
  return Array.isArray(flowTrace) && flowTrace.length > 0 ? flowTrace : Array.isArray(ticketTrace) ? ticketTrace : []
})
// communicationMessages：从工单详情、Agent 流程和 Trace 中归并出的 Agent 通信消息
const communicationMessages = computed(() => collectCommunicationMessages())
// communicationChains：按 correlation_id 聚合出的协作链路
const communicationChains = computed(() => {
  // groups：链路 ID 到消息链的映射
  const groups = new Map()
  // 遍历通信消息：优先按 correlation_id 聚合，其次用 hypothesis_id/related_to/message_id 兜底
  for (const message of communicationMessages.value) {
    // chainId：当前消息所属链路 ID
    const chainId = message.correlation_id || message.hypothesis_id || message.related_to || message.message_id || message._key
    // 链路初始化判断：首次出现时创建链路容器
    if (!groups.has(chainId)) {
      groups.set(chainId, {
        chain_id: chainId,
        messages: [],
      })
    }
    groups.get(chainId).messages.push(message)
  }

  return Array.from(groups.values()).map((chain) => {
    // sortedMessages：同一链路内按 Trace 顺序和时间排序后的消息
    const sortedMessages = [...chain.messages].sort(compareCommunicationMessages)
    // agents：参与该链路的 Agent 名称集合
    const agents = Array.from(new Set(sortedMessages.flatMap((message) => [message.sender, message.receiver]).filter(Boolean)))
    return {
      ...chain,
      agents,
      messages: sortedMessages,
      title: buildChainTitle(sortedMessages),
    }
  }).sort((left, right) => compareCommunicationMessages(left.messages[0], right.messages[0]))
})
// communicationStats：协作链路的摘要统计
const communicationStats = computed(() => {
  // stats：按消息类型聚合后的计数
  const stats = {
    hypothesis: 0,
    requests: 0,
    responses: 0,
  }
  // 遍历消息：统计假设、证据请求和证据回复
  for (const message of communicationMessages.value) {
    // 假设消息判断：用于展示链路起点数量
    if (message.msg_type === 'hypothesis') {
      stats.hypothesis += 1
    }
    // 证据请求判断：用于展示跨 Agent 求证次数
    if (message.msg_type === 'evidence_request') {
      stats.requests += 1
    }
    // 证据回复判断：用于展示跨 Agent 反馈次数
    if (message.msg_type === 'evidence_response') {
      stats.responses += 1
    }
  }
  return stats
})
// evidenceItems：从工单复杂结构中递归收集出的证据列表
const evidenceItems = computed(() => {
  // collectedGroups：递归找到的 evidence 字段集合
  const collectedGroups = collectByKey(selectedTicket.value, 'evidence', MAX_COLLECTED_ITEMS)
  // flattenedItems：拉平后的证据列表
  const flattenedItems = []
  // 遍历证据集合：兼容 evidence 为数组或单对象的情况
  for (const group of collectedGroups) {
    // 数组证据处理：逐条追加并保留稳定 key
    if (Array.isArray(group)) {
      for (const item of group) {
        flattenedItems.push(normalizeEvidenceItem(item, flattenedItems.length))
      }
    } else if (group && typeof group === 'object') {
      flattenedItems.push(normalizeEvidenceItem(group, flattenedItems.length))
    }
  }
  return flattenedItems
})
// diagnosisFields：诊断总览字段
const diagnosisFields = computed(() => {
  // diagnosis：当前诊断结果对象
  const diagnosis = selectedTicket.value?.diagnosis_result || {}
  return [
    { key: 'diagnosis_type', label: '诊断类型', value: diagnosisTypeLabel(selectedTicket.value?.diagnosis_type) },
    { key: 'urgency', label: '紧急程度', value: urgencyLabel(selectedTicket.value?.urgency) },
    { key: 'diagnosis', label: '诊断结论', value: diagnosis.diagnosis || '无' },
    { key: 'fault_type', label: '故障类型', value: diagnosis.fault_type || '无' },
    { key: 'confidence', label: '置信度', value: confidenceText(diagnosis.confidence) },
    { key: 'hypothesis', label: '假设', value: diagnosis.hypothesis || '无' },
    { key: 'possible_causes', label: '可能原因', value: listText(diagnosis.possible_causes) },
  ]
})
// canApproveSelectedTicket：当前工单是否可以直接审批
const canApproveSelectedTicket = computed(() => {
  // approvalStatus：后端保存的审批状态
  const approvalStatus = selectedTicket.value?.approval_status
  // ticketStatus：后端保存的工单状态
  const ticketStatus = selectedTicket.value?.status
  return Boolean(fixSteps.value.length > 0 && approvalStatus === 'pending' && ['pending', 'pending_approval'].includes(ticketStatus))
})
// primaryDiagnosisText：流程总览里展示的主诊断结论
const primaryDiagnosisText = computed(() => {
  // diagnosis：当前诊断结果对象
  const diagnosis = selectedTicket.value?.diagnosis_result || {}
  return diagnosis.diagnosis || diagnosis.reasoning || '诊断尚未生成'
})
// primaryFaultType：流程总览里展示的故障类型
const primaryFaultType = computed(() => {
  // diagnosis：当前诊断结果对象
  const diagnosis = selectedTicket.value?.diagnosis_result || {}
  return diagnosis.fault_type || selectedTicket.value?.diagnosis_type || '未知'
})
// primaryConfidence：流程总览里展示的置信度
const primaryConfidence = computed(() => {
  // diagnosis：当前诊断结果对象
  const diagnosis = selectedTicket.value?.diagnosis_result || {}
  return confidenceText(diagnosis.confidence)
})
// verificationSummary：恢复验证摘要
const verificationSummary = computed(() => {
  // executionResult：当前工单执行结果
  const executionResult = selectedTicket.value?.execution_result || {}
  // probes：恢复验证探针列表
  const probes = executionResult.probes || executionResult.verification_probes || []
  // verified：恢复验证是否通过
  const verified = executionResult.verified
  if (typeof verified === 'boolean') {
    return verified ? '验证通过' : '验证未通过'
  }
  if (Array.isArray(probes) && probes.length > 0) {
    return `${probes.filter((probe) => probe.success).length}/${probes.length} 个探针通过`
  }
  return '暂无验证'
})
// currentPhase：当前工单所在主阶段
const currentPhase = computed(() => {
  // ticketStatus：工单状态
  const ticketStatus = selectedTicket.value?.status
  // approvalStatus：审批状态
  const approvalStatus = selectedTicket.value?.approval_status
  if (ticketStatus === 'completed') {
    return { label: '已完成', className: 'status-ok' }
  }
  if (ticketStatus === 'rejected' || approvalStatus === 'rejected') {
    return { label: '已拒绝', className: 'status-failed' }
  }
  if (canApproveSelectedTicket.value) {
    return { label: '待审批', className: 'status-pending' }
  }
  if (ticketStatus === 'approved' || approvalStatus === 'approved') {
    return { label: '执行中', className: 'status-pending' }
  }
  if (fixSteps.value.length > 0) {
    return { label: '方案已生成', className: 'status-pending' }
  }
  if (selectedTicket.value?.diagnosis_result) {
    return { label: '已诊断', className: 'status-ok' }
  }
  return { label: '已提交', className: 'status-muted' }
})
// nextActionText：当前工单下一步动作说明
const nextActionText = computed(() => {
  // ticketStatus：工单状态
  const ticketStatus = selectedTicket.value?.status
  // approvalStatus：审批状态
  const approvalStatus = selectedTicket.value?.approval_status
  if (canApproveSelectedTicket.value) {
    return '审批修复方案'
  }
  if (ticketStatus === 'completed') {
    return '查看执行与验证结果'
  }
  if (ticketStatus === 'rejected' || approvalStatus === 'rejected') {
    return '查看拒绝原因'
  }
  if (ticketStatus === 'approved' || approvalStatus === 'approved') {
    return '等待执行和验证'
  }
  if (!fixSteps.value.length) {
    return '等待修复方案'
  }
  return '查看详情'
})
// actionPanelText：不可审批时动作面板展示文本
const actionPanelText = computed(() => {
  // ticketStatus：工单状态
  const ticketStatus = selectedTicket.value?.status
  if (ticketStatus === 'completed') {
    return '工单已完成，可查看执行与验证详情。'
  }
  if (ticketStatus === 'rejected') {
    return selectedTicket.value?.approver_comments || '工单已被拒绝。'
  }
  if (selectedTicket.value?.approval_status === 'approved') {
    return '方案已批准，等待执行链路更新。'
  }
  return '当前阶段暂不需要人工动作。'
})
// primaryCommunicationChain：流程总览里展示的第一条关键协作链
const primaryCommunicationChain = computed(() => communicationChains.value[0] || null)
// processStages：主流程阶段列表
const processStages = computed(() => buildProcessStages())
// completedStageCount：已经完成的阶段数量
const completedStageCount = computed(() => {
  return processStages.value.filter((stage) => stage.className === 'stage-done').length
})
// healthText：健康检查展示文本
const healthText = computed(() => health.value?.status === 'ok' ? '后端正常' : '后端未连接')
// healthClass：健康检查状态样式
const healthClass = computed(() => health.value?.status === 'ok' ? 'status-ok' : 'status-failed')

/**
 * 组件挂载后加载首页数据。
 *
 * 参数说明：
 * - 无
 *
 * 返回值说明：
 * - Promise<void>
 *
 * 异常说明：
 * - 单个接口异常会写入 errors，不阻断其他接口
 */
onMounted(async () => {
  await refreshAll()
})

/**
 * 拼接 API 请求地址。
 *
 * 参数说明：
 * - path: 接口路径
 *
 * 返回值说明：
 * - 完整请求路径
 *
 * 异常说明：
 * - 无
 */
function buildApiUrl(path) {
  // 前缀判断：完整 URL 直接返回，便于后续接入外部网关
  if (path.startsWith('http')) {
    return path
  }
  return `${API_BASE_URL}${path}`
}

/**
 * 解析后端统一响应。
 *
 * 参数说明：
 * - response: fetch 返回对象
 *
 * 返回值说明：
 * - Promise<any>，返回 data 字段或原始 JSON
 *
 * 异常说明：
 * - HTTP 错误或业务码错误时抛出 Error
 */
async function unwrapResponse(response) {
  // payload：后端返回的 JSON 数据
  const payload = await response.json()
  // HTTP 状态判断：非 2xx 直接抛错
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || '请求失败')
  }
  // 业务状态判断：兼容 FastAPI 统一 APIResponse
  if (payload.code && payload.code !== 200) {
    throw new Error(payload.message || '请求失败')
  }
  return Object.prototype.hasOwnProperty.call(payload, 'data') ? payload.data : payload
}

/**
 * 发起 GET 请求。
 *
 * 参数说明：
 * - path: 接口路径
 *
 * 返回值说明：
 * - Promise<any>
 *
 * 异常说明：
 * - 网络错误或响应错误会向上抛出
 */
async function apiGet(path) {
  // response：浏览器 fetch 响应对象
  const response = await fetch(buildApiUrl(path))
  return unwrapResponse(response)
}

/**
 * 发起 POST 请求。
 *
 * 参数说明：
 * - path: 接口路径
 * - body: 请求体对象
 *
 * 返回值说明：
 * - Promise<any>
 *
 * 异常说明：
 * - 网络错误或响应错误会向上抛出
 */
async function apiPost(path, body) {
  // response：浏览器 fetch 响应对象
  const response = await fetch(buildApiUrl(path), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })
  return unwrapResponse(response)
}

/**
 * 刷新所有顶部与列表数据。
 *
 * 参数说明：
 * - 无
 *
 * 返回值说明：
 * - Promise<void>
 *
 * 异常说明：
 * - 子请求各自处理错误
 */
async function refreshAll() {
  await Promise.all([loadHealth(), loadRateStats(), loadTickets()])
  // 已选工单判断：刷新列表后同步刷新详情和流程
  if (selectedTicketId.value) {
    await Promise.all([loadTicketDetail(selectedTicketId.value), loadAgentFlow(selectedTicketId.value)])
  }
}

/**
 * 加载后端健康状态。
 *
 * 参数说明：
 * - 无
 *
 * 返回值说明：
 * - Promise<void>
 *
 * 异常说明：
 * - 请求失败时 health 置空
 */
async function loadHealth() {
  try {
    // response：健康检查原始响应
    const response = await fetch(HEALTH_URL)
    health.value = await response.json()
  } catch (error) {
    health.value = null
  }
}

/**
 * 加载限流器统计。
 *
 * 参数说明：
 * - 无
 *
 * 返回值说明：
 * - Promise<void>
 *
 * 异常说明：
 * - 请求失败时写入 errors.stats
 */
async function loadRateStats() {
  loading.stats = true
  errors.stats = ''
  try {
    rateStats.value = await apiGet('/rate-limiter/stats')
  } catch (error) {
    errors.stats = error.message
    rateStats.value = null
  } finally {
    loading.stats = false
  }
}

/**
 * 加载工单列表。
 *
 * 参数说明：
 * - 无
 *
 * 返回值说明：
 * - Promise<void>
 *
 * 异常说明：
 * - 请求失败时写入 errors.tickets
 */
async function loadTickets() {
  loading.tickets = true
  errors.tickets = ''
  try {
    // data：列表接口返回数据，兼容数组与分页对象
    const data = await apiGet('/tickets')
    tickets.value = Array.isArray(data) ? data : data.items || []
    // 默认选择判断：首次加载时自动选中第一张工单
    if (!selectedTicketId.value && tickets.value.length > 0) {
      await selectTicket(tickets.value[0].ticket_id)
    }
  } catch (error) {
    errors.tickets = error.message
  } finally {
    loading.tickets = false
  }
}

/**
 * 选择工单并加载详情。
 *
 * 参数说明：
 * - ticketId: 工单 ID
 *
 * 返回值说明：
 * - Promise<void>
 *
 * 异常说明：
 * - 子请求各自处理错误
 */
async function selectTicket(ticketId) {
  selectedTicketId.value = ticketId
  activeTab.value = 'process'
  await Promise.all([loadTicketDetail(ticketId), loadAgentFlow(ticketId)])
}

/**
 * 加载工单详情。
 *
 * 参数说明：
 * - ticketId: 工单 ID
 *
 * 返回值说明：
 * - Promise<void>
 *
 * 异常说明：
 * - 请求失败时写入 errors.detail
 */
async function loadTicketDetail(ticketId) {
  loading.detail = true
  errors.detail = ''
  try {
    selectedTicket.value = await apiGet(`/tickets/${encodeURIComponent(ticketId)}`)
  } catch (error) {
    errors.detail = error.message
  } finally {
    loading.detail = false
  }
}

/**
 * 加载工单 Agent 流程。
 *
 * 参数说明：
 * - ticketId: 工单 ID
 *
 * 返回值说明：
 * - Promise<void>
 *
 * 异常说明：
 * - 请求失败时写入 errors.flow
 */
async function loadAgentFlow(ticketId) {
  loading.flow = true
  errors.flow = ''
  try {
    agentFlow.value = await apiGet(`/tickets/${encodeURIComponent(ticketId)}/agent-flow`)
  } catch (error) {
    errors.flow = error.message
    agentFlow.value = null
  } finally {
    loading.flow = false
  }
}

/**
 * 提交新工单。
 *
 * 参数说明：
 * - 无
 *
 * 返回值说明：
 * - Promise<void>
 *
 * 异常说明：
 * - 请求失败时写入 errors.submit
 */
async function submitTicket() {
  loading.submit = true
  errors.submit = ''
  try {
    await apiPost('/tickets', {
      ticket_id: createForm.ticket_id,
      symptom: createForm.symptom,
    })
    // createdTicketId：提交成功后的工单 ID，用于自动选中
    const createdTicketId = createForm.ticket_id
    createForm.ticket_id = ''
    createForm.symptom = ''
    await loadTickets()
    await selectTicket(createdTicketId)
  } catch (error) {
    errors.submit = error.message
  } finally {
    loading.submit = false
  }
}

/**
 * 提交当前工单审批。
 *
 * 参数说明：
 * - 无
 *
 * 返回值说明：
 * - Promise<void>
 *
 * 异常说明：
 * - 请求失败时写入 errors.approval
 */
async function approveSelectedTicket() {
  // 选中工单判断：没有工单时不发起审批
  if (!selectedTicketId.value) {
    return
  }
  loading.approval = true
  errors.approval = ''
  try {
    await apiPost(`/tickets/${encodeURIComponent(selectedTicketId.value)}/approve`, {
      approved: approvalForm.approved,
      comments: approvalForm.comments,
    })
    approvalForm.comments = ''
    await Promise.all([loadTickets(), loadTicketDetail(selectedTicketId.value), loadAgentFlow(selectedTicketId.value)])
  } catch (error) {
    errors.approval = error.message
  } finally {
    loading.approval = false
  }
}

/**
 * 构建主流程阶段列表。
 *
 * 参数说明：
 * - 无
 *
 * 返回值说明：
 * - 主流程阶段数组
 *
 * 异常说明：
 * - 无
 */
function buildProcessStages() {
  // ticketStatus：工单状态
  const ticketStatus = selectedTicket.value?.status
  // approvalStatus：审批状态
  const approvalStatus = selectedTicket.value?.approval_status
  // hasDiagnosis：是否已有诊断结果
  const hasDiagnosis = Boolean(selectedTicket.value?.diagnosis_result)
  // hasCollaboration：是否已有 Agent 协作消息
  const hasCollaboration = communicationMessages.value.length > 0
  // hasFixPlan：是否已有修复步骤
  const hasFixPlan = fixSteps.value.length > 0
  // isRejected：是否已拒绝
  const isRejected = ticketStatus === 'rejected' || approvalStatus === 'rejected'
  // isApproved：是否已批准
  const isApproved = ticketStatus === 'approved' || approvalStatus === 'approved' || ticketStatus === 'completed'
  // isCompleted：是否已完成
  const isCompleted = ticketStatus === 'completed'

  return [
    {
      key: 'submitted',
      index: 1,
      title: '接收工单',
      summary: selectedTicket.value?.ticket_id || '暂无工单',
      meta: formatDate(selectedTicket.value?.created_at),
      className: 'stage-done',
    },
    {
      key: 'diagnosis',
      index: 2,
      title: 'Agent 诊断',
      summary: hasDiagnosis ? primaryDiagnosisText.value : '等待 Agent 输出诊断',
      meta: `${diagnosisTypeLabel(selectedTicket.value?.diagnosis_type)} / ${urgencyLabel(selectedTicket.value?.urgency)}`,
      className: hasDiagnosis ? 'stage-done' : 'stage-active',
    },
    {
      key: 'collaboration',
      index: 3,
      title: '证据协作',
      summary: hasCollaboration ? `${communicationMessages.value.length} 条通信消息` : '无跨 Agent 证据请求',
      meta: hasCollaboration ? `${communicationChains.value.length} 条链路` : '按需触发',
      className: hasCollaboration ? 'stage-done' : hasDiagnosis ? 'stage-muted' : 'stage-waiting',
    },
    {
      key: 'fix',
      index: 4,
      title: '生成方案',
      summary: hasFixPlan ? (fixPlan.value.description || `${fixSteps.value.length} 个步骤`) : '等待修复方案',
      meta: hasFixPlan ? `风险：${fixPlan.value.risk_level || '未知'}` : 'Fix Agent',
      className: hasFixPlan ? 'stage-done' : hasDiagnosis ? 'stage-active' : 'stage-waiting',
    },
    {
      key: 'approval',
      index: 5,
      title: '人工审批',
      summary: approvalLabel(approvalStatus),
      meta: selectedTicket.value?.approver_comments || '安全门禁',
      className: isRejected ? 'stage-error' : isApproved ? 'stage-done' : canApproveSelectedTicket.value ? 'stage-active' : 'stage-waiting',
    },
    {
      key: 'execution',
      index: 6,
      title: '执行验证',
      summary: verificationSummary.value,
      meta: selectedTicket.value?.updated_at ? formatDate(selectedTicket.value.updated_at) : '等待执行器',
      className: isCompleted ? 'stage-done' : isApproved ? 'stage-active' : isRejected ? 'stage-muted' : 'stage-waiting',
    },
  ]
}

/**
 * 收集并归并 Agent 通信消息。
 *
 * 参数说明：
 * - 无
 *
 * 返回值说明：
 * - Agent 通信消息列表
 *
 * 异常说明：
 * - 无
 */
function collectCommunicationMessages() {
  // messageMap：按 message_id 去重后的消息映射
  const messageMap = new Map()
  // order：消息出现顺序，用于没有时间戳时排序
  let order = 0

  /**
   * 追加候选消息。
   *
   * 参数说明：
   * - candidate: 候选消息对象
   * - context: 候选消息所在上下文
   *
   * 返回值说明：
   * - 无
   *
   * 异常说明：
   * - 无
   */
  function addCandidate(candidate, context = {}) {
    // 通信形态判断：没有消息身份或通信字段时跳过
    if (!isCommunicationMessageCandidate(candidate)) {
      return
    }
    // message：规范化后的通信消息
    const message = normalizeCommunicationMessage(candidate, context, order)
    order += 1
    mergeCommunicationMessage(messageMap, message)
  }

  /**
   * 追加消息组。
   *
   * 参数说明：
   * - group: 消息数组或单个消息对象
   * - context: 消息来源上下文
   *
   * 返回值说明：
   * - 无
   *
   * 异常说明：
   * - 无
   */
  function addGroup(group, context = {}) {
    // 数组判断：agent_messages 通常是数组，需要逐条处理
    if (Array.isArray(group)) {
      for (const item of group) {
        addCandidate(item, context)
      }
      return
    }
    addCandidate(group, context)
  }

  // ticketMessageGroups：工单详情中可能持久化的 agent_messages 字段
  const ticketMessageGroups = collectByKey(selectedTicket.value, 'agent_messages', MAX_COLLECTED_ITEMS)
  // 工单消息遍历：优先保留业务状态里的原始通信记录
  for (const group of ticketMessageGroups) {
    addGroup(group, { source: 'ticket_agent_messages' })
  }

  // flowMessageGroups：Agent Flow 中可能包含的 agent_messages 字段
  const flowMessageGroups = collectByKey(agentFlow.value, 'agent_messages', MAX_COLLECTED_ITEMS)
  // 流程消息遍历：兼容后端未来扩展的流程消息字段
  for (const group of flowMessageGroups) {
    addGroup(group, { source: 'flow_agent_messages' })
  }

  // Trace 遍历：从标准 Trace 的 output/metadata 恢复协作请求与回复
  traceEvents.value.forEach((event, index) => {
    addCandidate(event.output, { source: 'trace_output', event, index })
    addCandidate(event.metadata, { source: 'trace_metadata', event, index })
  })

  // flowSteps 遍历：兜底扫描审计步骤里嵌套的 agent_messages
  flowSteps.value.forEach((step, index) => {
    // stepMessageGroups：单个审计步骤里可能嵌套的消息组
    const stepMessageGroups = collectByKey(step, 'agent_messages', MAX_COLLECTED_ITEMS)
    for (const group of stepMessageGroups) {
      addGroup(group, { source: 'flow_step_agent_messages', index })
    }
  })

  return Array.from(messageMap.values()).sort(compareCommunicationMessages)
}

/**
 * 判断对象是否像 Agent 通信消息。
 *
 * 参数说明：
 * - value: 待判断对象
 *
 * 返回值说明：
 * - true 表示可以作为通信消息展示
 *
 * 异常说明：
 * - 无
 */
function isCommunicationMessageCandidate(value) {
  // 类型判断：只处理对象
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }
  // hasIdentity：消息身份字段，至少需要一个才能串联链路
  const hasIdentity = Boolean(value.message_id || value.correlation_id || value.related_to || value.hypothesis_id)
  // hasCommunicationShape：通信语义字段，避免普通对象误入链路视图
  const hasCommunicationShape = Boolean(
    value.sender ||
    value.receiver ||
    value.target_agent ||
    value.msg_type ||
    value.content ||
    value.hypothesis ||
    Array.isArray(value.required_evidence) ||
    Array.isArray(value.suggested_tools) ||
    Array.isArray(value.evidence)
  )
  return hasIdentity && hasCommunicationShape
}

/**
 * 规范化 Agent 通信消息。
 *
 * 参数说明：
 * - rawMessage: 原始消息对象
 * - context: 来源上下文
 * - order: 出现顺序
 *
 * 返回值说明：
 * - 标准化后的前端消息对象
 *
 * 异常说明：
 * - 无
 */
function normalizeCommunicationMessage(rawMessage, context, order) {
  // event：消息所属的 Trace 事件，可能为空
  const event = context.event || {}
  // messageType：消息类型，优先使用原始 msg_type，缺省时从 Trace 事件推断
  const messageType = rawMessage.msg_type || inferMessageType(rawMessage, event)
  // messageId：消息唯一 ID，缺省时用来源和顺序生成稳定展示 key
  const messageId = rawMessage.message_id || `${context.source || 'message'}-${context.index ?? order}`
  // evidence：消息携带的结构化证据列表
  const evidence = Array.isArray(rawMessage.evidence)
    ? rawMessage.evidence.map((item, index) => normalizeEvidenceItem(item, index))
    : []

  return {
    ...rawMessage,
    _key: messageId,
    _order: order,
    _source: context.source || 'unknown',
    message_id: messageId,
    correlation_id: rawMessage.correlation_id || rawMessage.hypothesis_id || rawMessage.related_to || messageId,
    related_to: rawMessage.related_to || '',
    hypothesis_id: rawMessage.hypothesis_id || '',
    status: rawMessage.status || event.status || 'unknown',
    sender: rawMessage.sender || event.agent_name || rawMessage.source_agent || 'unknown',
    receiver: rawMessage.receiver || rawMessage.target_agent || event.metadata?.target_agent || 'broadcast',
    content: rawMessage.content || inferCommunicationContent(rawMessage, event),
    msg_type: messageType,
    timestamp: rawMessage.timestamp || event.timestamp || rawMessage.created_at || '',
    evidence,
    required_evidence: normalizeStringList(rawMessage.required_evidence || event.metadata?.required_evidence),
    suggested_tools: normalizeStringList(rawMessage.suggested_tools || event.metadata?.suggested_tools),
  }
}

/**
 * 合并重复通信消息。
 *
 * 参数说明：
 * - messageMap: 消息映射
 * - message: 待合并消息
 *
 * 返回值说明：
 * - 无
 *
 * 异常说明：
 * - 无
 */
function mergeCommunicationMessage(messageMap, message) {
  // key：去重使用的消息 ID
  const key = message.message_id || message._key
  // 首次出现判断：直接写入映射
  if (!messageMap.has(key)) {
    messageMap.set(key, {
      ...message,
      _sources: [message._source],
    })
    return
  }

  // existing：已存在的消息对象
  const existing = messageMap.get(key)
  // 字段合并：只用新消息补齐旧消息缺失的字段
  for (const [field, value] of Object.entries(message)) {
    // 元字段判断：来源和顺序单独处理
    if (field === '_source' || field === '_order') {
      continue
    }
    // 数组字段判断：旧数组为空时使用新数组
    if (Array.isArray(value)) {
      if (!Array.isArray(existing[field]) || existing[field].length === 0) {
        existing[field] = value
      }
      continue
    }
    // 缺失值判断：旧值为空时用新值补齐
    if (isEmptyValue(existing[field]) && !isEmptyValue(value)) {
      existing[field] = value
    }
  }
  existing._order = Math.min(existing._order, message._order)
  // 来源合并：保留消息来自哪些结构，方便调试原始 JSON
  if (!existing._sources.includes(message._source)) {
    existing._sources.push(message._source)
  }
}

/**
 * 判断值是否为空。
 *
 * 参数说明：
 * - value: 待判断值
 *
 * 返回值说明：
 * - true 表示值为空
 *
 * 异常说明：
 * - 无
 */
function isEmptyValue(value) {
  // 数组判断：空数组视为空值
  if (Array.isArray(value)) {
    return value.length === 0
  }
  return value === undefined || value === null || value === ''
}

/**
 * 推断通信消息类型。
 *
 * 参数说明：
 * - message: 原始消息对象
 * - event: Trace 事件对象
 *
 * 返回值说明：
 * - 消息类型字符串
 *
 * 异常说明：
 * - 无
 */
function inferMessageType(message, event) {
  // 请求字段判断：包含 required_evidence 的消息通常是证据请求
  if (Array.isArray(message.required_evidence) && message.required_evidence.length > 0) {
    return 'evidence_request'
  }
  // 回复字段判断：携带 evidence 且声明 supports_hypothesis 的消息通常是证据回复
  if (Array.isArray(message.evidence) && message.evidence.length > 0 && message.supports_hypothesis !== undefined) {
    return 'evidence_response'
  }
  // Trace 事件判断：handoff_requested 映射为证据请求
  if (event.event_type === 'handoff_requested') {
    return 'evidence_request'
  }
  // Trace 事件判断：observation_received 映射为证据回复
  if (event.event_type === 'observation_received') {
    return 'evidence_response'
  }
  // Trace 事件判断：diagnosis_generated 映射为诊断结论
  if (event.event_type === 'diagnosis_generated') {
    return 'diagnosis'
  }
  return 'info'
}

/**
 * 推断通信消息内容。
 *
 * 参数说明：
 * - message: 原始消息对象
 * - event: Trace 事件对象
 *
 * 返回值说明：
 * - 可读消息内容
 *
 * 异常说明：
 * - 无
 */
function inferCommunicationContent(message, event) {
  // 假设字段判断：优先展示可验证假设
  if (message.hypothesis) {
    return message.hypothesis
  }
  // 证据请求判断：把请求证据字段拼成一句可读文本
  if (Array.isArray(message.required_evidence) && message.required_evidence.length > 0) {
    return `请求证据：${message.required_evidence.join('、')}`
  }
  // Trace 类型判断：用事件类型作为兜底摘要
  if (event.event_type) {
    return traceEventLabel(event.event_type)
  }
  return '无消息内容'
}

/**
 * 规范化字符串列表。
 *
 * 参数说明：
 * - value: 字符串、数组或空值
 *
 * 返回值说明：
 * - 字符串数组
 *
 * 异常说明：
 * - 无
 */
function normalizeStringList(value) {
  // 数组判断：过滤空值并统一转成字符串
  if (Array.isArray(value)) {
    return value.filter((item) => item !== null && item !== undefined && item !== '').map((item) => String(item))
  }
  // 字符串判断：单值包装为数组
  if (typeof value === 'string' && value) {
    return [value]
  }
  return []
}

/**
 * 比较通信消息顺序。
 *
 * 参数说明：
 * - left: 左侧消息
 * - right: 右侧消息
 *
 * 返回值说明：
 * - 排序比较值
 *
 * 异常说明：
 * - 无
 */
function compareCommunicationMessages(left, right) {
  // leftTime：左侧消息时间戳毫秒数
  const leftTime = Date.parse(left?.timestamp || '')
  // rightTime：右侧消息时间戳毫秒数
  const rightTime = Date.parse(right?.timestamp || '')
  // 时间有效判断：两边都有时间时按时间排序
  if (!Number.isNaN(leftTime) && !Number.isNaN(rightTime) && leftTime !== rightTime) {
    return leftTime - rightTime
  }
  return (left?._order || 0) - (right?._order || 0)
}

/**
 * 生成协作链路标题。
 *
 * 参数说明：
 * - messages: 同一链路下的消息列表
 *
 * 返回值说明：
 * - 链路标题文本
 *
 * 异常说明：
 * - 无
 */
function buildChainTitle(messages) {
  // titleSource：优先使用假设，其次使用请求内容，再兜底使用第一条消息内容
  const titleSource =
    messages.find((message) => message.msg_type === 'hypothesis')?.hypothesis ||
    messages.find((message) => message.msg_type === 'evidence_request')?.content ||
    messages[0]?.content ||
    'Agent 协作链路'
  return titleSource.length > 48 ? `${titleSource.slice(0, 48)}...` : titleSource
}

/**
 * 递归收集指定字段。
 *
 * 参数说明：
 * - source: 待搜索对象
 * - keyName: 目标字段名
 * - maxItems: 最大收集数量
 *
 * 返回值说明：
 * - 匹配字段值列表
 *
 * 异常说明：
 * - 无
 */
function collectByKey(source, keyName, maxItems) {
  // results：收集到的字段值
  const results = []
  // visited：已访问对象集合，防止循环引用
  const visited = new WeakSet()

  /**
   * 深度优先遍历对象。
   *
   * 参数说明：
   * - node: 当前节点
   *
   * 返回值说明：
   * - 无
   *
   * 异常说明：
   * - 无
   */
  function walk(node) {
    // 数量判断：达到上限后停止递归
    if (results.length >= maxItems) {
      return
    }
    // 节点类型判断：只遍历对象和数组
    if (!node || typeof node !== 'object') {
      return
    }
    // 循环引用判断：已访问对象不重复处理
    if (visited.has(node)) {
      return
    }
    visited.add(node)

    // 数组遍历：继续深入每个元素
    if (Array.isArray(node)) {
      for (const item of node) {
        walk(item)
      }
      return
    }

    // 对象遍历：匹配目标字段并继续深入字段值
    for (const [key, value] of Object.entries(node)) {
      // 字段名判断：命中目标字段时收集字段值
      if (key === keyName) {
        results.push(value)
      }
      walk(value)
    }
  }

  walk(source)
  return results
}

/**
 * 规范化证据条目。
 *
 * 参数说明：
 * - item: 原始证据条目
 * - index: 当前序号
 *
 * 返回值说明：
 * - 带稳定 key 的证据对象
 *
 * 异常说明：
 * - 无
 */
function normalizeEvidenceItem(item, index) {
  // 基础类型判断：字符串证据包装成对象展示
  if (!item || typeof item !== 'object') {
    return {
      _key: `evidence-${index}`,
      observed: String(item || ''),
      status: 'unknown',
    }
  }
  return {
    _key: item.evidence_id || `evidence-${index}`,
    ...item,
  }
}

/**
 * 格式化日期。
 *
 * 参数说明：
 * - value: 日期字符串
 *
 * 返回值说明：
 * - 本地化日期文本
 *
 * 异常说明：
 * - 日期解析失败时返回原值
 */
function formatDate(value) {
  // 空值判断：无时间时展示占位
  if (!value) {
    return '无时间'
  }
  // date：浏览器解析后的日期对象
  const date = new Date(value)
  // 日期有效性判断：无效日期返回原始文本
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleString('zh-CN', { hour12: false })
}

/**
 * 格式化列表文本。
 *
 * 参数说明：
 * - value: 数组或任意值
 *
 * 返回值说明：
 * - 可读文本
 *
 * 异常说明：
 * - 无
 */
function listText(value) {
  // 数组判断：数组按顿号拼接
  if (Array.isArray(value)) {
    return value.length > 0 ? value.join('、') : '无'
  }
  return value || '无'
}

/**
 * 格式化置信度。
 *
 * 参数说明：
 * - value: 置信度数值
 *
 * 返回值说明：
 * - 百分比文本
 *
 * 异常说明：
 * - 无
 */
function confidenceText(value) {
  // 数值判断：非数值展示占位
  if (typeof value !== 'number') {
    return '无'
  }
  return `${Math.round(value * 100)}%`
}

/**
 * 格式化布尔值。
 *
 * 参数说明：
 * - value: 布尔值或空值
 *
 * 返回值说明：
 * - 中文文本
 *
 * 异常说明：
 * - 无
 */
function booleanText(value) {
  // true 判断：明确支持
  if (value === true) {
    return '是'
  }
  // false 判断：明确不支持
  if (value === false) {
    return '否'
  }
  return '未知'
}

/**
 * 获取状态中文文案。
 *
 * 参数说明：
 * - value: 状态值
 *
 * 返回值说明：
 * - 中文状态文本
 *
 * 异常说明：
 * - 无
 */
function statusLabel(value) {
  return statusLabels[value] || value || '未知'
}

/**
 * 获取审批中文文案。
 *
 * 参数说明：
 * - value: 审批状态
 *
 * 返回值说明：
 * - 中文审批文本
 *
 * 异常说明：
 * - 无
 */
function approvalLabel(value) {
  return statusLabel(value || 'pending')
}

/**
 * 获取紧急程度中文文案。
 *
 * 参数说明：
 * - value: 紧急程度
 *
 * 返回值说明：
 * - 中文紧急程度文本
 *
 * 异常说明：
 * - 无
 */
function urgencyLabel(value) {
  return urgencyLabels[value] || value || '未知'
}

/**
 * 获取诊断类型中文文案。
 *
 * 参数说明：
 * - value: 诊断类型
 *
 * 返回值说明：
 * - 中文诊断类型文本
 *
 * 异常说明：
 * - 无
 */
function diagnosisTypeLabel(value) {
  return diagnosisTypeLabels[value] || value || '未知'
}

/**
 * 获取 Trace 事件中文文案。
 *
 * 参数说明：
 * - value: Trace 事件类型
 *
 * 返回值说明：
 * - 中文事件文本
 *
 * 异常说明：
 * - 无
 */
function traceEventLabel(value) {
  return traceEventLabels[value] || value || '未知事件'
}

/**
 * 获取通信消息类型中文文案。
 *
 * 参数说明：
 * - value: 消息类型
 *
 * 返回值说明：
 * - 中文消息类型文本
 *
 * 异常说明：
 * - 无
 */
function messageTypeLabel(value) {
  return messageTypeLabels[value] || value || '未知消息'
}

/**
 * 获取通信消息类型样式类。
 *
 * 参数说明：
 * - value: 消息类型
 *
 * 返回值说明：
 * - CSS 类名
 *
 * 异常说明：
 * - 无
 */
function messageTypeClass(value) {
  // 假设判断：链路起点使用蓝绿色
  if (value === 'hypothesis') {
    return 'message-hypothesis'
  }
  // 证据请求判断：请求使用黄色提示
  if (value === 'evidence_request') {
    return 'message-request'
  }
  // 证据回复判断：回复使用绿色提示
  if (value === 'evidence_response') {
    return 'message-response'
  }
  // 质疑判断：质疑使用红色提示
  if (value === 'challenge') {
    return 'message-challenge'
  }
  return 'message-info'
}

/**
 * 获取状态样式类。
 *
 * 参数说明：
 * - value: 状态值
 *
 * 返回值说明：
 * - CSS 类名
 *
 * 异常说明：
 * - 无
 */
function statusClass(value) {
  // 成功状态判断：绿色强调
  if (['completed', 'approved', 'success', 'ok'].includes(value)) {
    return 'status-ok'
  }
  // 失败状态判断：红色强调
  if (['rejected', 'failure', 'failed'].includes(value)) {
    return 'status-failed'
  }
  // 跳过状态判断：灰色弱化
  if (value === 'skipped') {
    return 'status-muted'
  }
  return 'status-pending'
}

/**
 * 获取紧急程度样式类。
 *
 * 参数说明：
 * - value: 紧急程度
 *
 * 返回值说明：
 * - CSS 类名
 *
 * 异常说明：
 * - 无
 */
function urgencyClass(value) {
  // 高风险判断：高和紧急使用红色提示
  if (['high', 'critical'].includes(value)) {
    return 'status-failed'
  }
  // 中风险判断：中等使用黄色提示
  if (value === 'medium') {
    return 'status-pending'
  }
  return 'status-ok'
}

/**
 * 获取风险等级样式类。
 *
 * 参数说明：
 * - value: 风险等级
 *
 * 返回值说明：
 * - CSS 类名
 *
 * 异常说明：
 * - 无
 */
function riskClass(value) {
  // 高风险判断：使用红色提示
  if (value === 'high') {
    return 'status-failed'
  }
  // 中风险判断：使用黄色提示
  if (value === 'medium') {
    return 'status-pending'
  }
  return 'status-ok'
}
</script>
