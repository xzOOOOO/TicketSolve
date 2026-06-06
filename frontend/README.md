# AI 工单前端

Vue 3 + Vite 前端工作台，用于展示工单列表、工单详情、诊断结果、修复方案、审批结果、执行结果、Agent 审计流和标准 Trace。

## 启动

```bash
cd frontend
npm install
npm run dev
```

默认访问地址：

- 前端：http://127.0.0.1:5173
- 后端：http://localhost:8000

## 接口

前端默认通过 Vite 代理访问：

- `GET /api/tickets`
- `GET /api/tickets/{ticket_id}`
- `GET /api/tickets/{ticket_id}/agent-flow`
- `POST /api/tickets`
- `POST /api/tickets/{ticket_id}/approve`
- `GET /api/rate-limiter/stats`
- `GET /health`
