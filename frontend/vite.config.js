import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 后端服务地址：本地开发时把前端请求转发到 FastAPI
const backendTarget = 'http://localhost:8000'

/**
 * Vite 配置。
 *
 * 参数说明：
 * - 无
 *
 * 返回值说明：
 * - Vite 配置对象
 *
 * 异常说明：
 * - 无
 */
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/health': {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
})
