import axios from 'axios'
import { useAuthStore } from '../store/authStore'

const API_BASE_URL = '/api/v1'

// 임시 토큰 저장 (로그인 직후 getCurrentUser 호출 전)
let tempAuthToken: string | null = null

export const setAuthToken = (token: string | null) => {
  tempAuthToken = token
}

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for adding auth token
api.interceptors.request.use(
  (config) => {
    // 임시 토큰이 있으면 우선 사용 (로그인 직후)
    if (tempAuthToken) {
      config.headers.Authorization = `Bearer ${tempAuthToken}`
    } else {
      const token = useAuthStore.getState().token
      if (token?.access_token) {
        config.headers.Authorization = `Bearer ${token.access_token}`
      }
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout()
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// Auth API
export const authApi = {
  login: async (email: string, password: string) => {
    const formData = new FormData()
    formData.append('username', email)
    formData.append('password', password)
    const response = await api.post('/auth/login', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },
  getCurrentUser: async () => {
    const response = await api.get('/auth/me')
    return response.data
  },
}

// Users API
export const usersApi = {
  getAll: async (params?: Record<string, unknown>) => {
    const response = await api.get('/users', { params })
    return response.data
  },
  getById: async (id: string) => {
    const response = await api.get(`/users/${id}`)
    return response.data
  },
  create: async (data: Record<string, unknown>) => {
    const response = await api.post('/users', data)
    return response.data
  },
  update: async (id: string, data: Record<string, unknown>) => {
    const response = await api.put(`/users/${id}`, data)
    return response.data
  },
  delete: async (id: string) => {
    await api.delete(`/users/${id}`)
  },
}

// Students API
export const studentsApi = {
  getAll: async (params?: Record<string, unknown>) => {
    const response = await api.get('/students', { params })
    return response.data
  },
  getById: async (id: string) => {
    const response = await api.get(`/students/${id}`)
    return response.data
  },
  create: async (data: Record<string, unknown>) => {
    const response = await api.post('/students', data)
    return response.data
  },
  update: async (id: string, data: Record<string, unknown>) => {
    const response = await api.put(`/students/${id}`, data)
    return response.data
  },
  delete: async (id: string) => {
    await api.delete(`/students/${id}`)
  },
  enroll: async (studentId: string, classId: string) => {
    const response = await api.post(`/students/${studentId}/enroll`, {
      student_id: studentId,
      class_id: classId,
    })
    return response.data
  },
}

// Courses API
export const coursesApi = {
  getAll: async (params?: Record<string, unknown>) => {
    const response = await api.get('/courses', { params })
    return response.data
  },
  getById: async (id: string) => {
    const response = await api.get(`/courses/${id}`)
    return response.data
  },
  create: async (data: Record<string, unknown>) => {
    const response = await api.post('/courses', data)
    return response.data
  },
  update: async (id: string, data: Record<string, unknown>) => {
    const response = await api.put(`/courses/${id}`, data)
    return response.data
  },
  delete: async (id: string) => {
    await api.delete(`/courses/${id}`)
  },
  getClasses: async (courseId: string) => {
    const response = await api.get(`/courses/${courseId}/classes`)
    return response.data
  },
  createClass: async (courseId: string, data: Record<string, unknown>) => {
    const response = await api.post(`/courses/${courseId}/classes`, data)
    return response.data
  },
}

// Attendance API
export const attendanceApi = {
  getAll: async (params?: Record<string, unknown>) => {
    const response = await api.get('/attendance', { params })
    return response.data
  },
  create: async (data: Record<string, unknown>) => {
    const response = await api.post('/attendance', data)
    return response.data
  },
  createBulk: async (data: Record<string, unknown>) => {
    const response = await api.post('/attendance/bulk', data)
    return response.data
  },
  update: async (id: string, data: Record<string, unknown>) => {
    const response = await api.put(`/attendance/${id}`, data)
    return response.data
  },
  getStudentSummary: async (studentId: string, classId: string) => {
    const response = await api.get(`/attendance/student/${studentId}/summary`, {
      params: { class_id: classId },
    })
    return response.data
  },
}

// Grades API
export const gradesApi = {
  getAll: async (params?: Record<string, unknown>) => {
    const response = await api.get('/grades', { params })
    return response.data
  },
  create: async (data: Record<string, unknown>) => {
    const response = await api.post('/grades', data)
    return response.data
  },
  createBulk: async (data: Record<string, unknown>) => {
    const response = await api.post('/grades/bulk', data)
    return response.data
  },
  update: async (id: string, data: Record<string, unknown>) => {
    const response = await api.put(`/grades/${id}`, data)
    return response.data
  },
  getEvaluations: async (params?: Record<string, unknown>) => {
    const response = await api.get('/grades/evaluations', { params })
    return response.data
  },
  createEvaluation: async (data: Record<string, unknown>) => {
    const response = await api.post('/grades/evaluations', data)
    return response.data
  },
  getStudentSummary: async (studentId: string, courseId: string) => {
    const response = await api.get(`/grades/student/${studentId}/summary`, {
      params: { course_id: courseId },
    })
    return response.data
  },
}

// Reports API
export const reportsApi = {
  downloadAttendanceExcel: async (classId: string) => {
    const response = await api.get('/reports/attendance/excel', {
      params: { class_id: classId },
      responseType: 'blob',
    })
    return response.data
  },
  downloadAttendancePdf: async (classId: string) => {
    const response = await api.get('/reports/attendance/pdf', {
      params: { class_id: classId },
      responseType: 'blob',
    })
    return response.data
  },
  downloadGradesExcel: async (courseId: string) => {
    const response = await api.get('/reports/grades/excel', {
      params: { course_id: courseId },
      responseType: 'blob',
    })
    return response.data
  },
  downloadGradesPdf: async (courseId: string) => {
    const response = await api.get('/reports/grades/pdf', {
      params: { course_id: courseId },
      responseType: 'blob',
    })
    return response.data
  },
  downloadTranscript: async (studentId: string) => {
    const response = await api.get(`/reports/transcript/${studentId}`, {
      responseType: 'blob',
    })
    return response.data
  },
}

export default api
