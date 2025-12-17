// User Types
export interface User {
  id: string
  email: string
  name: string
  phone?: string
  role_id: number
  role?: Role
  is_active: boolean
  created_at: string
  updated_at: string
  last_login?: string
}

export interface Role {
  id: number
  name: string
  description?: string
  permissions: string[]
}

export interface Token {
  access_token: string
  refresh_token: string
  token_type: string
}

// Student Types
export interface Student {
  id: string
  user_id?: string
  student_number: string
  status: StudentStatus
  enrolled_at: string
  created_at: string
  updated_at: string
  user_name?: string
  user_email?: string
  user_phone?: string
}

export type StudentStatus = '재학' | '휴학' | '수료' | '중퇴'

// Course Types
export interface Course {
  id: string
  code: string
  name: string
  description?: string
  teacher_id?: string
  teacher_name?: string
  start_date?: string
  end_date?: string
  total_sessions: number
  created_at: string
  updated_at: string
}

export interface Class {
  id: string
  course_id: string
  course_name?: string
  name: string
  capacity: number
  schedule?: Record<string, unknown>
  student_count: number
  created_at: string
  updated_at: string
}

export interface Enrollment {
  id: string
  student_id: string
  class_id: string
  enrolled_at: string
  dropped_at?: string
  student_name?: string
  student_number?: string
  class_name?: string
}

// Attendance Types
export type AttendanceStatus = '출석' | '지각' | '조퇴' | '결석' | '공결'

export interface Attendance {
  id: string
  student_id: string
  class_id: string
  session_no: number
  date: string
  status: AttendanceStatus
  check_in_time?: string
  check_out_time?: string
  notes?: string
  created_at: string
  updated_at: string
  student_name?: string
  student_number?: string
}

export interface AttendanceSummary {
  student_id: string
  student_name: string
  student_number: string
  total_sessions: number
  present_count: number
  late_count: number
  early_leave_count: number
  absent_count: number
  excused_count: number
  attendance_rate: number
}

// Grade Types
export type EvaluationType = '중간고사' | '기말고사' | '퀴즈' | '과제' | '프로젝트' | '출결' | '실기'

export interface Evaluation {
  id: string
  course_id: string
  course_name?: string
  name: string
  type: EvaluationType
  description?: string
  weight: number
  max_score: number
  due_date?: string
  created_at: string
  updated_at: string
}

export interface Grade {
  id: string
  evaluation_id: string
  student_id: string
  score?: number
  comments?: string
  graded_by?: string
  graded_at?: string
  created_at: string
  updated_at: string
  evaluation_name?: string
  evaluation_type?: EvaluationType
  max_score?: number
  weight?: number
  student_name?: string
  student_number?: string
}

export interface StudentGradeSummary {
  student_id: string
  student_name: string
  student_number: string
  course_id: string
  course_name: string
  grades: Grade[]
  total_weighted_score: number
  letter_grade: string
  rank?: number
}
