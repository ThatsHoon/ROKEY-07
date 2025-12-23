import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { studentsApi, coursesApi } from '../services/api'
import type { Student, Course, Class, StudentStatus } from '../types'
import { Plus, Search, Edit, Trash2, X, UserPlus, BookOpen } from 'lucide-react'

const STATUS_OPTIONS: StudentStatus[] = ['재학', '휴학', '수료', '중퇴']

export default function Students() {
  const queryClient = useQueryClient()
  const [search, setSearch] = useState('')
  const [showModal, setShowModal] = useState(false)
  const [editingStudent, setEditingStudent] = useState<Student | null>(null)
  const [enrollingStudent, setEnrollingStudent] = useState<Student | null>(null)
  const [selectedCourse, setSelectedCourse] = useState<Course | null>(null)

  const { data: students, isLoading } = useQuery({
    queryKey: ['students', search],
    queryFn: () => studentsApi.getAll({ search }),
  })

  const { data: courses } = useQuery({
    queryKey: ['courses'],
    queryFn: () => coursesApi.getAll(),
    enabled: !!enrollingStudent,
  })

  const { data: classes } = useQuery({
    queryKey: ['classes', selectedCourse?.id],
    queryFn: () => coursesApi.getClasses(selectedCourse!.id),
    enabled: !!selectedCourse,
  })

  const createMutation = useMutation({
    mutationFn: studentsApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['students'] })
      setShowModal(false)
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: Record<string, unknown> }) =>
      studentsApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['students'] })
      setEditingStudent(null)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: studentsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['students'] })
    },
  })

  const enrollMutation = useMutation({
    mutationFn: ({ studentId, classId }: { studentId: string; classId: string }) =>
      studentsApi.enroll(studentId, classId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['students'] })
      queryClient.invalidateQueries({ queryKey: ['classes'] })
      alert('수강 배정이 완료되었습니다.')
      setEnrollingStudent(null)
      setSelectedCourse(null)
    },
    onError: (error: Error & { response?: { data?: { detail?: string } } }) => {
      const message = error.response?.data?.detail || '수강 배정에 실패했습니다.'
      alert(message)
    },
  })

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const formData = new FormData(e.currentTarget)
    const data = {
      student_number: formData.get('student_number'),
      email: formData.get('email'),
      name: formData.get('name'),
      phone: formData.get('phone'),
      password: formData.get('password') || 'default123',
    }
    createMutation.mutate(data)
  }

  const handleEditSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!editingStudent) return
    const formData = new FormData(e.currentTarget)
    const data = {
      student_number: formData.get('student_number'),
      status: formData.get('status'),
    }
    updateMutation.mutate({ id: editingStudent.id, data })
  }

  const handleEnroll = (classId: string) => {
    if (!enrollingStudent) return
    enrollMutation.mutate({ studentId: enrollingStudent.id, classId })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">학생 관리</h1>
          <p className="text-gray-400">학생 목록을 조회하고 관리합니다.</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-500 rounded-lg transition-colors"
        >
          <Plus className="w-5 h-5" />
          학생 등록
        </button>
      </div>

      {/* Search */}
      <div className="bg-surface rounded-xl border border-border p-4">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="이름 또는 학번으로 검색..."
            className="w-full pl-12 pr-4 py-3 bg-background border border-border rounded-lg"
          />
        </div>
      </div>

      {/* Student list */}
      <div className="bg-surface rounded-xl border border-border overflow-hidden">
        <table className="w-full">
          <thead className="bg-background">
            <tr>
              <th className="px-6 py-4 text-left text-sm font-semibold">학번 (고유ID)</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">이름</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">이메일</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">연락처</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">상태</th>
              <th className="px-6 py-4 text-right text-sm font-semibold">작업</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {isLoading ? (
              <tr>
                <td colSpan={6} className="px-6 py-8 text-center text-gray-400">
                  로딩 중...
                </td>
              </tr>
            ) : students?.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-6 py-8 text-center text-gray-400">
                  등록된 학생이 없습니다.
                </td>
              </tr>
            ) : (
              students?.map((student: Student) => (
                <tr key={student.id} className="hover:bg-background/50">
                  <td className="px-6 py-4">
                    <span className="px-2 py-1 bg-primary/20 text-primary rounded font-mono text-sm">
                      {student.student_number}
                    </span>
                  </td>
                  <td className="px-6 py-4">{student.user_name || '-'}</td>
                  <td className="px-6 py-4 text-gray-400">{student.user_email || '-'}</td>
                  <td className="px-6 py-4 text-gray-400">{student.user_phone || '-'}</td>
                  <td className="px-6 py-4">
                    <span
                      className={`px-2 py-1 rounded text-xs ${
                        student.status === '재학'
                          ? 'bg-green-500/20 text-green-400'
                          : student.status === '휴학'
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}
                    >
                      {student.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <button
                        onClick={() => setEnrollingStudent(student)}
                        className="p-2 hover:bg-blue-500/20 text-blue-400 rounded-lg transition-colors"
                        title="수강 배정"
                      >
                        <UserPlus className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => setEditingStudent(student)}
                        className="p-2 hover:bg-background rounded-lg transition-colors"
                        title="정보 수정"
                      >
                        <Edit className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => {
                          if (confirm('정말 삭제하시겠습니까?')) {
                            deleteMutation.mutate(student.id)
                          }
                        }}
                        className="p-2 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors"
                        title="삭제"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Create Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-surface rounded-xl border border-border w-full max-w-md mx-4">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h2 className="text-lg font-semibold">학생 등록</h2>
              <button onClick={() => setShowModal(false)}>
                <X className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={handleSubmit} className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">학번 *</label>
                <input
                  name="student_number"
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  placeholder="2024001"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">이름 *</label>
                <input
                  name="name"
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  placeholder="홍길동"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">이메일 *</label>
                <input
                  name="email"
                  type="email"
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  placeholder="student@example.com"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">연락처</label>
                <input
                  name="phone"
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  placeholder="010-1234-5678"
                />
              </div>
              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="flex-1 py-2 bg-background border border-border rounded-lg hover:bg-border transition-colors"
                >
                  취소
                </button>
                <button
                  type="submit"
                  disabled={createMutation.isPending}
                  className="flex-1 py-2 bg-primary hover:bg-primary-500 rounded-lg transition-colors disabled:opacity-50"
                >
                  {createMutation.isPending ? '등록 중...' : '등록'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit Modal */}
      {editingStudent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-surface rounded-xl border border-border w-full max-w-md mx-4">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h2 className="text-lg font-semibold">학생 정보 수정</h2>
              <button onClick={() => setEditingStudent(null)}>
                <X className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={handleEditSubmit} className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">이름</label>
                <input
                  value={editingStudent.user_name || ''}
                  disabled
                  className="w-full px-4 py-2 bg-background/50 border border-border rounded-lg text-gray-400"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">학번 *</label>
                <input
                  name="student_number"
                  defaultValue={editingStudent.student_number}
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">상태 *</label>
                <select
                  name="status"
                  defaultValue={editingStudent.status}
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                >
                  {STATUS_OPTIONS.map((status) => (
                    <option key={status} value={status}>
                      {status}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setEditingStudent(null)}
                  className="flex-1 py-2 bg-background border border-border rounded-lg hover:bg-border transition-colors"
                >
                  취소
                </button>
                <button
                  type="submit"
                  disabled={updateMutation.isPending}
                  className="flex-1 py-2 bg-primary hover:bg-primary-500 rounded-lg transition-colors disabled:opacity-50"
                >
                  {updateMutation.isPending ? '저장 중...' : '저장'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Enrollment Modal */}
      {enrollingStudent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-surface rounded-xl border border-border w-full max-w-lg mx-4">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <div>
                <h2 className="text-lg font-semibold">수강 배정</h2>
                <p className="text-sm text-gray-400">
                  {enrollingStudent.user_name} ({enrollingStudent.student_number})
                </p>
              </div>
              <button onClick={() => { setEnrollingStudent(null); setSelectedCourse(null); }}>
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 space-y-4">
              {/* Course Selection */}
              <div>
                <label className="block text-sm font-medium mb-2">과정 선택</label>
                <select
                  value={selectedCourse?.id || ''}
                  onChange={(e) => {
                    const course = courses?.find((c: Course) => c.id === e.target.value)
                    setSelectedCourse(course || null)
                  }}
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                >
                  <option value="">과정을 선택하세요</option>
                  {courses?.map((course: Course) => (
                    <option key={course.id} value={course.id}>
                      [{course.code}] {course.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Class List */}
              {selectedCourse && (
                <div>
                  <label className="block text-sm font-medium mb-2">반 선택</label>
                  {!classes || classes.length === 0 ? (
                    <p className="text-gray-400 text-sm py-4 text-center">
                      해당 과정에 등록된 반이 없습니다.
                    </p>
                  ) : (
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {classes.map((cls: Class) => (
                        <div
                          key={cls.id}
                          className="flex items-center justify-between p-3 bg-background rounded-lg border border-border"
                        >
                          <div className="flex items-center gap-3">
                            <BookOpen className="w-5 h-5 text-primary" />
                            <div>
                              <p className="font-medium">{cls.name}</p>
                              <p className="text-xs text-gray-400">
                                정원: {cls.student_count}/{cls.capacity}
                              </p>
                            </div>
                          </div>
                          <button
                            onClick={() => handleEnroll(cls.id)}
                            disabled={enrollMutation.isPending || cls.student_count >= cls.capacity}
                            className="px-3 py-1 bg-primary hover:bg-primary-500 rounded text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {enrollMutation.isPending ? '배정 중...' : '배정'}
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div className="pt-4 border-t border-border">
                <button
                  type="button"
                  onClick={() => { setEnrollingStudent(null); setSelectedCourse(null); }}
                  className="w-full py-2 bg-background border border-border rounded-lg hover:bg-border transition-colors"
                >
                  닫기
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
