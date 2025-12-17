import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { studentsApi } from '../services/api'
import type { Student } from '../types'
import { Plus, Search, Edit, Trash2, X } from 'lucide-react'

export default function Students() {
  const queryClient = useQueryClient()
  const [search, setSearch] = useState('')
  const [showModal, setShowModal] = useState(false)
  const [editingStudent, setEditingStudent] = useState<Student | null>(null)

  const { data: students, isLoading } = useQuery({
    queryKey: ['students', search],
    queryFn: () => studentsApi.getAll({ search }),
  })

  const createMutation = useMutation({
    mutationFn: studentsApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['students'] })
      setShowModal(false)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: studentsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['students'] })
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
              <th className="px-6 py-4 text-left text-sm font-semibold">학번</th>
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
                  <td className="px-6 py-4 font-mono">{student.student_number}</td>
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
                        onClick={() => setEditingStudent(student)}
                        className="p-2 hover:bg-background rounded-lg transition-colors"
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
    </div>
  )
}
