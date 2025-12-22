import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { coursesApi } from '../services/api'
import type { Course } from '../types'
import { Plus, Search, Edit, Trash2, X, Users, ChevronRight } from 'lucide-react'

interface ClassData {
  id: string
  name: string
  capacity: number
  schedule: string
  student_count?: number
}

export default function Courses() {
  const queryClient = useQueryClient()
  const [search, setSearch] = useState('')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [editingCourse, setEditingCourse] = useState<Course | null>(null)
  const [selectedCourse, setSelectedCourse] = useState<Course | null>(null)
  const [showClassModal, setShowClassModal] = useState(false)

  const { data: courses, isLoading } = useQuery({
    queryKey: ['courses', search],
    queryFn: () => coursesApi.getAll({ search }),
  })

  const { data: classes } = useQuery({
    queryKey: ['classes', selectedCourse?.id],
    queryFn: () => coursesApi.getClasses(selectedCourse!.id),
    enabled: !!selectedCourse,
  })

  const createMutation = useMutation({
    mutationFn: coursesApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['courses'] })
      setShowCreateModal(false)
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: Record<string, unknown> }) =>
      coursesApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['courses'] })
      setEditingCourse(null)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: coursesApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['courses'] })
    },
  })

  const createClassMutation = useMutation({
    mutationFn: ({ courseId, data }: { courseId: string; data: Record<string, unknown> }) =>
      coursesApi.createClass(courseId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['classes', selectedCourse?.id] })
      setShowClassModal(false)
    },
  })

  const handleCreateSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const formData = new FormData(e.currentTarget)
    const data = {
      code: formData.get('code'),
      name: formData.get('name'),
      description: formData.get('description'),
      total_sessions: parseInt(formData.get('total_sessions') as string) || 0,
      start_date: formData.get('start_date') || null,
      end_date: formData.get('end_date') || null,
    }
    createMutation.mutate(data)
  }

  const handleEditSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!editingCourse) return
    const formData = new FormData(e.currentTarget)
    const data = {
      code: formData.get('code'),
      name: formData.get('name'),
      description: formData.get('description'),
      total_sessions: parseInt(formData.get('total_sessions') as string) || 0,
      start_date: formData.get('start_date') || null,
      end_date: formData.get('end_date') || null,
    }
    updateMutation.mutate({ id: editingCourse.id, data })
  }

  const handleClassSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!selectedCourse) return
    const formData = new FormData(e.currentTarget)
    const data = {
      name: formData.get('name'),
      capacity: parseInt(formData.get('capacity') as string) || 30,
      schedule: formData.get('schedule'),
    }
    createClassMutation.mutate({ courseId: selectedCourse.id, data })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">과정 관리</h1>
          <p className="text-gray-400">과정 및 반을 관리합니다.</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-500 rounded-lg transition-colors"
        >
          <Plus className="w-5 h-5" />
          과정 등록
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
            placeholder="과정명 또는 코드로 검색..."
            className="w-full pl-12 pr-4 py-3 bg-background border border-border rounded-lg"
          />
        </div>
      </div>

      {/* Course grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {isLoading ? (
          <div className="col-span-full text-center py-8 text-gray-400">
            로딩 중...
          </div>
        ) : courses?.length === 0 ? (
          <div className="col-span-full text-center py-8 text-gray-400">
            등록된 과정이 없습니다.
          </div>
        ) : (
          courses?.map((course: Course) => (
            <div
              key={course.id}
              className="bg-surface rounded-xl border border-border overflow-hidden"
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <span className="px-2 py-1 bg-primary/20 text-primary rounded text-xs font-mono">
                      {course.code}
                    </span>
                    <h3 className="text-lg font-semibold mt-2">{course.name}</h3>
                  </div>
                  <div className="flex gap-1">
                    <button
                      onClick={() => setEditingCourse(course)}
                      className="p-2 hover:bg-background rounded-lg transition-colors"
                    >
                      <Edit className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => {
                        if (confirm('정말 삭제하시겠습니까?')) {
                          deleteMutation.mutate(course.id)
                        }
                      }}
                      className="p-2 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
                <p className="text-sm text-gray-400 mb-4 line-clamp-2">
                  {course.description || '설명 없음'}
                </p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">
                    담당: {course.teacher_name || '미정'}
                  </span>
                  <span className="text-gray-400">
                    총 {course.total_sessions}회차
                  </span>
                </div>
              </div>
              <div className="px-6 py-4 bg-background border-t border-border">
                <div className="flex items-center justify-between">
                  <button
                    onClick={() => setSelectedCourse(course)}
                    className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
                  >
                    <Users className="w-4 h-4" />
                    <span>반 관리</span>
                  </button>
                  <button
                    onClick={() => setSelectedCourse(course)}
                    className="flex items-center gap-1 text-sm text-primary hover:underline"
                  >
                    상세 보기
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-surface rounded-xl border border-border w-full max-w-md mx-4">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h2 className="text-lg font-semibold">과정 등록</h2>
              <button onClick={() => setShowCreateModal(false)}>
                <X className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={handleCreateSubmit} className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">과정 코드 *</label>
                <input
                  name="code"
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  placeholder="PY2024-01"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">과정명 *</label>
                <input
                  name="name"
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  placeholder="Python 심화과정"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">설명</label>
                <textarea
                  name="description"
                  rows={3}
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  placeholder="과정 설명..."
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">총 회차</label>
                <input
                  name="total_sessions"
                  type="number"
                  min="0"
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  placeholder="20"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">시작일</label>
                  <input
                    name="start_date"
                    type="date"
                    className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">종료일</label>
                  <input
                    name="end_date"
                    type="date"
                    className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  />
                </div>
              </div>
              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
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
      {editingCourse && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-surface rounded-xl border border-border w-full max-w-md mx-4">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h2 className="text-lg font-semibold">과정 수정</h2>
              <button onClick={() => setEditingCourse(null)}>
                <X className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={handleEditSubmit} className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">과정 코드 *</label>
                <input
                  name="code"
                  required
                  defaultValue={editingCourse.code}
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">과정명 *</label>
                <input
                  name="name"
                  required
                  defaultValue={editingCourse.name}
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">설명</label>
                <textarea
                  name="description"
                  rows={3}
                  defaultValue={editingCourse.description || ''}
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">총 회차</label>
                <input
                  name="total_sessions"
                  type="number"
                  min="0"
                  defaultValue={editingCourse.total_sessions}
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">시작일</label>
                  <input
                    name="start_date"
                    type="date"
                    defaultValue={editingCourse.start_date || ''}
                    className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">종료일</label>
                  <input
                    name="end_date"
                    type="date"
                    defaultValue={editingCourse.end_date || ''}
                    className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  />
                </div>
              </div>
              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setEditingCourse(null)}
                  className="flex-1 py-2 bg-background border border-border rounded-lg hover:bg-border transition-colors"
                >
                  취소
                </button>
                <button
                  type="submit"
                  disabled={updateMutation.isPending}
                  className="flex-1 py-2 bg-primary hover:bg-primary-500 rounded-lg transition-colors disabled:opacity-50"
                >
                  {updateMutation.isPending ? '수정 중...' : '수정'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Course Detail / Class Management Modal */}
      {selectedCourse && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-surface rounded-xl border border-border w-full max-w-2xl mx-4 max-h-[80vh] overflow-hidden flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <div>
                <span className="px-2 py-1 bg-primary/20 text-primary rounded text-xs font-mono">
                  {selectedCourse.code}
                </span>
                <h2 className="text-lg font-semibold mt-1">{selectedCourse.name}</h2>
              </div>
              <button onClick={() => setSelectedCourse(null)}>
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-4 overflow-y-auto flex-1">
              {/* Course Info */}
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-gray-400 mb-2">과정 정보</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">총 회차:</span>
                    <span className="ml-2">{selectedCourse.total_sessions}회</span>
                  </div>
                  <div>
                    <span className="text-gray-400">담당:</span>
                    <span className="ml-2">{selectedCourse.teacher_name || '미정'}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">시작일:</span>
                    <span className="ml-2">{selectedCourse.start_date || '-'}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">종료일:</span>
                    <span className="ml-2">{selectedCourse.end_date || '-'}</span>
                  </div>
                </div>
                {selectedCourse.description && (
                  <p className="mt-3 text-sm text-gray-400">{selectedCourse.description}</p>
                )}
              </div>

              {/* Classes */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-semibold text-gray-400">반 목록</h3>
                  <button
                    onClick={() => setShowClassModal(true)}
                    className="flex items-center gap-1 px-3 py-1 bg-primary/20 text-primary rounded-lg text-sm hover:bg-primary/30 transition-colors"
                  >
                    <Plus className="w-4 h-4" />
                    반 추가
                  </button>
                </div>

                {classes?.length === 0 ? (
                  <div className="text-center py-8 text-gray-400 bg-background rounded-lg">
                    등록된 반이 없습니다.
                  </div>
                ) : (
                  <div className="space-y-2">
                    {classes?.map((cls: ClassData) => (
                      <div
                        key={cls.id}
                        className="flex items-center justify-between p-4 bg-background rounded-lg"
                      >
                        <div>
                          <h4 className="font-medium">{cls.name}</h4>
                          <p className="text-sm text-gray-400">
                            정원: {cls.capacity}명 | {cls.schedule || '일정 미정'}
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-gray-400">
                            {cls.student_count || 0}명 등록
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Create Class Modal */}
      {showClassModal && selectedCourse && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50">
          <div className="bg-surface rounded-xl border border-border w-full max-w-md mx-4">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h2 className="text-lg font-semibold">반 추가</h2>
              <button onClick={() => setShowClassModal(false)}>
                <X className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={handleClassSubmit} className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">반 이름 *</label>
                <input
                  name="name"
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  placeholder="1반"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">정원</label>
                <input
                  name="capacity"
                  type="number"
                  min="1"
                  defaultValue={30}
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">일정</label>
                <input
                  name="schedule"
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  placeholder="월/수/금 09:00-12:00"
                />
              </div>
              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowClassModal(false)}
                  className="flex-1 py-2 bg-background border border-border rounded-lg hover:bg-border transition-colors"
                >
                  취소
                </button>
                <button
                  type="submit"
                  disabled={createClassMutation.isPending}
                  className="flex-1 py-2 bg-primary hover:bg-primary-500 rounded-lg transition-colors disabled:opacity-50"
                >
                  {createClassMutation.isPending ? '추가 중...' : '추가'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
