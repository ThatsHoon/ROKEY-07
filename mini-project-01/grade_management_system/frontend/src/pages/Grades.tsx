import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { gradesApi, coursesApi, studentsApi } from '../services/api'
import type { Grade, Evaluation, EvaluationType, Course, Class } from '../types'
import { Filter, Plus, TrendingUp, Edit, X, Save } from 'lucide-react'

const EVALUATION_TYPES: EvaluationType[] = ['중간고사', '기말고사', '퀴즈', '과제', '프로젝트', '출결', '실기']

export default function Grades() {
  const queryClient = useQueryClient()
  const [selectedCourse, setSelectedCourse] = useState<string>('')
  const [showEvalModal, setShowEvalModal] = useState(false)
  const [selectedEvaluation, setSelectedEvaluation] = useState<Evaluation | null>(null)
  const [showGradeModal, setShowGradeModal] = useState(false)
  const [editingGrade, setEditingGrade] = useState<Grade | null>(null)

  const { data: courses } = useQuery({
    queryKey: ['courses'],
    queryFn: () => coursesApi.getAll(),
  })

  const { data: evaluations } = useQuery({
    queryKey: ['evaluations', selectedCourse],
    queryFn: () => gradesApi.getEvaluations({ course_id: selectedCourse }),
    enabled: !!selectedCourse,
  })

  const { data: grades, isLoading } = useQuery({
    queryKey: ['grades', selectedCourse],
    queryFn: () => gradesApi.getAll({ course_id: selectedCourse }),
    enabled: !!selectedCourse,
  })

  const { data: classes } = useQuery({
    queryKey: ['classes', selectedCourse],
    queryFn: () => coursesApi.getClasses(selectedCourse),
    enabled: !!selectedCourse && showGradeModal,
  })

  const { data: courseStudents } = useQuery({
    queryKey: ['courseStudents', selectedCourse, classes],
    queryFn: async () => {
      if (!classes || classes.length === 0) return []
      const allStudents: { student_id: string; student_name: string; student_number: string }[] = []
      for (const cls of classes as Class[]) {
        const students = await coursesApi.getClassStudents(cls.id)
        for (const s of students) {
          if (!allStudents.find(x => x.student_id === s.student_id)) {
            allStudents.push(s)
          }
        }
      }
      return allStudents
    },
    enabled: !!classes && classes.length > 0 && showGradeModal,
  })

  const createEvalMutation = useMutation({
    mutationFn: gradesApi.createEvaluation,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['evaluations'] })
      setShowEvalModal(false)
    },
  })

  const createGradeMutation = useMutation({
    mutationFn: gradesApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['grades'] })
      setShowGradeModal(false)
      setSelectedEvaluation(null)
    },
  })

  const updateGradeMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: Record<string, unknown> }) =>
      gradesApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['grades'] })
      setEditingGrade(null)
    },
  })

  const calculateAverage = (evalId: string) => {
    const evalGrades = grades?.filter((g: Grade) => g.evaluation_id === evalId && g.score !== null) || []
    if (evalGrades.length === 0) return 0
    const sum = evalGrades.reduce((acc: number, g: Grade) => acc + (g.score || 0), 0)
    return (sum / evalGrades.length).toFixed(1)
  }

  const handleEvalSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const formData = new FormData(e.currentTarget)
    const data = {
      course_id: selectedCourse,
      name: formData.get('name'),
      type: formData.get('type') as EvaluationType,
      description: formData.get('description') || null,
      weight: parseInt(formData.get('weight') as string) || 0,
      max_score: parseInt(formData.get('max_score') as string) || 100,
      due_date: formData.get('due_date') || null,
    }
    createEvalMutation.mutate(data)
  }

  const handleGradeSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const formData = new FormData(e.currentTarget)
    const data = {
      evaluation_id: selectedEvaluation?.id,
      student_id: formData.get('student_id'),
      score: parseFloat(formData.get('score') as string) || 0,
      comments: formData.get('comments') || null,
    }
    createGradeMutation.mutate(data)
  }

  const handleGradeEditSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!editingGrade) return
    const formData = new FormData(e.currentTarget)
    const data = {
      score: parseFloat(formData.get('score') as string) || 0,
      comments: formData.get('comments') || null,
    }
    updateGradeMutation.mutate({ id: editingGrade.id, data })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">성적 관리</h1>
          <p className="text-gray-400">평가 항목 및 성적을 관리합니다.</p>
        </div>
        {selectedCourse && (
          <button
            onClick={() => setShowEvalModal(true)}
            className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-500 rounded-lg transition-colors"
          >
            <Plus className="w-5 h-5" />
            평가 항목 추가
          </button>
        )}
      </div>

      {/* Filters */}
      <div className="bg-surface rounded-xl border border-border p-4">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-gray-400" />
            <span className="text-sm font-medium">과정 선택</span>
          </div>
          <select
            value={selectedCourse}
            onChange={(e) => setSelectedCourse(e.target.value)}
            className="px-4 py-2 bg-background border border-border rounded-lg"
          >
            <option value="">과정 선택</option>
            {courses?.map((course: Course) => (
              <option key={course.id} value={course.id}>
                [{course.code}] {course.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {!selectedCourse ? (
        <div className="bg-surface rounded-xl border border-border p-12 text-center text-gray-400">
          과정을 선택해주세요.
        </div>
      ) : (
        <>
          {/* Evaluation items */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {evaluations?.map((evaluation: Evaluation) => (
              <div
                key={evaluation.id}
                className="bg-surface rounded-xl border border-border p-6 hover:border-primary/50 transition-colors cursor-pointer"
                onClick={() => {
                  setSelectedEvaluation(evaluation)
                  setShowGradeModal(true)
                }}
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <span className="px-2 py-1 bg-accent/20 text-accent rounded text-xs">
                      {evaluation.type}
                    </span>
                    <h3 className="text-lg font-semibold mt-2">
                      {evaluation.name}
                    </h3>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-bold text-primary">
                      {evaluation.weight}%
                    </p>
                    <p className="text-xs text-gray-400">가중치</p>
                  </div>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">
                    만점: {evaluation.max_score}점
                  </span>
                  <span className="flex items-center gap-1 text-secondary">
                    <TrendingUp className="w-4 h-4" />
                    평균 {calculateAverage(evaluation.id)}점
                  </span>
                </div>
                {evaluation.due_date && (
                  <p className="mt-2 text-xs text-gray-400">
                    마감: {evaluation.due_date}
                  </p>
                )}
                <p className="mt-2 text-xs text-primary">클릭하여 성적 입력</p>
              </div>
            ))}
            {evaluations?.length === 0 && (
              <div className="col-span-full text-center py-8 text-gray-400">
                등록된 평가 항목이 없습니다. 평가 항목 추가 버튼을 눌러 추가하세요.
              </div>
            )}
          </div>

          {/* Grades table */}
          <div className="bg-surface rounded-xl border border-border overflow-hidden">
            <div className="p-4 border-b border-border">
              <h2 className="text-lg font-semibold">성적 현황</h2>
            </div>
            <table className="w-full">
              <thead className="bg-background">
                <tr>
                  <th className="px-6 py-4 text-left text-sm font-semibold">
                    학번
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold">
                    이름
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold">
                    평가항목
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold">
                    점수
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold">
                    만점
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold">
                    채점일
                  </th>
                  <th className="px-6 py-4 text-right text-sm font-semibold">
                    작업
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {isLoading ? (
                  <tr>
                    <td
                      colSpan={7}
                      className="px-6 py-8 text-center text-gray-400"
                    >
                      로딩 중...
                    </td>
                  </tr>
                ) : grades?.length === 0 ? (
                  <tr>
                    <td
                      colSpan={7}
                      className="px-6 py-8 text-center text-gray-400"
                    >
                      성적 기록이 없습니다. 평가 항목 카드를 클릭하여 성적을 입력하세요.
                    </td>
                  </tr>
                ) : (
                  grades?.map((grade: Grade) => (
                    <tr key={grade.id} className="hover:bg-background/50">
                      <td className="px-6 py-4 font-mono">
                        {grade.student_number}
                      </td>
                      <td className="px-6 py-4">{grade.student_name}</td>
                      <td className="px-6 py-4">{grade.evaluation_name}</td>
                      <td className="px-6 py-4">
                        <span
                          className={`font-semibold ${
                            (grade.score || 0) >= (grade.max_score || 100) * 0.9
                              ? 'text-green-400'
                              : (grade.score || 0) >=
                                (grade.max_score || 100) * 0.6
                              ? 'text-yellow-400'
                              : 'text-red-400'
                          }`}
                        >
                          {grade.score ?? '-'}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-gray-400">
                        {grade.max_score}
                      </td>
                      <td className="px-6 py-4 text-gray-400">
                        {grade.graded_at
                          ? new Date(grade.graded_at).toLocaleDateString()
                          : '-'}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <button
                          onClick={() => setEditingGrade(grade)}
                          className="p-2 hover:bg-background rounded-lg transition-colors"
                          title="수정"
                        >
                          <Edit className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Create Evaluation Modal */}
      {showEvalModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-surface rounded-xl border border-border w-full max-w-md mx-4">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h2 className="text-lg font-semibold">평가 항목 추가</h2>
              <button onClick={() => setShowEvalModal(false)}>
                <X className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={handleEvalSubmit} className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">평가명 *</label>
                <input
                  name="name"
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  placeholder="중간고사"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">유형 *</label>
                <select
                  name="type"
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                >
                  {EVALUATION_TYPES.map((type) => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">가중치 (%) *</label>
                  <input
                    name="weight"
                    type="number"
                    min="0"
                    max="100"
                    required
                    className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                    placeholder="30"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">만점 *</label>
                  <input
                    name="max_score"
                    type="number"
                    min="1"
                    required
                    defaultValue={100}
                    className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">마감일</label>
                <input
                  name="due_date"
                  type="date"
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">설명</label>
                <textarea
                  name="description"
                  rows={2}
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg resize-none"
                  placeholder="평가에 대한 설명"
                />
              </div>
              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowEvalModal(false)}
                  className="flex-1 py-2 bg-background border border-border rounded-lg hover:bg-border transition-colors"
                >
                  취소
                </button>
                <button
                  type="submit"
                  disabled={createEvalMutation.isPending}
                  className="flex-1 py-2 bg-primary hover:bg-primary-500 rounded-lg transition-colors disabled:opacity-50"
                >
                  {createEvalMutation.isPending ? '추가 중...' : '추가'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Grade Input Modal */}
      {showGradeModal && selectedEvaluation && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-surface rounded-xl border border-border w-full max-w-md mx-4">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <div>
                <h2 className="text-lg font-semibold">성적 입력</h2>
                <p className="text-sm text-gray-400">{selectedEvaluation.name} (만점: {selectedEvaluation.max_score}점)</p>
              </div>
              <button onClick={() => { setShowGradeModal(false); setSelectedEvaluation(null); }}>
                <X className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={handleGradeSubmit} className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">학생 선택 *</label>
                <select
                  name="student_id"
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                >
                  <option value="">학생을 선택하세요</option>
                  {courseStudents?.map((student: { student_id: string; student_name: string; student_number: string }) => (
                    <option key={student.student_id} value={student.student_id}>
                      {student.student_name} ({student.student_number})
                    </option>
                  ))}
                </select>
                {(!courseStudents || courseStudents.length === 0) && (
                  <p className="text-xs text-gray-400 mt-1">해당 과정에 등록된 학생이 없습니다.</p>
                )}
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">점수 * (0 ~ {selectedEvaluation.max_score})</label>
                <input
                  name="score"
                  type="number"
                  min="0"
                  max={selectedEvaluation.max_score}
                  step="0.1"
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  placeholder="85"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">코멘트</label>
                <textarea
                  name="comments"
                  rows={2}
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg resize-none"
                  placeholder="성적에 대한 코멘트"
                />
              </div>
              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => { setShowGradeModal(false); setSelectedEvaluation(null); }}
                  className="flex-1 py-2 bg-background border border-border rounded-lg hover:bg-border transition-colors"
                >
                  취소
                </button>
                <button
                  type="submit"
                  disabled={createGradeMutation.isPending}
                  className="flex-1 py-2 bg-primary hover:bg-primary-500 rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  <Save className="w-4 h-4" />
                  {createGradeMutation.isPending ? '저장 중...' : '저장'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit Grade Modal */}
      {editingGrade && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-surface rounded-xl border border-border w-full max-w-md mx-4">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <div>
                <h2 className="text-lg font-semibold">성적 수정</h2>
                <p className="text-sm text-gray-400">
                  {editingGrade.student_name} - {editingGrade.evaluation_name}
                </p>
              </div>
              <button onClick={() => setEditingGrade(null)}>
                <X className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={handleGradeEditSubmit} className="p-4 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">학생</label>
                  <input
                    value={`${editingGrade.student_name} (${editingGrade.student_number})`}
                    disabled
                    className="w-full px-4 py-2 bg-background/50 border border-border rounded-lg text-gray-400"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">평가항목</label>
                  <input
                    value={editingGrade.evaluation_name || ''}
                    disabled
                    className="w-full px-4 py-2 bg-background/50 border border-border rounded-lg text-gray-400"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">점수 * (만점: {editingGrade.max_score})</label>
                <input
                  name="score"
                  type="number"
                  min="0"
                  max={editingGrade.max_score}
                  step="0.1"
                  defaultValue={editingGrade.score || ''}
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">코멘트</label>
                <textarea
                  name="comments"
                  rows={2}
                  defaultValue={editingGrade.comments || ''}
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg resize-none"
                />
              </div>
              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setEditingGrade(null)}
                  className="flex-1 py-2 bg-background border border-border rounded-lg hover:bg-border transition-colors"
                >
                  취소
                </button>
                <button
                  type="submit"
                  disabled={updateGradeMutation.isPending}
                  className="flex-1 py-2 bg-primary hover:bg-primary-500 rounded-lg transition-colors disabled:opacity-50"
                >
                  {updateGradeMutation.isPending ? '저장 중...' : '저장'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
