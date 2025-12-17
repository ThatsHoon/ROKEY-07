import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { gradesApi, coursesApi } from '../services/api'
import type { Grade, Evaluation } from '../types'
import { Filter, Plus, TrendingUp } from 'lucide-react'

export default function Grades() {
  const [selectedCourse, setSelectedCourse] = useState<string>('')

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

  const calculateAverage = (evalId: string) => {
    const evalGrades = grades?.filter((g: Grade) => g.evaluation_id === evalId) || []
    if (evalGrades.length === 0) return 0
    const sum = evalGrades.reduce((acc: number, g: Grade) => acc + (g.score || 0), 0)
    return (sum / evalGrades.length).toFixed(1)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">성적 관리</h1>
          <p className="text-gray-400">평가 항목 및 성적을 관리합니다.</p>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-500 rounded-lg transition-colors">
          <Plus className="w-5 h-5" />
          평가 항목 추가
        </button>
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
            {courses?.map((course: { id: string; name: string; code: string }) => (
              <option key={course.id} value={course.id}>
                {course.name}
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
                className="bg-surface rounded-xl border border-border p-6"
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
              </div>
            ))}
            {evaluations?.length === 0 && (
              <div className="col-span-full text-center py-8 text-gray-400">
                등록된 평가 항목이 없습니다.
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
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {isLoading ? (
                  <tr>
                    <td
                      colSpan={6}
                      className="px-6 py-8 text-center text-gray-400"
                    >
                      로딩 중...
                    </td>
                  </tr>
                ) : grades?.length === 0 ? (
                  <tr>
                    <td
                      colSpan={6}
                      className="px-6 py-8 text-center text-gray-400"
                    >
                      성적 기록이 없습니다.
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
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
