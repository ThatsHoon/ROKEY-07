import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { attendanceApi, coursesApi } from '../services/api'
import type { Attendance as AttendanceType, AttendanceStatus, Course, Class } from '../types'
import { Calendar, Filter, Check, Clock, X as XIcon, AlertCircle, Plus, Edit, Save } from 'lucide-react'

const statusConfig: Record<AttendanceStatus, { icon: typeof Check; color: string; bg: string }> = {
  출석: { icon: Check, color: 'text-green-400', bg: 'bg-green-500/20' },
  지각: { icon: Clock, color: 'text-yellow-400', bg: 'bg-yellow-500/20' },
  조퇴: { icon: AlertCircle, color: 'text-orange-400', bg: 'bg-orange-500/20' },
  결석: { icon: XIcon, color: 'text-red-400', bg: 'bg-red-500/20' },
  공결: { icon: Calendar, color: 'text-blue-400', bg: 'bg-blue-500/20' },
}

const STATUS_OPTIONS: AttendanceStatus[] = ['출석', '지각', '조퇴', '결석', '공결']

export default function Attendance() {
  const queryClient = useQueryClient()
  const [selectedCourse, setSelectedCourse] = useState<string>('')
  const [selectedClass, setSelectedClass] = useState<string>('')
  const [selectedDate, setSelectedDate] = useState<string>(
    new Date().toISOString().split('T')[0]
  )
  const [showInputForm, setShowInputForm] = useState(false)
  const [editingRecord, setEditingRecord] = useState<AttendanceType | null>(null)
  const [sessionNo, setSessionNo] = useState(1)

  const { data: courses } = useQuery({
    queryKey: ['courses'],
    queryFn: () => coursesApi.getAll(),
  })

  const { data: classes } = useQuery({
    queryKey: ['classes', selectedCourse],
    queryFn: () => coursesApi.getClasses(selectedCourse),
    enabled: !!selectedCourse,
  })

  const { data: attendance, isLoading } = useQuery({
    queryKey: ['attendance', selectedClass, selectedDate],
    queryFn: () =>
      attendanceApi.getAll({
        class_id: selectedClass || undefined,
        date_from: selectedDate,
        date_to: selectedDate,
      }),
    enabled: !!selectedClass,
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: Record<string, unknown> }) =>
      attendanceApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['attendance'] })
      setEditingRecord(null)
    },
  })

  const bulkCreateMutation = useMutation({
    mutationFn: attendanceApi.createBulk,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['attendance'] })
      setShowInputForm(false)
    },
  })

  const handleBulkSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const formData = new FormData(e.currentTarget)
    const records: { student_id: string; status: AttendanceStatus }[] = []

    formData.forEach((value, key) => {
      if (key.startsWith('status_')) {
        const studentId = key.replace('status_', '')
        records.push({
          student_id: studentId,
          status: value as AttendanceStatus,
        })
      }
    })

    bulkCreateMutation.mutate({
      class_id: selectedClass,
      date: selectedDate,
      session_no: sessionNo,
      records,
    })
  }

  const handleEditSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!editingRecord) return
    const formData = new FormData(e.currentTarget)
    const data = {
      status: formData.get('status') as AttendanceStatus,
      check_in_time: formData.get('check_in_time') || null,
      check_out_time: formData.get('check_out_time') || null,
      notes: formData.get('notes') || null,
    }
    updateMutation.mutate({ id: editingRecord.id, data })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">출결 관리</h1>
          <p className="text-gray-400">학생들의 출결 현황을 관리합니다.</p>
        </div>
        {selectedClass && (
          <button
            onClick={() => setShowInputForm(true)}
            className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-500 rounded-lg transition-colors"
          >
            <Plus className="w-5 h-5" />
            출결 입력
          </button>
        )}
      </div>

      {/* Filters */}
      <div className="bg-surface rounded-xl border border-border p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-gray-400" />
            <span className="text-sm font-medium">필터</span>
          </div>
          <select
            value={selectedCourse}
            onChange={(e) => {
              setSelectedCourse(e.target.value)
              setSelectedClass('')
            }}
            className="px-4 py-2 bg-background border border-border rounded-lg"
          >
            <option value="">과정 선택</option>
            {courses?.map((course: Course) => (
              <option key={course.id} value={course.id}>
                [{course.code}] {course.name}
              </option>
            ))}
          </select>
          <select
            value={selectedClass}
            onChange={(e) => setSelectedClass(e.target.value)}
            className="px-4 py-2 bg-background border border-border rounded-lg"
            disabled={!selectedCourse}
          >
            <option value="">반 선택</option>
            {classes?.map((cls: Class) => (
              <option key={cls.id} value={cls.id}>
                {cls.name} ({cls.student_count}/{cls.capacity})
              </option>
            ))}
          </select>
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="px-4 py-2 bg-background border border-border rounded-lg"
          />
        </div>
      </div>

      {/* Attendance status summary */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {Object.entries(statusConfig).map(([status, config]) => {
          const count = attendance?.filter(
            (a: AttendanceType) => a.status === status
          ).length || 0
          return (
            <div
              key={status}
              className="bg-surface rounded-xl border border-border p-4"
            >
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${config.bg}`}>
                  <config.icon className={`w-5 h-5 ${config.color}`} />
                </div>
                <div>
                  <p className="text-sm text-gray-400">{status}</p>
                  <p className="text-xl font-bold">{count}</p>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Attendance table */}
      <div className="bg-surface rounded-xl border border-border overflow-hidden">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold">출결 현황</h2>
        </div>
        <table className="w-full">
          <thead className="bg-background">
            <tr>
              <th className="px-6 py-4 text-left text-sm font-semibold">학번</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">이름</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">회차</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">날짜</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">상태</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">입실</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">퇴실</th>
              <th className="px-6 py-4 text-left text-sm font-semibold">비고</th>
              <th className="px-6 py-4 text-right text-sm font-semibold">작업</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {!selectedClass ? (
              <tr>
                <td colSpan={9} className="px-6 py-8 text-center text-gray-400">
                  과정과 반을 선택해주세요.
                </td>
              </tr>
            ) : isLoading ? (
              <tr>
                <td colSpan={9} className="px-6 py-8 text-center text-gray-400">
                  로딩 중...
                </td>
              </tr>
            ) : attendance?.length === 0 ? (
              <tr>
                <td colSpan={9} className="px-6 py-8 text-center text-gray-400">
                  출결 기록이 없습니다. 출결 입력 버튼을 눌러 기록을 추가하세요.
                </td>
              </tr>
            ) : (
              attendance?.map((record: AttendanceType) => {
                const config = statusConfig[record.status]
                return (
                  <tr key={record.id} className="hover:bg-background/50">
                    <td className="px-6 py-4 font-mono">
                      {record.student_number}
                    </td>
                    <td className="px-6 py-4">{record.student_name}</td>
                    <td className="px-6 py-4">{record.session_no}회</td>
                    <td className="px-6 py-4 text-gray-400">{record.date}</td>
                    <td className="px-6 py-4">
                      <span
                        className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs ${config.bg} ${config.color}`}
                      >
                        <config.icon className="w-3 h-3" />
                        {record.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-gray-400">
                      {record.check_in_time || '-'}
                    </td>
                    <td className="px-6 py-4 text-gray-400">
                      {record.check_out_time || '-'}
                    </td>
                    <td className="px-6 py-4 text-gray-400 max-w-32 truncate">
                      {record.notes || '-'}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button
                        onClick={() => setEditingRecord(record)}
                        className="p-2 hover:bg-background rounded-lg transition-colors"
                        title="수정"
                      >
                        <Edit className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Bulk Input Modal */}
      {showInputForm && selectedClass && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-surface rounded-xl border border-border w-full max-w-2xl mx-4 max-h-[80vh] overflow-hidden flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h2 className="text-lg font-semibold">출결 일괄 입력</h2>
              <button onClick={() => setShowInputForm(false)}>
                <XIcon className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={handleBulkSubmit} className="flex flex-col flex-1 overflow-hidden">
              <div className="p-4 space-y-4 border-b border-border">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">날짜</label>
                    <input
                      type="date"
                      value={selectedDate}
                      onChange={(e) => setSelectedDate(e.target.value)}
                      className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">회차</label>
                    <input
                      type="number"
                      min="1"
                      value={sessionNo}
                      onChange={(e) => setSessionNo(parseInt(e.target.value) || 1)}
                      className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                    />
                  </div>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-4">
                <p className="text-sm text-gray-400 mb-4">
                  각 학생의 출결 상태를 선택하세요. (현재 반에 등록된 학생 기준)
                </p>
                <div className="space-y-2">
                  {attendance && attendance.length > 0 ? (
                    Array.from(new Map<string, AttendanceType>(attendance.map((a: AttendanceType) => [a.student_id, a])).values()).map((record) => (
                      <div key={record.student_id} className="flex items-center justify-between p-3 bg-background rounded-lg">
                        <div>
                          <p className="font-medium">{record.student_name}</p>
                          <p className="text-xs text-gray-400">{record.student_number}</p>
                        </div>
                        <select
                          name={`status_${record.student_id}`}
                          defaultValue="출석"
                          className="px-3 py-2 bg-surface border border-border rounded-lg text-sm"
                        >
                          {STATUS_OPTIONS.map((status) => (
                            <option key={status} value={status}>{status}</option>
                          ))}
                        </select>
                      </div>
                    ))
                  ) : (
                    <p className="text-center text-gray-400 py-8">
                      해당 반에 등록된 학생이 없거나 출결 기록이 없습니다.
                    </p>
                  )}
                </div>
              </div>
              <div className="flex gap-3 p-4 border-t border-border">
                <button
                  type="button"
                  onClick={() => setShowInputForm(false)}
                  className="flex-1 py-2 bg-background border border-border rounded-lg hover:bg-border transition-colors"
                >
                  취소
                </button>
                <button
                  type="submit"
                  disabled={bulkCreateMutation.isPending}
                  className="flex-1 py-2 bg-primary hover:bg-primary-500 rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  <Save className="w-4 h-4" />
                  {bulkCreateMutation.isPending ? '저장 중...' : '저장'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit Modal */}
      {editingRecord && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-surface rounded-xl border border-border w-full max-w-md mx-4">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <div>
                <h2 className="text-lg font-semibold">출결 수정</h2>
                <p className="text-sm text-gray-400">
                  {editingRecord.student_name} ({editingRecord.student_number})
                </p>
              </div>
              <button onClick={() => setEditingRecord(null)}>
                <XIcon className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={handleEditSubmit} className="p-4 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">날짜</label>
                  <input
                    value={editingRecord.date}
                    disabled
                    className="w-full px-4 py-2 bg-background/50 border border-border rounded-lg text-gray-400"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">회차</label>
                  <input
                    value={`${editingRecord.session_no}회`}
                    disabled
                    className="w-full px-4 py-2 bg-background/50 border border-border rounded-lg text-gray-400"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">출결 상태 *</label>
                <select
                  name="status"
                  defaultValue={editingRecord.status}
                  required
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                >
                  {STATUS_OPTIONS.map((status) => (
                    <option key={status} value={status}>{status}</option>
                  ))}
                </select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">입실 시간</label>
                  <input
                    name="check_in_time"
                    type="time"
                    defaultValue={editingRecord.check_in_time || ''}
                    className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">퇴실 시간</label>
                  <input
                    name="check_out_time"
                    type="time"
                    defaultValue={editingRecord.check_out_time || ''}
                    className="w-full px-4 py-2 bg-background border border-border rounded-lg"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">비고</label>
                <textarea
                  name="notes"
                  defaultValue={editingRecord.notes || ''}
                  rows={2}
                  className="w-full px-4 py-2 bg-background border border-border rounded-lg resize-none"
                  placeholder="특이사항을 입력하세요"
                />
              </div>
              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setEditingRecord(null)}
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
    </div>
  )
}
