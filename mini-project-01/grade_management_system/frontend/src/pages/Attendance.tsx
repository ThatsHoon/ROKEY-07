import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { attendanceApi, coursesApi } from '../services/api'
import type { Attendance as AttendanceType, Class } from '../types'
import { Calendar, Filter, Check, Clock, X as XIcon, AlertCircle } from 'lucide-react'

const statusConfig = {
  출석: { icon: Check, color: 'text-green-400', bg: 'bg-green-500/20' },
  지각: { icon: Clock, color: 'text-yellow-400', bg: 'bg-yellow-500/20' },
  조퇴: { icon: AlertCircle, color: 'text-orange-400', bg: 'bg-orange-500/20' },
  결석: { icon: XIcon, color: 'text-red-400', bg: 'bg-red-500/20' },
  공결: { icon: Calendar, color: 'text-blue-400', bg: 'bg-blue-500/20' },
}

export default function Attendance() {
  const [selectedClass, setSelectedClass] = useState<string>('')
  const [selectedDate, setSelectedDate] = useState<string>(
    new Date().toISOString().split('T')[0]
  )

  const { data: courses } = useQuery({
    queryKey: ['courses'],
    queryFn: () => coursesApi.getAll(),
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">출결 관리</h1>
        <p className="text-gray-400">학생들의 출결 현황을 관리합니다.</p>
      </div>

      {/* Filters */}
      <div className="bg-surface rounded-xl border border-border p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-gray-400" />
            <span className="text-sm font-medium">필터</span>
          </div>
          <select
            value={selectedClass}
            onChange={(e) => setSelectedClass(e.target.value)}
            className="px-4 py-2 bg-background border border-border rounded-lg"
          >
            <option value="">반 선택</option>
            {courses?.map((course: { id: string; name: string; code: string }) => (
              <option key={course.id} value={course.id}>
                {course.name}
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
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {!selectedClass ? (
              <tr>
                <td colSpan={8} className="px-6 py-8 text-center text-gray-400">
                  반을 선택해주세요.
                </td>
              </tr>
            ) : isLoading ? (
              <tr>
                <td colSpan={8} className="px-6 py-8 text-center text-gray-400">
                  로딩 중...
                </td>
              </tr>
            ) : attendance?.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-6 py-8 text-center text-gray-400">
                  출결 기록이 없습니다.
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
                    <td className="px-6 py-4 text-gray-400">
                      {record.notes || '-'}
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
