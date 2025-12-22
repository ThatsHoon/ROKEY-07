import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { coursesApi, reportsApi } from '../services/api'
import type { Course, Class } from '../types'
import { FileText, Download, FileSpreadsheet, File } from 'lucide-react'

export default function Reports() {
  const [selectedCourse, setSelectedCourse] = useState<string>('')
  const [selectedClass, setSelectedClass] = useState<string>('')
  const [downloading, setDownloading] = useState<string | null>(null)

  const { data: courses } = useQuery({
    queryKey: ['courses'],
    queryFn: () => coursesApi.getAll(),
  })

  const { data: classes } = useQuery({
    queryKey: ['classes', selectedCourse],
    queryFn: () => coursesApi.getClasses(selectedCourse),
    enabled: !!selectedCourse,
  })

  const downloadFile = async (
    type: string,
    format: string,
    fetchFn: () => Promise<Blob>
  ) => {
    const key = `${type}-${format}`
    setDownloading(key)
    try {
      const blob = await fetchFn()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${type}_report.${format}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('Download failed:', error)
      alert('다운로드에 실패했습니다.')
    } finally {
      setDownloading(null)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">리포트</h1>
        <p className="text-gray-400">출결부, 성적표 등 각종 리포트를 출력합니다.</p>
      </div>

      {/* Selection */}
      <div className="bg-surface rounded-xl border border-border p-6">
        <h2 className="text-lg font-semibold mb-4">과정 및 반 선택</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl">
          <div>
            <label className="block text-sm font-medium mb-2">과정</label>
            <select
              value={selectedCourse}
              onChange={(e) => {
                setSelectedCourse(e.target.value)
                setSelectedClass('')
              }}
              className="w-full px-4 py-3 bg-background border border-border rounded-lg"
            >
              <option value="">과정을 선택하세요</option>
              {courses?.map((course: Course) => (
                <option key={course.id} value={course.id}>
                  [{course.code}] {course.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">반 (출결부용)</label>
            <select
              value={selectedClass}
              onChange={(e) => setSelectedClass(e.target.value)}
              disabled={!selectedCourse}
              className="w-full px-4 py-3 bg-background border border-border rounded-lg disabled:opacity-50"
            >
              <option value="">반을 선택하세요</option>
              {classes?.map((cls: Class) => (
                <option key={cls.id} value={cls.id}>
                  {cls.name} ({cls.student_count}/{cls.capacity})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Report cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Attendance Report */}
        <div className="bg-surface rounded-xl border border-border overflow-hidden">
          <div className="p-6">
            <div className="flex items-start gap-4">
              <div className="p-3 rounded-lg bg-secondary/20">
                <FileText className="w-6 h-6 text-secondary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">출결부</h3>
                <p className="text-sm text-gray-400 mt-1">
                  반별 출결 현황을 출력합니다.
                </p>
                {!selectedClass && selectedCourse && (
                  <p className="text-xs text-yellow-400 mt-2">
                    출결부 출력을 위해 반을 선택해주세요.
                  </p>
                )}
              </div>
            </div>
          </div>
          <div className="px-6 py-4 bg-background border-t border-border">
            <div className="flex gap-3">
              <button
                onClick={() => downloadFile('attendance', 'xlsx', () =>
                  reportsApi.downloadAttendanceExcel(selectedClass)
                )}
                disabled={!selectedClass || downloading !== null}
                className="flex items-center gap-2 px-4 py-2 bg-surface border border-border rounded-lg hover:border-primary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <FileSpreadsheet className="w-4 h-4" />
                <span>Excel</span>
                <Download className="w-4 h-4" />
              </button>
              <button
                onClick={() => downloadFile('attendance', 'pdf', () =>
                  reportsApi.downloadAttendancePdf(selectedClass)
                )}
                disabled={!selectedClass || downloading !== null}
                className="flex items-center gap-2 px-4 py-2 bg-surface border border-border rounded-lg hover:border-primary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <File className="w-4 h-4" />
                <span>PDF</span>
                <Download className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Grades Report */}
        <div className="bg-surface rounded-xl border border-border overflow-hidden">
          <div className="p-6">
            <div className="flex items-start gap-4">
              <div className="p-3 rounded-lg bg-accent/20">
                <FileText className="w-6 h-6 text-accent" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">성적표</h3>
                <p className="text-sm text-gray-400 mt-1">
                  과정별 성적 현황을 출력합니다.
                </p>
              </div>
            </div>
          </div>
          <div className="px-6 py-4 bg-background border-t border-border">
            <div className="flex gap-3">
              <button
                onClick={() => downloadFile('grades', 'xlsx', () =>
                  reportsApi.downloadGradesExcel(selectedCourse)
                )}
                disabled={!selectedCourse || downloading !== null}
                className="flex items-center gap-2 px-4 py-2 bg-surface border border-border rounded-lg hover:border-primary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <FileSpreadsheet className="w-4 h-4" />
                <span>Excel</span>
                <Download className="w-4 h-4" />
              </button>
              <button
                onClick={() => downloadFile('grades', 'pdf', () =>
                  reportsApi.downloadGradesPdf(selectedCourse)
                )}
                disabled={!selectedCourse || downloading !== null}
                className="flex items-center gap-2 px-4 py-2 bg-surface border border-border rounded-lg hover:border-primary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <File className="w-4 h-4" />
                <span>PDF</span>
                <Download className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="bg-surface rounded-xl border border-border p-6">
        <h2 className="text-lg font-semibold mb-4">사용 안내</h2>
        <ul className="space-y-2 text-gray-400">
          <li className="flex items-start gap-2">
            <span className="text-primary">1.</span>
            출결부를 출력하려면 과정과 반을 모두 선택해야 합니다.
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary">2.</span>
            성적표는 과정만 선택하면 출력할 수 있습니다.
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary">3.</span>
            Excel 형식은 추가 편집이 가능하며, PDF 형식은 인쇄에 적합합니다.
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary">4.</span>
            개인 성적증명서는 학생 관리 페이지에서 출력할 수 있습니다.
          </li>
        </ul>
      </div>
    </div>
  )
}
