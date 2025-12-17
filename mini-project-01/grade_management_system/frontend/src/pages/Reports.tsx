import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { coursesApi, reportsApi } from '../services/api'
import { FileText, Download, FileSpreadsheet, File } from 'lucide-react'

export default function Reports() {
  const [selectedCourse, setSelectedCourse] = useState<string>('')
  const [downloading, setDownloading] = useState<string | null>(null)

  const { data: courses } = useQuery({
    queryKey: ['courses'],
    queryFn: () => coursesApi.getAll(),
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

  const reportTypes = [
    {
      title: '출결부',
      description: '반별 출결 현황을 출력합니다.',
      icon: FileText,
      color: 'text-secondary',
      bgColor: 'bg-secondary/20',
      formats: [
        {
          name: 'Excel',
          icon: FileSpreadsheet,
          action: () =>
            downloadFile('attendance', 'xlsx', () =>
              reportsApi.downloadAttendanceExcel(selectedCourse)
            ),
        },
        {
          name: 'PDF',
          icon: File,
          action: () =>
            downloadFile('attendance', 'pdf', () =>
              reportsApi.downloadAttendancePdf(selectedCourse)
            ),
        },
      ],
    },
    {
      title: '성적표',
      description: '과정별 성적 현황을 출력합니다.',
      icon: FileText,
      color: 'text-accent',
      bgColor: 'bg-accent/20',
      formats: [
        {
          name: 'Excel',
          icon: FileSpreadsheet,
          action: () =>
            downloadFile('grades', 'xlsx', () =>
              reportsApi.downloadGradesExcel(selectedCourse)
            ),
        },
        {
          name: 'PDF',
          icon: File,
          action: () =>
            downloadFile('grades', 'pdf', () =>
              reportsApi.downloadGradesPdf(selectedCourse)
            ),
        },
      ],
    },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">리포트</h1>
        <p className="text-gray-400">출결부, 성적표 등 각종 리포트를 출력합니다.</p>
      </div>

      {/* Course selection */}
      <div className="bg-surface rounded-xl border border-border p-6">
        <h2 className="text-lg font-semibold mb-4">과정 선택</h2>
        <select
          value={selectedCourse}
          onChange={(e) => setSelectedCourse(e.target.value)}
          className="w-full max-w-md px-4 py-3 bg-background border border-border rounded-lg"
        >
          <option value="">과정을 선택하세요</option>
          {courses?.map((course: { id: string; name: string; code: string }) => (
            <option key={course.id} value={course.id}>
              {course.name} ({course.code})
            </option>
          ))}
        </select>
      </div>

      {/* Report cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {reportTypes.map((report) => (
          <div
            key={report.title}
            className="bg-surface rounded-xl border border-border overflow-hidden"
          >
            <div className="p-6">
              <div className="flex items-start gap-4">
                <div className={`p-3 rounded-lg ${report.bgColor}`}>
                  <report.icon className={`w-6 h-6 ${report.color}`} />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">{report.title}</h3>
                  <p className="text-sm text-gray-400 mt-1">
                    {report.description}
                  </p>
                </div>
              </div>
            </div>
            <div className="px-6 py-4 bg-background border-t border-border">
              <div className="flex gap-3">
                {report.formats.map((format) => (
                  <button
                    key={format.name}
                    onClick={format.action}
                    disabled={!selectedCourse || downloading !== null}
                    className="flex items-center gap-2 px-4 py-2 bg-surface border border-border rounded-lg hover:border-primary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <format.icon className="w-4 h-4" />
                    <span>{format.name}</span>
                    <Download className="w-4 h-4" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Instructions */}
      <div className="bg-surface rounded-xl border border-border p-6">
        <h2 className="text-lg font-semibold mb-4">사용 안내</h2>
        <ul className="space-y-2 text-gray-400">
          <li className="flex items-start gap-2">
            <span className="text-primary">•</span>
            리포트를 출력하려면 먼저 과정을 선택해야 합니다.
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary">•</span>
            Excel 형식은 추가 편집이 가능하며, PDF 형식은 인쇄에 적합합니다.
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary">•</span>
            개인 성적증명서는 학생 관리 페이지에서 출력할 수 있습니다.
          </li>
        </ul>
      </div>
    </div>
  )
}
