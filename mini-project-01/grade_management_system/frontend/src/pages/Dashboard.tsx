import { useQuery } from '@tanstack/react-query'
import { studentsApi, coursesApi } from '../services/api'
import { useAuthStore } from '../store/authStore'
import {
  Users,
  BookOpen,
  Calendar,
  TrendingUp,
  UserCheck,
} from 'lucide-react'

export default function Dashboard() {
  const { user } = useAuthStore()

  const { data: students } = useQuery({
    queryKey: ['students'],
    queryFn: () => studentsApi.getAll({ limit: 1000 }),
  })

  const { data: courses } = useQuery({
    queryKey: ['courses'],
    queryFn: () => coursesApi.getAll(),
  })

  const stats = [
    {
      name: '총 학생 수',
      value: students?.length || 0,
      icon: Users,
      color: 'bg-primary',
    },
    {
      name: '진행 중인 과정',
      value: courses?.length || 0,
      icon: BookOpen,
      color: 'bg-secondary',
    },
    {
      name: '오늘 출석률',
      value: '94.5%',
      icon: UserCheck,
      color: 'bg-green-500',
    },
    {
      name: '평균 성적',
      value: '82.3',
      icon: TrendingUp,
      color: 'bg-accent',
    },
  ]

  return (
    <div className="space-y-6">
      {/* Welcome message */}
      <div className="bg-surface rounded-xl border border-border p-6">
        <h1 className="text-2xl font-bold">
          안녕하세요, {user?.name}님!
        </h1>
        <p className="text-gray-400 mt-2">
          오늘의 학사 현황을 확인하세요.
        </p>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <div
            key={stat.name}
            className="bg-surface rounded-xl border border-border p-6"
          >
            <div className="flex items-center gap-4">
              <div className={`${stat.color} p-3 rounded-lg`}>
                <stat.icon className="w-6 h-6 text-white" />
              </div>
              <div>
                <p className="text-sm text-gray-400">{stat.name}</p>
                <p className="text-2xl font-bold">{stat.value}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Recent activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent students */}
        <div className="bg-surface rounded-xl border border-border overflow-hidden">
          <div className="p-4 border-b border-border">
            <h2 className="text-lg font-semibold">최근 등록 학생</h2>
          </div>
          <div className="divide-y divide-border">
            {students?.slice(0, 5).map((student: { id: string; student_number: string; user_name: string; created_at: string }) => (
              <div key={student.id} className="p-4 flex items-center gap-4">
                <div className="w-10 h-10 bg-primary/20 rounded-full flex items-center justify-center text-primary font-semibold">
                  {student.user_name?.charAt(0) || 'S'}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{student.user_name || '이름 없음'}</p>
                  <p className="text-sm text-gray-400">{student.student_number}</p>
                </div>
                <span className="text-xs text-gray-400">
                  {new Date(student.created_at).toLocaleDateString()}
                </span>
              </div>
            )) || (
              <div className="p-4 text-center text-gray-400">
                등록된 학생이 없습니다.
              </div>
            )}
          </div>
        </div>

        {/* Recent courses */}
        <div className="bg-surface rounded-xl border border-border overflow-hidden">
          <div className="p-4 border-b border-border">
            <h2 className="text-lg font-semibold">진행 중인 과정</h2>
          </div>
          <div className="divide-y divide-border">
            {courses?.slice(0, 5).map((course: { id: string; name: string; code: string; teacher_name: string }) => (
              <div key={course.id} className="p-4 flex items-center gap-4">
                <div className="w-10 h-10 bg-secondary/20 rounded-full flex items-center justify-center text-secondary">
                  <BookOpen className="w-5 h-5" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{course.name}</p>
                  <p className="text-sm text-gray-400">{course.code}</p>
                </div>
                <span className="text-xs text-gray-400">
                  {course.teacher_name || '담당자 미정'}
                </span>
              </div>
            )) || (
              <div className="p-4 text-center text-gray-400">
                진행 중인 과정이 없습니다.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick actions */}
      <div className="bg-surface rounded-xl border border-border p-6">
        <h2 className="text-lg font-semibold mb-4">빠른 작업</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <button className="p-4 bg-background rounded-lg hover:bg-border transition-colors text-left">
            <Users className="w-6 h-6 text-primary mb-2" />
            <p className="font-medium">학생 등록</p>
          </button>
          <button className="p-4 bg-background rounded-lg hover:bg-border transition-colors text-left">
            <Calendar className="w-6 h-6 text-secondary mb-2" />
            <p className="font-medium">출결 입력</p>
          </button>
          <button className="p-4 bg-background rounded-lg hover:bg-border transition-colors text-left">
            <TrendingUp className="w-6 h-6 text-accent mb-2" />
            <p className="font-medium">성적 입력</p>
          </button>
          <button className="p-4 bg-background rounded-lg hover:bg-border transition-colors text-left">
            <BookOpen className="w-6 h-6 text-green-500 mb-2" />
            <p className="font-medium">리포트 출력</p>
          </button>
        </div>
      </div>
    </div>
  )
}
