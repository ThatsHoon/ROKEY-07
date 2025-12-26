#  Design Blueprint

이 문서는  웹사이트의 디자인 시스템을 정리한 것으로, 유사한 스타일의 다른 프로젝트에 적용할 수 있습니다.

---

## 1. 색상 팔레트 (Color Palette)

### 1.1 브랜드 컬러

| 이름 | HEX | HSL | 용도 |
|------|-----|-----|------|
| **Primary** | `#4B39EF` | 250 77% 58% | 주요 액션, 링크, 강조 |
| **Secondary** | `#39D2C0` | 173 63% 53% | 보조 강조, 성공 상태 |
| **Accent** | `#EE8B60` | 21 81% 65% | CTA 버튼, 알림, 도네이션 |

### 1.2 배경 및 표면 컬러

| 이름 | HEX | 용도 |
|------|-----|------|
| **Background** | `#1D2428` | 페이지 배경 |
| **Surface** | `#14181B` | 카드, 컨테이너 배경 |
| **Border** | `#2A3137` | 테두리, 구분선 |

### 1.3 텍스트 컬러

| 이름 | 값 | 용도 |
|------|-----|------|
| **Foreground** | 95% 밝기 (거의 흰색) | 기본 텍스트 |
| **Muted Foreground** | 63.9% 밝기 | 보조 텍스트, 설명 |

### 1.4 Primary 색상 스케일

```css
primary-50:  #E8E5FC
primary-100: #D1CBF9
primary-200: #A397F3
primary-300: #7563ED
primary-400: #4B39EF (DEFAULT)
primary-500: #3520E3
primary-600: #2819B5
primary-700: #1E1287
primary-800: #140C59
primary-900: #0A062B
```

### 1.5 Secondary 색상 스케일

```css
secondary-50:  #E5F9F6
secondary-100: #CCF3ED
secondary-200: #99E7DB
secondary-300: #66DBC9
secondary-400: #39D2C0 (DEFAULT)
secondary-500: #2DB5A5
secondary-600: #248E82
secondary-700: #1B6A60
secondary-800: #12453D
secondary-900: #09211B
```

### 1.6 Accent 색상 스케일

```css
accent-50:  #FDF3EE
accent-100: #FCE7DD
accent-200: #F9CFBB
accent-300: #F6B799
accent-400: #F3A07C
accent-500: #EE8B60 (DEFAULT)
accent-600: #E96D3C
accent-700: #D54C1A
accent-800: #A53B14
accent-900: #75290E
```

---

## 2. 타이포그래피 (Typography)

### 2.1 폰트 패밀리

```css
font-family: 'Inter Tight', sans-serif;
```

**Google Fonts 임포트:**
```css
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&display=swap');
```

### 2.2 폰트 크기 시스템

| 클래스 | 크기 | 용도 |
|--------|------|------|
| `text-xs` | 0.75rem (12px) | 캡션, 메타 정보 |
| `text-sm` | 0.875rem (14px) | 보조 텍스트 |
| `text-base` | 1rem (16px) | 기본 본문 |
| `text-lg` | 1.125rem (18px) | 서브타이틀 |
| `text-xl` | 1.25rem (20px) | 섹션 제목 |
| `text-2xl` | 1.5rem (24px) | 컨테이너 타이틀 |
| `text-3xl` | 1.875rem (30px) | 페이지 타이틀 |

### 2.3 폰트 웨이트

| 클래스 | 값 | 용도 |
|--------|-----|------|
| `font-normal` | 400 | 기본 본문 |
| `font-medium` | 500 | 강조 텍스트 |
| `font-semibold` | 600 | 라벨, 서브타이틀 |
| `font-bold` | 700 | 타이틀, 헤딩 |

---

## 3. 간격 시스템 (Spacing)

### 3.1 기본 간격 단위

Tailwind 기본 간격 시스템 사용 (4px 단위)

| 클래스 | 값 | 주요 용도 |
|--------|-----|----------|
| `gap-1` | 4px | 아이콘과 텍스트 사이 |
| `gap-2` | 8px | 인라인 요소 간격 |
| `gap-3` | 12px | 리스트 아이템 간격 |
| `gap-4` | 16px | 카드 내 요소 간격 |
| `gap-6` | 24px | 섹션 간 간격 |
| `space-y-6` | 24px | 페이지 내 컨테이너 간격 |

### 3.2 패딩 패턴

```css
/* 컨테이너 패딩 */
p-4      /* 16px - 기본 카드 */
p-6      /* 24px - 넓은 카드 */
px-4 py-2  /* 버튼 */
px-6 py-4  /* 큰 버튼 */

/* 반응형 패딩 */
p-3 md:p-6  /* 모바일 12px, 데스크탑 24px */
```

---

## 4. 레이아웃 (Layout)

### 4.1 컨테이너

```css
.container-max {
  max-width: 800px;
  margin: 0 auto;
  padding: 0 16px;
}
```

### 4.2 페이지 구조

```jsx
<div className="container-max py-6 space-y-6">
  {/* 페이지 콘텐츠 */}
</div>
```

### 4.3 그리드 시스템

```css
/* 2열 그리드 */
grid grid-cols-2 gap-4

/* 반응형 그리드 */
grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3
```

---

## 5. 컴포넌트 스타일 (Components)

### 5.1 카드 / 컨테이너

```jsx
{/* 기본 카드 */}
<div className="bg-surface rounded-xl border border-border p-4">
  {/* 내용 */}
</div>

{/* 헤더가 있는 카드 */}
<div className="bg-surface rounded-xl border border-border overflow-hidden">
  <div className="p-4 border-b border-border">
    <h3 className="text-lg font-semibold">제목</h3>
  </div>
  <div className="p-4">
    {/* 내용 */}
  </div>
</div>
```

### 5.2 버튼

```jsx
{/* Primary 버튼 */}
<button className="px-6 py-4 bg-primary hover:bg-primary/90 rounded-lg font-semibold transition-colors">
  버튼 텍스트
</button>

{/* Secondary 버튼 */}
<button className="px-4 py-2 bg-secondary hover:bg-secondary/90 rounded-lg font-semibold transition-colors">
  버튼 텍스트
</button>

{/* Accent/CTA 버튼 */}
<button className="px-6 py-3 bg-accent hover:bg-accent/90 rounded-lg font-semibold text-white transition-colors">
  버튼 텍스트
</button>

{/* Outline 버튼 */}
<button className="px-4 py-2 bg-surface border border-border hover:border-primary rounded-lg transition-colors">
  버튼 텍스트
</button>

{/* Disabled 버튼 */}
<button className="... disabled:opacity-50 disabled:cursor-not-allowed">
  버튼 텍스트
</button>

{/* 아이콘 + 텍스트 버튼 */}
<button className="flex items-center gap-2 px-4 py-2 ...">
  <Icon className="w-5 h-5" />
  <span>버튼 텍스트</span>
</button>
```

### 5.3 입력 필드

```jsx
{/* 텍스트 입력 */}
<input
  type="text"
  className="w-full px-4 py-2 bg-surface border border-border rounded-lg focus:outline-none focus:border-primary"
/>

{/* 셀렉트 박스 */}
<select className="w-full px-4 py-2 bg-surface border border-border rounded-lg focus:outline-none focus:border-primary">
  <option>옵션 1</option>
</select>

{/* 반응형 입력 */}
<input className="w-full px-1.5 py-1.5 sm:px-2 sm:py-2 md:px-4 text-[11px] sm:text-xs md:text-base bg-surface border border-border rounded-lg focus:outline-none focus:border-primary" />
```

### 5.4 드롭다운 메뉴

```jsx
{/* 드롭다운 컨테이너 */}
<div className="absolute top-full left-0 mt-2 w-64 bg-surface border border-border rounded-lg shadow-lg">
  {/* 메뉴 아이템 */}
  <button className="w-full text-left px-4 py-3 hover:bg-background transition-colors flex items-center gap-3">
    <Icon className="w-4 h-4" />
    <span>메뉴 항목</span>
  </button>
</div>
```

### 5.5 토글 버튼 그룹

```jsx
<div className="flex gap-0.5">
  {[1, 2, 3, 4, 5].map((value) => (
    <button
      key={value}
      className={`flex-1 h-8 rounded text-xs border transition-colors ${
        selected === value
          ? 'bg-primary border-primary text-white'
          : 'bg-surface border-border hover:border-primary'
      }`}
    >
      {value}
    </button>
  ))}
</div>
```

### 5.6 접이식 컨테이너 (Collapsible)

```jsx
<div className="bg-surface rounded-xl border border-border overflow-hidden">
  <button
    onClick={() => setOpen(!open)}
    className="w-full flex items-center justify-between p-4 hover:bg-background/50 transition-colors"
  >
    <div className="flex items-center gap-2">
      <Icon className="w-5 h-5 text-yellow-500" />
      <h2 className="text-lg font-bold text-yellow-500">Notice!</h2>
    </div>
    {open ? <ChevronUp /> : <ChevronDown />}
  </button>
  {open && (
    <div className="px-4 pb-4 space-y-3 border-t border-border pt-4">
      {/* 내용 */}
    </div>
  )}
</div>
```

---

## 6. 헤더 스타일 (Header)

```jsx
<header className="bg-primary/70 backdrop-blur-md border-b border-primary-dark sticky top-0 z-50">
  <div className="container-max">
    <div className="flex items-center justify-between h-16 md:h-20">
      {/* 로고 */}
      <Link href="/" className="flex items-center space-x-2">
        <Image src="/favicon.ico" alt="Logo" width={64} height={64} />
        <div className="text-2xl font-bold text-white">브랜드명</div>
      </Link>

      {/* 네비게이션 */}
      <nav className="hidden md:flex items-center space-x-4">
        {/* 메뉴 항목들 */}
      </nav>
    </div>
  </div>
</header>
```

---

## 7. 푸터 스타일 (Footer)

```jsx
<footer className="bg-surface border-t border-border mt-auto">
  <div className="container-max py-12">
    <div className="text-center space-y-4">
      <p className="text-sm text-muted-foreground">
        면책 조항 텍스트
      </p>
      <div className="flex items-center justify-center space-x-2 text-sm">
        <Mail className="w-4 h-4" />
        <a href="mailto:email@example.com" className="text-primary hover:underline">
          email@example.com
        </a>
      </div>
      <p className="text-xs text-muted-foreground">
        © 2024 브랜드명. All rights reserved.
      </p>
    </div>
  </div>
</footer>
```

---

## 8. 특수 효과 (Effects)

### 8.1 커스텀 스크롤바

```css
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #14181B;
}

::-webkit-scrollbar-thumb {
  background: #4B39EF;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #3520E3;
}
```

### 8.2 Shimmer 효과 (로딩)

```css
.shimmer {
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 255, 255, 0.1),
    transparent
  );
  background-size: 200% 100%;
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
```

### 8.3 Fade In 애니메이션

```css
.animate-fadeIn {
  animation: fadeIn 0.3s ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
```

### 8.4 호버 효과

```css
/* 기본 전환 */
transition-colors
transition-all

/* 스케일 효과 */
hover:scale-105

/* 그림자 효과 */
hover:shadow-2xl

/* 테두리 강조 */
hover:border-primary
```

---

## 9. 반응형 디자인 (Responsive)

### 9.1 브레이크포인트

| 접두사 | 최소 너비 | 용도 |
|--------|----------|------|
| (없음) | 0px | 모바일 |
| `sm:` | 640px | 작은 태블릿 |
| `md:` | 768px | 태블릿/데스크탑 |
| `lg:` | 1024px | 데스크탑 |
| `xl:` | 1280px | 와이드 데스크탑 |

### 9.2 반응형 패턴

```jsx
{/* 숨기기/보이기 */}
<div className="hidden md:block">데스크탑 전용</div>
<div className="md:hidden">모바일 전용</div>

{/* 반응형 텍스트 크기 */}
<h1 className="text-xl md:text-2xl lg:text-3xl">제목</h1>

{/* 반응형 그리드 */}
<div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 md:gap-4">

{/* 반응형 패딩 */}
<div className="p-3 md:p-6">

{/* 반응형 flex 방향 */}
<div className="flex flex-col md:flex-row">
```

---

## 10. 아이콘 시스템 (Icons)

### 10.1 아이콘 라이브러리

**Lucide React** 사용

```bash
npm install lucide-react
```

### 10.2 아이콘 사용법

```jsx
import { Home, User, Settings, ChevronDown } from 'lucide-react';

{/* 기본 아이콘 */}
<Home className="w-5 h-5" />

{/* 색상이 있는 아이콘 */}
<Home className="w-5 h-5 text-primary" />

{/* 버튼 내 아이콘 */}
<button className="flex items-center gap-2">
  <Home className="w-5 h-5" />
  <span>홈</span>
</button>
```

### 10.3 자주 사용하는 아이콘

| 아이콘 | 용도 |
|--------|------|
| `Menu`, `X` | 햄버거 메뉴 |
| `ChevronDown`, `ChevronUp` | 드롭다운, 접기/펼치기 |
| `Globe` | 언어 선택 |
| `LogIn`, `LogOut` | 로그인/로그아웃 |
| `User` | 사용자 프로필 |
| `Settings` | 설정 |
| `AlertTriangle` | 경고, 공지 |
| `Coffee` | 후원, 도네이션 |

---

## 11. 상태 색상 (State Colors)

| 상태 | 색상 | 클래스 |
|------|------|--------|
| 성공 | 초록 | `text-green-500`, `bg-green-500` |
| 경고 | 노랑 | `text-yellow-500`, `bg-yellow-500` |
| 에러 | 빨강 | `text-red-500`, `bg-red-500` |
| 정보 | 파랑 | `text-blue-500`, `bg-blue-500` |
| 프리미엄 | 골드 | `text-yellow-400` |

---

## 12. Tailwind 설정 파일

### 12.1 tailwind.config.ts

```typescript
import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#4B39EF',
          // ... 스케일
        },
        secondary: {
          DEFAULT: '#39D2C0',
          // ... 스케일
        },
        accent: {
          DEFAULT: '#EE8B60',
          // ... 스케일
        },
        background: {
          DEFAULT: '#1D2428',
          light: '#2A3137',
        },
        surface: {
          DEFAULT: '#14181B',
          light: '#1D2428',
        },
        border: "hsl(var(--border))",
        foreground: "hsl(var(--foreground))",
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
      },
      fontFamily: {
        sans: ['Inter Tight', 'sans-serif'],
      },
      borderRadius: {
        lg: "0.5rem",
        md: "calc(0.5rem - 2px)",
        sm: "calc(0.5rem - 4px)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

export default config;
```

### 12.2 globals.css

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 210 40% 13%;
    --foreground: 0 0% 95%;
    --card: 210 40% 10%;
    --primary: 250 77% 58%;
    --secondary: 173 63% 53%;
    --accent: 21 81% 65%;
    --muted: 210 40% 20%;
    --muted-foreground: 0 0% 63.9%;
    --border: 210 40% 20%;
    --radius: 0.5rem;
  }
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
    font-family: 'Inter Tight', sans-serif;
  }
}

@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&display=swap');

.container-max {
  max-width: 800px;
  margin: 0 auto;
  padding: 0 16px;
}
```

---

## 13. 참고사항

### 13.1 필수 패키지

```bash
npm install tailwindcss postcss autoprefixer
npm install tailwindcss-animate
npm install lucide-react
npm install next-intl  # 다국어 지원 시
```

### 13.2 디자인 원칙

1. **다크 테마 기본** - 어두운 배경에 밝은 텍스트
2. **일관된 둥근 모서리** - `rounded-lg` (8px) 또는 `rounded-xl` (12px)
3. **미묘한 테두리** - `border border-border`로 요소 구분
4. **호버 피드백** - 모든 인터랙티브 요소에 호버 효과
5. **반응형 우선** - 모바일부터 설계 후 데스크탑 확장

### 13.3 색상 조합 가이드

| 용도 | 배경 | 텍스트 | 테두리 |
|------|------|--------|--------|
| 페이지 | `bg-background` | `text-foreground` | - |
| 카드 | `bg-surface` | `text-foreground` | `border-border` |
| Primary 버튼 | `bg-primary` | `text-white` | - |
| CTA 버튼 | `bg-accent` | `text-white` | - |
| 입력 필드 | `bg-surface` | `text-foreground` | `border-border` |
| 비활성 텍스트 | - | `text-muted-foreground` | - |
