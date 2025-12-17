/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#4B39EF',
          50: '#E8E5FC',
          100: '#D1CBF9',
          200: '#A397F3',
          300: '#7563ED',
          400: '#4B39EF',
          500: '#3520E3',
          600: '#2819B5',
          700: '#1E1287',
          800: '#140C59',
          900: '#0A062B',
        },
        secondary: {
          DEFAULT: '#39D2C0',
          50: '#E5F9F6',
          100: '#CCF3ED',
          200: '#99E7DB',
          300: '#66DBC9',
          400: '#39D2C0',
          500: '#2DB5A5',
        },
        accent: {
          DEFAULT: '#EE8B60',
          50: '#FDF3EE',
          100: '#FCE7DD',
          200: '#F9CFBB',
          300: '#F6B799',
          400: '#F3A07C',
          500: '#EE8B60',
        },
        background: '#1D2428',
        surface: '#14181B',
        border: '#2A3137',
      },
      fontFamily: {
        sans: ['Inter Tight', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
