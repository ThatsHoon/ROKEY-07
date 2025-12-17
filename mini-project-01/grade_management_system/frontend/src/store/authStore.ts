import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { User, Token } from '../types'

interface AuthState {
  user: User | null
  token: Token | null
  isAuthenticated: boolean
  setAuth: (user: User, token: Token) => void
  logout: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,

      setAuth: (user: User, token: Token) => {
        set({ user, token, isAuthenticated: true })
      },

      logout: () => {
        set({ user: null, token: null, isAuthenticated: false })
      },
    }),
    {
      name: 'auth-storage',
    }
  )
)
