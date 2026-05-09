/*
 * AuthContext — global authentication state for the app.
 *
 * What it does:
 *   - Stores the current clinician (user object) and their JWT token
 *   - Exposes login() and logout() actions
 *   - Persists auth to localStorage so sessions survive page refresh
 *
 * HOW TO CONNECT TO BACKEND:
 *   The login() function already calls api/index.js → login(), which is
 *   stubbed to mock data now. When your Flask backend is ready, update
 *   src/api/index.js login() to hit POST /api/auth/login.
 *   The token from the response should be sent as:
 *     Authorization: Bearer <token>
 *   Add that header in api/index.js for all authenticated requests.
 */

import { createContext, useContext, useState, useEffect } from 'react'
import { login as apiLogin } from '../api/index.js'

// The context value shape: { user, token, isAuthenticated, login, logout }
const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser]   = useState(null)
  const [token, setToken] = useState(null)

  // Rehydrate from localStorage on app load
  useEffect(() => {
    const storedToken = localStorage.getItem('ll_token')
    const storedUser  = localStorage.getItem('ll_user')
    if (storedToken && storedUser) {
      setToken(storedToken)
      setUser(JSON.parse(storedUser))
    }
  }, [])

  // login — calls the API, stores results, returns to caller
  async function login(email, password) {
    const { token: newToken, user: newUser } = await apiLogin(email, password)
    setToken(newToken)
    setUser(newUser)
    localStorage.setItem('ll_token', newToken)
    localStorage.setItem('ll_user', JSON.stringify(newUser))
    return newUser
  }

  // logout — clears all state and localStorage
  function logout() {
    setToken(null)
    setUser(null)
    localStorage.removeItem('ll_token')
    localStorage.removeItem('ll_user')
  }

  return (
    <AuthContext.Provider value={{ user, token, isAuthenticated: !!token, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

// Convenience hook — use in any component: const { user, login } = useAuth()
export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
