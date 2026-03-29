import { useState, useEffect } from 'react'
import { supabase } from './supabase'
import Home from './screens/Home'
import Chat from './screens/Chat'
import Identify from './screens/Identify'
import Reminders from './screens/Reminders'
import Login from './screens/Login'
import Diagnose from './screens/Diagnose'
import './App.css'

function App() {
  const [screen, setScreen] = useState('home')
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null)
      setLoading(false)
    })
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null)
    })
    return () => subscription.unsubscribe()
  }, [])

  const handleLogout = async () => {
    await supabase.auth.signOut()
    setScreen('home')
  }

  if (loading) return <div className="loading-screen">🌿</div>
  if (!user) return <div className="app-container"><div className="app"><Login onLogin={() => {}} /></div></div>

  const screens = {
    home: <Home setScreen={setScreen} user={user} onLogout={handleLogout} />,
    chat: <Chat />,
    identify: <Identify />,
    diagnose: <Diagnose />,
    reminders: <Reminders user={user} />
  }

  return (
    <div className="app-container">
      <div className="app">
        {screens[screen]}
        <div className="bottom-nav">
          {[
            { id: 'home', icon: '🏠', label: 'Home' },
            { id: 'chat', icon: '💬', label: 'PlantDoc' },
            { id: 'identify', icon: '🔍', label: 'Identify' },
            { id: 'diagnose', icon: '🚑', label: 'Diagnose' },
            { id: 'reminders', icon: '⏰', label: 'Reminders' },
          ].map(nav => (
            <button key={nav.id} className={`nav-item ${screen === nav.id ? 'active' : ''}`} onClick={() => setScreen(nav.id)}>
              <span className="nav-icon">{nav.icon}</span>
              <span className="nav-label">{nav.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
export default App