import { useState, useEffect } from 'react'
import { supabase } from '../supabase'

function Reminders({ user }) {
  const [reminders, setReminders] = useState([])
  const [showForm, setShowForm] = useState(false)
  const [newPlant, setNewPlant] = useState('')
  const [newCare, setNewCare] = useState('💧 Watering')
  const [newFreq, setNewFreq] = useState('Every 3 days')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchReminders()
  }, [])

  const fetchReminders = async () => {
    setLoading(true)
    const { data, error } = await supabase
      .from('reminders')
      .select('*')
      .eq('user_id', user.id)
      .order('created_at', { ascending: false })
    if (!error) setReminders(data || [])
    setLoading(false)
  }

  const addReminder = async () => {
    if (!newPlant.trim()) return
    const newItem = {
      user_id: user.id,
      plant_name: newPlant,
      care_type: newCare,
      frequency: newFreq,
      done: false
    }
    const { data, error } = await supabase
      .from('reminders')
      .insert([newItem])
      .select()
    if (!error && data) {
      setReminders(prev => [data[0], ...prev])
      setNewPlant('')
      setShowForm(false)
    }
  }

  const toggleDone = async (id, currentDone) => {
    await supabase
      .from('reminders')
      .update({ done: !currentDone })
      .eq('id', id)
    setReminders(prev =>
      prev.map(r => r.id === id ? { ...r, done: !currentDone } : r)
    )
  }

  const deleteReminder = async (id) => {
    await supabase.from('reminders').delete().eq('id', id)
    setReminders(prev => prev.filter(r => r.id !== id))
  }

  return (
    <div className="screen">
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>⏰ Care Reminders</h2>
        <button className="add-btn" onClick={() => setShowForm(!showForm)}>+ Add</button>
      </div>
      <div className="page-body">

        {showForm && (
          <div className="add-form">
            <input
              placeholder="Plant name e.g. My Monstera"
              value={newPlant}
              onChange={e => setNewPlant(e.target.value)}
            />
            <select value={newCare} onChange={e => setNewCare(e.target.value)}>
              <option>💧 Watering</option>
              <option>🌱 Fertilizing</option>
              <option>☀️ Rotating</option>
              <option>✂️ Pruning</option>
            </select>
            <select value={newFreq} onChange={e => setNewFreq(e.target.value)}>
              <option>Every day</option>
              <option>Every 3 days</option>
              <option>Weekly</option>
              <option>Every 2 weeks</option>
              <option>Monthly</option>
            </select>
            <button className="save-btn" onClick={addReminder}>Save Reminder</button>
          </div>
        )}

        {loading && <div className="loading-card">Loading your reminders...</div>}

        {!loading && reminders.length === 0 && (
          <div className="empty-state">
            <div style={{ fontSize: 40, marginBottom: 12 }}>⏰</div>
            <p style={{ fontWeight: 500, marginBottom: 6 }}>No reminders yet!</p>
            <p style={{ fontSize: 13, color: '#888' }}>Tap "+ Add" to create your first plant care reminder</p>
          </div>
        )}

        {reminders.map((r) => (
          <div key={r.id} className={`rem-item ${r.done ? 'done' : ''}`}>
            <div className="rem-circle">{r.care_type.split(' ')[0]}</div>
            <div className="rem-info">
              <p>{r.care_type.split(' ')[1]} {r.plant_name}</p>
              <span>{r.frequency}</span>
            </div>
            <div
              className={`check-btn ${r.done ? 'checked' : ''}`}
              onClick={() => toggleDone(r.id, r.done)}
            >
              {r.done ? '✓' : ''}
            </div>
            <div
              style={{ marginLeft: 8, cursor: 'pointer', fontSize: 16, color: '#ccc' }}
              onClick={() => deleteReminder(r.id)}
            >
              ✕
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
export default Reminders