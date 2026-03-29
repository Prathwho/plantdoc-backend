import { useState } from 'react'
import axios from 'axios'

function Chat() {
  const [messages, setMessages] = useState([
    { role: 'bot', text: "Hi! I'm PlantDoc 👋 Describe your plant's symptoms and I'll help you out!" }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  const quickMessages = [
    'My plant has yellow leaves',
    'How to water succulents?',
    'My plant has brown spots',
    'Why is my plant wilting?'
  ]

  const sendMessage = async (text) => {
    const msg = text || input
    if (!msg.trim()) return
    setMessages(prev => [...prev, { role: 'user', text: msg }])
    setInput('')
    setLoading(true)
    try {
      const res = await axios.post('http://localhost:8000/chat', { message: msg })
      setMessages(prev => [...prev, { role: 'bot', text: res.data.response }])
    } catch {
      setMessages(prev => [...prev, { role: 'bot', text: 'Something went wrong. Please try again.' }])
    }
    setLoading(false)
  }

  return (
    <div className="screen chat-screen">
      <div className="chat-header">
        <div className="chat-avatar">🌿</div>
        <div>
          <h2>PlantDoc AI</h2>
          <p>Expert plant doctor</p>
        </div>
        <div className="online-dot"></div>
      </div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg ${msg.role}`}>
            <div className={`chat-av ${msg.role}`}>{msg.role === 'bot' ? '🌿' : 'You'}</div>
            <div className="chat-bubble">{msg.text}</div>
          </div>
        ))}
        {loading && (
          <div className="chat-msg bot">
            <div className="chat-av bot">🌿</div>
            <div className="chat-bubble typing">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}
      </div>

      <div className="suggestions">
        {quickMessages.map((q, i) => (
          <button key={i} className="sug-chip" onClick={() => sendMessage(q)}>{q}</button>
        ))}
      </div>

      <div className="chat-input-row">
        <input
          className="chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
          placeholder="Describe your plant issue..."
        />
        <button className="send-btn" onClick={() => sendMessage()} disabled={loading}>➤</button>
      </div>
    </div>
  )
}
export default Chat