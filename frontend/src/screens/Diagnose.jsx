import { useState } from 'react'
import axios from 'axios'

const steps = [
  {
    id: 'plant_type',
    question: '🌿 What type of plant is it?',
    type: 'grid',
    options: ['Rose', 'Tomato', 'Monstera', 'Cactus', 'Mango', 'Banana', 'Basil', 'Other']
  },
  {
    id: 'symptoms',
    question: '🔍 What symptoms do you see?',
    type: 'multi',
    options: ['Yellow leaves', 'Brown spots', 'Wilting', 'White powder', 'Black spots', 'Dropping leaves', 'Stunted growth', 'Rotting stem']
  },
  {
    id: 'watering',
    question: '💧 How often do you water it?',
    type: 'single',
    options: ['Every day', 'Every 2-3 days', 'Once a week', 'Rarely']
  },
  {
    id: 'sunlight',
    question: '☀️ How much sunlight does it get?',
    type: 'single',
    options: ['Full sun (6+ hrs)', 'Partial sun (3-6 hrs)', 'Low light (< 3 hrs)', 'Indoors, no direct sun']
  },
  {
    id: 'location',
    question: '🏠 Where is your plant kept?',
    type: 'single',
    options: ['Indoors', 'Outdoors', 'Balcony', 'Greenhouse']
  },
  {
    id: 'duration',
    question: '⏳ How long have you noticed these symptoms?',
    type: 'single',
    options: ['Just today', 'A few days', 'About a week', 'More than a week']
  }
]

function Diagnose() {
  const [currentStep, setCurrentStep] = useState(0)
  const [answers, setAnswers] = useState({})
  const [selectedOptions, setSelectedOptions] = useState([])
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [done, setDone] = useState(false)

  const step = steps[currentStep]

  const toggleOption = (option) => {
    if (step.type === 'single') {
      setSelectedOptions([option])
    } else {
      setSelectedOptions(prev =>
        prev.includes(option)
          ? prev.filter(o => o !== option)
          : [...prev, option]
      )
    }
  }

  const handleNext = async () => {
    if (selectedOptions.length === 0) return
    const newAnswers = { ...answers, [step.id]: selectedOptions }
    setAnswers(newAnswers)
    setSelectedOptions([])

    if (currentStep < steps.length - 1) {
      setCurrentStep(prev => prev + 1)
    } else {
      setDone(true)
      await generateReport(newAnswers)
    }
  }

  const generateReport = async (data) => {
    setLoading(true)
    const prompt = `
You are PlantDoc, an expert plant doctor. A user has described their plant problem:

Plant type: ${data.plant_type?.join(', ')}
Symptoms: ${data.symptoms?.join(', ')}
Watering frequency: ${data.watering?.join(', ')}
Sunlight: ${data.sunlight?.join(', ')}
Location: ${data.location?.join(', ')}
Duration of symptoms: ${data.duration?.join(', ')}

Based on this information, provide a detailed plant health report with:

1. 🌿 DIAGNOSIS - What is likely wrong with the plant
2. 🔍 CAUSE - Why this is happening based on their conditions
3. ⚠️ SEVERITY - Rate as Mild / Moderate / Severe with brief reason
4. 💊 TREATMENT PLAN - 3 specific steps to fix the problem
5. 🛡️ PREVENTION - How to prevent this in future
6. 💬 WATCH OUT FOR - One warning sign to monitor

Be specific, warm, and encouraging. Use simple language.`

    try {
      const res = await axios.post('https://plantdoc-backend-2.onrender.com/chat', { message: prompt })
      setReport(res.data.response)
    } catch {
      setReport('Something went wrong. Please try again.')
    }
    setLoading(false)
  }

  const resetChecker = () => {
    setCurrentStep(0)
    setAnswers({})
    setSelectedOptions([])
    setReport(null)
    setDone(false)
  }

  const progress = ((currentStep) / steps.length) * 100

  return (
    <div className="screen">
      <div className="page-header">
        <h2>🚑 Symptom Checker</h2>
      </div>
      <div className="page-body">

        {!done ? (
          <>
            {/* Progress bar */}
            <div className="progress-bar-container">
              <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
            </div>
            <p className="progress-text">Step {currentStep + 1} of {steps.length}</p>

            {/* Question */}
            <div className="question-card">
              <h3 className="question-text">{step.question}</h3>
              {step.type === 'multi' && (
                <p className="question-sub">Select all that apply</p>
              )}
            </div>

            {/* Options */}
            <div className={step.type === 'grid' ? 'options-grid' : 'options-list'}>
              {step.options.map((option, i) => (
                <div
                  key={i}
                  className={`option-item ${selectedOptions.includes(option) ? 'selected' : ''}`}
                  onClick={() => toggleOption(option)}
                >
                  {option}
                </div>
              ))}
            </div>

            {/* Next button */}
            <button
              className="diagnose-btn"
              onClick={handleNext}
              disabled={selectedOptions.length === 0}
            >
              {currentStep === steps.length - 1 ? '🔬 Generate Report' : 'Next →'}
            </button>
          </>
        ) : (
          <>
            {loading ? (
              <div className="report-loading">
                <div className="report-loading-icon">🔬</div>
                <p>Analyzing your plant's condition...</p>
                <p className="report-loading-sub">PlantDoc is generating your diagnosis report</p>
              </div>
            ) : (
              <>
                <div className="report-card">
                  <div className="report-header">
                    <span style={{ fontSize: 28 }}>📋</span>
                    <div>
                      <h3>Diagnosis Report</h3>
                      <p>{answers.plant_type?.join(', ')} · {answers.symptoms?.join(', ')}</p>
                    </div>
                  </div>
                  <div className="report-body">
                    <p>{report}</p>
                  </div>
                </div>
                <button className="diagnose-btn" onClick={resetChecker}>
                  🔄 Check Another Plant
                </button>
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}
export default Diagnose