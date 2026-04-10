import { useState, useRef } from 'react'
import axios from 'axios'

function Identify() {
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [imageResult, setImageResult] = useState(null)
  const [imageLoading, setImageLoading] = useState(false)
  const [imageError, setImageError] = useState('')
  const fileInputRef = useRef(null)

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (!file) return
    setImage(file)
    setImagePreview(URL.createObjectURL(file))
    setImageResult(null)
    setImageError('')
  }

  const analyzeImage = async () => {
    if (!image) return
    setImageLoading(true)
    setImageError('')
    setImageResult(null)
    try {
      const formData = new FormData()
      formData.append('file', image)
      const res = await axios.post('https://plantdoc-backend-2.onrender.com/identify-image', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      if (res.data.error) setImageError(res.data.error)
      else setImageResult(res.data)
    } catch {
      setImageError('Failed to analyze. Make sure your backend is running.')
    }
    setImageLoading(false)
  }

  const resetImage = () => {
    setImage(null)
    setImagePreview(null)
    setImageResult(null)
    setImageError('')
  }

  return (
    <div className="screen">
      <div className="page-header">
        <h2>🌿 Identify Plant</h2>
      </div>

      <div className="page-body">

        {/* UPLOAD ZONE */}
        <p className="section-title">Upload a plant photo 📷</p>

        <div className="upload-zone" onClick={() => fileInputRef.current.click()}>
          {imagePreview
            ? <img src={imagePreview} alt="Plant" className="upload-preview" />
            : <>
                <div className="upload-icon">📷</div>
                <p className="upload-text">Tap to upload a plant photo</p>
                <p className="upload-sub">Our ML model will identify it</p>
              </>
          }
        </div>

        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          onChange={handleImageChange}
          style={{ display: 'none' }}
        />

        {/* BUTTONS */}
        {imagePreview && (
          <div style={{ display: 'flex', gap: '8px', marginBottom: '4px' }}>
            <button
              className="diagnose-btn"
              onClick={analyzeImage}
              disabled={imageLoading}
              style={{ flex: 1, marginBottom: 0 }}
            >
              {imageLoading ? '🔬 Analyzing...' : '🔍 Identify This Plant'}
            </button>
            <button
              onClick={resetImage}
              style={{
                padding: '0 16px', border: '0.5px solid #ccc',
                borderRadius: '12px', background: 'white',
                fontSize: '13px', cursor: 'pointer',
                fontFamily: 'DM Sans, sans-serif', color: '#666'
              }}
            >
              ✕
            </button>
          </div>
        )}

        {/* LOADING */}
        {imageLoading && (
          <div className="loading-card" style={{ marginTop: '12px' }}>
            🧠 Running ML model on your image...
          </div>
        )}

        {/* ERROR */}
        {imageError && (
          <div className="error-card" style={{ marginTop: '12px' }}>
            {imageError}
          </div>
        )}

        {/* RESULT */}
        {imageResult && (
          <div className="result-card" style={{ marginTop: '12px' }}>
            {imageResult.is_plant === false ? (

              // NOT A PLANT — show rejection message only
              <div style={{ textAlign: 'center', padding: '10px' }}>
                <span style={{ fontSize: '40px' }}>⚠️</span>
                <p style={{
                  marginTop: '12px', fontSize: '14px',
                  color: '#A32D2D', lineHeight: '1.8', fontWeight: '500'
                }}>
                  PlantDoc could not identify this as a plant leaf.
                </p>
                <p style={{
                  marginTop: '8px', fontSize: '13px',
                  color: '#666', lineHeight: '1.8'
                }}>
                  Please upload a clear close-up photo of a plant leaf only.
                  Screenshots, objects, animals, and humans are not supported by PlantDoc.
                </p>
                <button
                  onClick={resetImage}
                  style={{
                    marginTop: '16px', width: '100%', padding: '10px',
                    background: '#FCEBEB', border: '0.5px solid #A32D2D',
                    borderRadius: '10px', color: '#A32D2D', fontSize: '13px',
                    cursor: 'pointer', fontFamily: 'DM Sans, sans-serif', fontWeight: '500'
                  }}
                >
                  📷 Try Again with a Plant Photo
                </button>
              </div>

            ) : (

              // IS A PLANT — show full result
              <>
                <div className="result-header">
                  <span className="result-emoji">🌿</span>
                  <div>
                    <h3 className="result-disease">
                      {imageResult.ml_result?.top_prediction || 'Plant Identified'}
                    </h3>
                    <p className="result-label">
                      ML Confidence: {imageResult.ml_result?.confidence || 0}%
                    </p>
                  </div>
                </div>

                {/* Confidence Bar */}
                <div className="confidence-section" style={{ marginBottom: '12px' }}>
                  <div className="confidence-bar">
                    <div className="confidence-fill" style={{
                      width: `${imageResult.ml_result?.confidence || 0}%`,
                      background: imageResult.ml_result?.confidence > 70
                        ? '#1D9E75'
                        : imageResult.ml_result?.confidence > 40
                        ? '#f59e0b' : '#ef4444'
                    }} />
                  </div>
                  <div className="confidence-label" style={{ marginTop: '6px' }}>
                    <span>Other possibilities:</span>
                    <span>
                      {imageResult.ml_result?.all_predictions
                        ?.slice(1, 3).map(p => p.label).join(', ')}
                    </span>
                  </div>
                </div>

                <div className="advice-section">
                  <p className="advice-title">🌿 AI Care Guide</p>
                  <p className="advice-text">{imageResult.response}</p>
                </div>

                <button
                  onClick={resetImage}
                  style={{
                    marginTop: '12px', width: '100%', padding: '10px',
                    background: '#E1F5EE', border: '0.5px solid #5DCAA5',
                    borderRadius: '10px', color: '#0a5c44', fontSize: '13px',
                    cursor: 'pointer', fontFamily: 'DM Sans, sans-serif', fontWeight: '500'
                  }}
                >
                  📷 Identify Another Plant
                </button>
              </>
            )}
          </div>
        )}

        {/* EMPTY STATE HINT */}
        {!imagePreview && (
          <div style={{
            textAlign: 'center', marginTop: '20px',
            color: '#888', fontSize: '13px', lineHeight: '2.2'
          }}>
            📸 Take a clear photo of your plant<br />
            🧠 Our ML model identifies it<br />
            🌿 Get full care guide instantly
          </div>
        )}

      </div>
    </div>
  )
}

export default Identify