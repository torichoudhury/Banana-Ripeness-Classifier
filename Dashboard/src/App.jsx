import { useState } from 'react'
import axios from 'axios'
import ImageUpload from './components/ImageUpload'
import PredictionResults from './components/PredictionResults'
import ModelStats from './components/ModelStats'
import './App.css'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleImageSelect = (file) => {
    setSelectedImage(file)
    setImagePreview(URL.createObjectURL(file))
    setPredictions(null)
    setError(null)
  }

  const handlePredict = async () => {
    if (!selectedImage) {
      setError('Please select an image first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedImage)

      const response = await axios.post('/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setPredictions(response.data.predictions)
    } catch (err) {
      setError(err.response?.data?.detail || 'Error processing image')
      console.error('Prediction error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedImage(null)
    setImagePreview(null)
    setPredictions(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1 className="title">
            <span className="icon">🍌</span>
            Banana Ripeness Classifier
          </h1>
          <p className="subtitle">
            Upload a banana image to classify its ripeness using three different AI models
          </p>
        </div>
      </header>

      <main className="main-content">
        <div className="container">
          <ImageUpload
            onImageSelect={handleImageSelect}
            imagePreview={imagePreview}
            onPredict={handlePredict}
            onReset={handleReset}
            loading={loading}
            hasImage={!!selectedImage}
          />

          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}

          {predictions && (
            <>
              <PredictionResults predictions={predictions} />
              <ModelStats predictions={predictions} />
            </>
          )}
        </div>
      </main>

      <footer className="footer">
        <p>Powered by PyTorch • EfficientNet-B0 • ResNet50 • MobileNetV2</p>
      </footer>
    </div>
  )
}

export default App
