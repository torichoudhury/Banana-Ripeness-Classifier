import { useRef } from 'react'

function ImageUpload({ onImageSelect, imagePreview, onPredict, onReset, loading, hasImage }) {
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file && file.type.startsWith('image/')) {
      onImageSelect(file)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      onImageSelect(file)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  return (
    <div className="upload-section">
      <div className="upload-card">
        {!imagePreview ? (
          <div
            className="upload-area"
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <div className="upload-icon">📤</div>
            <h3>Upload Banana Image</h3>
            <p>Click to browse or drag and drop</p>
            <p className="file-hint">Supports JPG, PNG, WEBP</p>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
          </div>
        ) : (
          <div className="preview-container">
            <img src={imagePreview} alt="Preview" className="image-preview" />
            <div className="button-group">
              <button
                className="btn btn-primary"
                onClick={onPredict}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <span>🔍</span>
                    Classify Ripeness
                  </>
                )}
              </button>
              <button
                className="btn btn-secondary"
                onClick={onReset}
                disabled={loading}
              >
                <span>🔄</span>
                Upload New Image
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default ImageUpload
