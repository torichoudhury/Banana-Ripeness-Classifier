function PredictionResults({ predictions }) {
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#10b981'
    if (confidence >= 0.6) return '#f59e0b'
    return '#ef4444'
  }

  const getRipenessEmoji = (className) => {
    const lower = className.toLowerCase()
    if (lower.includes('green') || lower.includes('unripe')) return '🟢'
    if (lower.includes('yellow') || lower.includes('ripe')) return '🟡'
    if (lower.includes('brown') || lower.includes('overripe')) return '🟤'
    return '🍌'
  }

  const modelIcons = {
    'EfficientNet-B0': '⚡',
    'ResNet50': '🎯',
    'MobileNetV2': '📱'
  }

  return (
    <div className="results-section">
      <h2 className="section-title">Classification Results</h2>
      <div className="results-grid">
        {Object.entries(predictions).map(([modelName, result]) => (
          <div key={modelName} className="result-card">
            <div className="model-header">
              <span className="model-icon">{modelIcons[modelName]}</span>
              <h3 className="model-name">{modelName}</h3>
            </div>

            <div className="prediction-main">
              <div className="predicted-class">
                <span className="class-emoji">
                  {getRipenessEmoji(result.predicted_class)}
                </span>
                <span className="class-name">{result.predicted_class}</span>
              </div>

              <div className="confidence-display">
                <div className="confidence-label">Confidence</div>
                <div
                  className="confidence-value"
                  style={{ color: getConfidenceColor(result.confidence) }}
                >
                  {(result.confidence * 100).toFixed(2)}%
                </div>
              </div>
            </div>

            <div className="confidence-bar-container">
              <div
                className="confidence-bar"
                style={{
                  width: `${result.confidence * 100}%`,
                  backgroundColor: getConfidenceColor(result.confidence)
                }}
              />
            </div>

            <div className="all-predictions">
              <div className="predictions-header">All Class Probabilities</div>
              {Object.entries(result.all_confidences)
                .sort(([, a], [, b]) => b - a)
                .map(([className, confidence]) => (
                  <div key={className} className="prediction-row">
                    <span className="class-label">
                      {getRipenessEmoji(className)} {className}
                    </span>
                    <div className="prediction-bar-container">
                      <div
                        className="prediction-bar"
                        style={{
                          width: `${confidence * 100}%`,
                          backgroundColor: className === result.predicted_class
                            ? getConfidenceColor(confidence)
                            : '#94a3b8'
                        }}
                      />
                      <span className="prediction-value">
                        {(confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default PredictionResults
