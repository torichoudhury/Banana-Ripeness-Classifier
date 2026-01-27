function ModelStats({ predictions }) {
  const modelAgreement = () => {
    const predictedClasses = Object.values(predictions).map(p => p.predicted_class)
    const uniqueClasses = new Set(predictedClasses)
    return uniqueClasses.size === 1
  }

  const averageConfidence = () => {
    const confidences = Object.values(predictions).map(p => p.confidence)
    const avg = confidences.reduce((a, b) => a + b, 0) / confidences.length
    return (avg * 100).toFixed(2)
  }

  const getMostConfidentModel = () => {
    let maxConfidence = 0
    let modelName = ''

    Object.entries(predictions).forEach(([name, result]) => {
      if (result.confidence > maxConfidence) {
        maxConfidence = result.confidence
        modelName = name
      }
    })

    return { name: modelName, confidence: (maxConfidence * 100).toFixed(2) }
  }

  const getConsensusClass = () => {
    const classes = Object.values(predictions).map(p => p.predicted_class)
    const counts = {}

    classes.forEach(cls => {
      counts[cls] = (counts[cls] || 0) + 1
    })

    const maxCount = Math.max(...Object.values(counts))
    const consensusClass = Object.entries(counts).find(([, count]) => count === maxCount)?.[0]

    return { class: consensusClass, count: maxCount, total: classes.length }
  }

  const agreement = modelAgreement()
  const avgConf = averageConfidence()
  const mostConfident = getMostConfidentModel()
  const consensus = getConsensusClass()

  return (
    <div className="stats-section">
      <h2 className="section-title">Model Statistics & Analysis</h2>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">{agreement ? '✅' : '⚠️'}</div>
          <div className="stat-content">
            <div className="stat-label">Model Agreement</div>
            <div className="stat-value">
              {agreement ? 'All models agree' : 'Models disagree'}
            </div>
            <div className="stat-detail">
              {consensus.count} of {consensus.total} models predict: {consensus.class}
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">📊</div>
          <div className="stat-content">
            <div className="stat-label">Average Confidence</div>
            <div className="stat-value">{avgConf}%</div>
            <div className="stat-detail">Across all models</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🏆</div>
          <div className="stat-content">
            <div className="stat-label">Most Confident</div>
            <div className="stat-value">{mostConfident.name}</div>
            <div className="stat-detail">{mostConfident.confidence}% confidence</div>
          </div>
        </div>
      </div>

      <div className="comparison-card">
        <h3 className="comparison-title">Model Comparison</h3>
        <div className="comparison-table">
          <div className="table-header">
            <div className="table-cell">Model</div>
            <div className="table-cell">Prediction</div>
            <div className="table-cell">Confidence</div>
            <div className="table-cell">Status</div>
          </div>
          {Object.entries(predictions)
            .sort(([, a], [, b]) => b.confidence - a.confidence)
            .map(([modelName, result]) => (
              <div key={modelName} className="table-row">
                <div className="table-cell model-cell">{modelName}</div>
                <div className="table-cell">{result.predicted_class}</div>
                <div className="table-cell confidence-cell">
                  {(result.confidence * 100).toFixed(2)}%
                </div>
                <div className="table-cell">
                  {result.predicted_class === consensus.class ? (
                    <span className="badge badge-success">Consensus</span>
                  ) : (
                    <span className="badge badge-warning">Different</span>
                  )}
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  )
}

export default ModelStats
