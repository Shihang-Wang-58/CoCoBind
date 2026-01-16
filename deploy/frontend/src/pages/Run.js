import React, { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import './Run.css';

const Run = () => {
  const [rnaSequence, setRnaSequence] = useState('');
  const [smiles, setSmiles] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [jobId, setJobId] = useState(null);

  const examples = {
    rna: 'AUGGCCUUACGAUCGAUACGUAUCGAUCGUACGUACGUACGAUCGUACGUA',
    smiles: 'CCO'
  };

  const loadExample = () => {
    setRnaSequence(examples.rna);
    setSmiles(examples.smiles);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!rnaSequence.trim() || !smiles.trim()) {
      setError('Please provide both RNA sequence and SMILES');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('http://localhost:8000/predict', {
        rna_sequence: rnaSequence,
        smiles: smiles
      });
      
      setResult(response.data);
      setJobId(Date.now().toString());
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Prediction failed. Please check your input and ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const clearForm = () => {
    setRnaSequence('');
    setSmiles('');
    setResult(null);
    setError(null);
    setJobId(null);
  };

  return (
    <div className="run-page">
      <div className="container">
        <div className="page-header">
          <h1>Run CoCoBind Prediction</h1>
          <p>Enter RNA sequence and small molecule SMILES to predict binding interaction and sites</p>
        </div>

        <div className="run-grid">
          {/* Input Section */}
          <div className="input-section">
            <div className="card">
              <h2>Input Data</h2>
              
              <button onClick={loadExample} className="example-btn">
                Load Example
              </button>

              <form onSubmit={handleSubmit}>
                <div className="form-group">
                  <label>
                    RNA Sequence
                    <span className="required">*</span>
                  </label>
                  <textarea
                    value={rnaSequence}
                    onChange={(e) => setRnaSequence(e.target.value)}
                    placeholder="Enter RNA sequence (A, U, G, C)..."
                    rows={6}
                  />
                  <small>{rnaSequence.length} nucleotides</small>
                </div>

                <div className="form-group">
                  <label>
                    Small Molecule SMILES
                    <span className="required">*</span>
                  </label>
                  <input
                    type="text"
                    value={smiles}
                    onChange={(e) => setSmiles(e.target.value)}
                    placeholder="e.g., CCO (ethanol)"
                  />
                </div>

                <div className="form-actions">
                  <button 
                    type="submit" 
                    className="btn btn-primary"
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <span className="spinner"></span>
                        Predicting...
                      </>
                    ) : (
                      <>
                        🚀 Run Prediction
                      </>
                    )}
                  </button>
                  <button 
                    type="button" 
                    className="btn btn-secondary"
                    onClick={clearForm}
                  >
                    Clear
                  </button>
                </div>
              </form>

              {error && (
                <div className="alert alert-error">
                  <strong>Error:</strong> {error}
                </div>
              )}
            </div>
          </div>

          {/* Results Section */}
          <div className="results-section">
            <div className="card">
              <h2>Results</h2>
              
              {!result && !loading && (
                <div className="empty-state">
                  <div className="empty-icon">📊</div>
                  <p>Results will appear here after prediction</p>
                </div>
              )}

              {loading && (
                <div className="loading-state">
                  <div className="loader-big"></div>
                  <p>Analyzing binding interaction...</p>
                </div>
              )}

              {result && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="results-content"
                >
                  {jobId && (
                    <div className="job-info">
                      <span>Job ID: <code>{jobId}</code></span>
                    </div>
                  )}

                  <div className="result-main">
                    <div className="result-score">
                      <span className="score-label">Interaction Probability</span>
                      <span className={`score-value ${result.interaction_probability > 0.5 ? 'positive' : 'negative'}`}>
                        {(result.interaction_probability * 100).toFixed(2)}%
                      </span>
                      <span className={`score-badge ${result.interaction_probability > 0.5 ? 'positive' : 'negative'}`}>
                        {result.interaction_probability > 0.5 ? '✓ Active Binding' : '✗ No Binding'}
                      </span>
                    </div>
                  </div>

                  <div className="binding-sites">
                    <h3>Binding Site Probabilities</h3>
                    <div className="sites-table-container">
                      <table className="sites-table">
                        <thead>
                          <tr>
                            <th>Position</th>
                            <th>Residue</th>
                            <th>Probability</th>
                            <th>Visualization</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.binding_sites.map((prob, idx) => (
                            <tr key={idx} className={prob > 0.5 ? 'high-prob' : ''}>
                              <td>{idx + 1}</td>
                              <td className="residue-cell">{rnaSequence[idx] || '-'}</td>
                              <td>{(prob * 100).toFixed(1)}%</td>
                              <td>
                                <div className="prob-bar-container">
                                  <div 
                                    className="prob-bar" 
                                    style={{ 
                                      width: `${prob * 100}%`,
                                      background: prob > 0.5 ? 'linear-gradient(90deg, #34a853, #2e7d32)' : 'linear-gradient(90deg, #93c5fd, #60a5fa)'
                                    }}
                                  ></div>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="result-actions">
                    <button className="btn btn-secondary">
                      💾 Download Results
                    </button>
                    <button className="btn btn-secondary">
                      📊 Visualize Structure
                    </button>
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Run;
