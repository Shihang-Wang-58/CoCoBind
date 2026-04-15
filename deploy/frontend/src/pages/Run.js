import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import './Run.css';

const API_BASE = '/cocobind/api';

// Validation helpers
const isValidRNA = (seq) => /^[AUGCaugc]+$/.test(seq.trim());
const isValidSMILES = (smi) => {
  const s = smi.trim();
  if (s.length === 0) return false;
  // Basic SMILES character check (letters, digits, common SMILES symbols)
  return /^[A-Za-z0-9@+\-\[\]\(\)\\\/=#$%.:~*]+$/.test(s);
};

const Run = () => {
  const [rnaSequence, setRnaSequence] = useState('');
  const [smiles, setSmiles] = useState('');
  const [result, setResult] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [screeningResult, setScreeningResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [validationErrors, setValidationErrors] = useState({});
  const [mode, setMode] = useState('single'); // 'single', 'batch', or 'screen'
  const [models, setModels] = useState([]);
  const [libraries, setLibraries] = useState([]);
  const [selectedModel, setSelectedModel] = useState('ECFP4');
  const [selectedLibrary, setSelectedLibrary] = useState('');
  const [topK, setTopK] = useState(20);
  const fileInputRef = useRef(null);

  const examples = {
    rna: 'GCUUUGAUUUGGUGAAAUUCCAAAACCGACAGUAGAGUCUGGAUGAGAGAAGAUUC',
    smiles: 'CCn1nc(C)c(CN2CCC(c3ccn[nH]3)C2)c1C'
  };

  const loadExample = () => {
    setRnaSequence(examples.rna);
    setSmiles(examples.smiles);
    setValidationErrors({});
  };

  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const [modelsRes, librariesRes] = await Promise.all([
          axios.get(`${API_BASE}/models`),
          axios.get(`${API_BASE}/libraries`)
        ]);
        const modelList = modelsRes.data || [];
        const libraryList = librariesRes.data || [];
        setModels(modelList);
        setLibraries(libraryList);
        const defaultModel = modelList.find(m => m.default) || modelList[0];
        if (defaultModel) setSelectedModel(defaultModel.id);
        const firstReadyLibrary = libraryList.find(lib => lib.ready) || libraryList[0];
        if (firstReadyLibrary) setSelectedLibrary(firstReadyLibrary.id);
      } catch (err) {
        console.error('Failed to load backend options:', err);
      }
    };
    fetchOptions();
  }, []);

  const validateInputs = () => {
    const errors = {};
    if (!rnaSequence.trim()) {
      errors.rna = 'RNA sequence is required';
    } else if (!isValidRNA(rnaSequence)) {
      errors.rna = 'Invalid RNA sequence. Only A, U, G, C characters are allowed.';
    } else if (rnaSequence.trim().length > 512) {
      errors.rna = 'RNA sequence too long (max 512 nucleotides)';
    }

    if (!smiles.trim()) {
      errors.smiles = 'SMILES string is required';
    } else if (!isValidSMILES(smiles)) {
      errors.smiles = 'Invalid SMILES format. Please check your input.';
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateInputs()) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setBatchResults(null);

    try {
      const response = await axios.post(`${API_BASE}/predict`, {
        rna_sequence: rnaSequence.trim().toUpperCase().replace(/T/g, 'U'),
        smiles: smiles.trim(),
        model_id: selectedModel
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      const detail = err.response?.data?.detail;
      if (typeof detail === 'string') {
        setError(detail);
      } else if (Array.isArray(detail)) {
        setError(detail.map(d => d.msg).join('; '));
      } else {
        setError('Prediction failed. Please check your input and ensure the backend is running.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleCSVUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
      setError('Please upload a CSV file (.csv)');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setBatchResults(null);
    setScreeningResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_id', selectedModel);

    try {
      const response = await axios.post(`${API_BASE}/predict_batch`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setBatchResults(response.data);
    } catch (err) {
      console.error(err);
      const detail = err.response?.data?.detail;
      setError(typeof detail === 'string' ? detail : 'Batch prediction failed. Ensure CSV has "sequence" and "smiles" columns.');
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const validateScreeningInputs = () => {
    const errors = {};
    if (!rnaSequence.trim()) {
      errors.rna = 'RNA sequence is required';
    } else if (!isValidRNA(rnaSequence)) {
      errors.rna = 'Invalid RNA sequence. Only A, U, G, C characters are allowed.';
    } else if (rnaSequence.trim().length > 512) {
      errors.rna = 'RNA sequence too long (max 512 nucleotides)';
    }
    if (!selectedLibrary) {
      errors.library = 'Choose a compound library';
    }
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleScreening = async (e) => {
    e.preventDefault();
    if (!validateScreeningInputs()) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setBatchResults(null);
    setScreeningResult(null);

    try {
      const libraryInfo = libraries.find(library => library.id === selectedLibrary);
      const response = await axios.post(`${API_BASE}/screen`, {
        rna_sequence: rnaSequence.trim().toUpperCase().replace(/T/g, 'U'),
        model_id: selectedModel,
        library_id: selectedLibrary,
        top_k: Number(topK) || 20,
        max_candidates: Math.min(libraryInfo?.n_rows || 20000, 200000)
      });
      setScreeningResult(response.data);
    } catch (err) {
      console.error(err);
      const detail = err.response?.data?.detail;
      setError(typeof detail === 'string' ? detail : 'Virtual screening failed. Check the selected model and library.');
    } finally {
      setLoading(false);
    }
  };

  const downloadResults = () => {
    if (screeningResult?.candidates?.length) {
      const headers = ['rank', 'title', 'smiles', 'interaction_prob', 'model_id', 'library_id'];
      const rows = screeningResult.candidates.map((r, idx) => [
        idx + 1,
        r.title || '',
        r.smiles || '',
        r.interaction_prob ?? '',
        screeningResult.model_id || '',
        screeningResult.library_id || ''
      ]);
      const csv = [headers.join(','), ...rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(','))].join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'cocobind_screening_candidates.csv';
      a.click();
      URL.revokeObjectURL(url);
      return;
    }

    const data = batchResults || (result ? [result] : []);
    if (data.length === 0) return;

    const headers = ['sequence', 'smiles', 'interaction_prob', 'confidence', 'noisy_or_prob', 'consistency_gap', 'site_probs'];
    const rows = data.map(r => [
      r.sequence || '',
      r.smiles || '',
      r.interaction_prob ?? '',
      r.confidence || '',
      r.noisy_or_prob ?? '',
      r.consistency_gap ?? '',
      JSON.stringify(r.site_probs || r.binding_sites || [])
    ]);

    const csv = [headers.join(','), ...rows.map(r => r.map(v => `"${v}"`).join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cocobind_predictions.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const clearForm = () => {
    setRnaSequence('');
    setSmiles('');
    setResult(null);
    setBatchResults(null);
    setScreeningResult(null);
    setError(null);
    setValidationErrors({});
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const renderSingleResult = (res) => {
    const prob = res.interaction_prob ?? res.interaction_probability;
    const sites = res.site_probs || res.binding_sites || [];
    const seq = res.sequence || rnaSequence;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="results-content"
      >
        {res.job_id && (
          <div className="job-info">
            <span>Job ID: <code>{res.job_id}</code></span>
          </div>
        )}

        <div className="result-main">
          <div className="result-score">
            <span className="score-label">Interaction Probability</span>
            <span className={`score-value ${prob > 0.5 ? 'positive' : 'negative'}`}>
              {(prob * 100).toFixed(2)}%
            </span>
            <span className={`score-badge ${prob > 0.5 ? 'positive' : 'negative'}`}>
              {prob > 0.5 ? '✓ Active Binding' : '✗ No Binding'}
            </span>
          </div>
        </div>

        {res.confidence && (
          <div className="meta-grid">
            <div className="meta-item">
              <span className="meta-label">Confidence</span>
              <span className={`meta-value confidence-${res.confidence}`}>{res.confidence.toUpperCase()}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">NoisyOR</span>
              <span className="meta-value">{(res.noisy_or_prob * 100).toFixed(2)}%</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Consistency Gap</span>
              <span className="meta-value">{res.consistency_gap?.toFixed(4)}</span>
            </div>
          </div>
        )}

        {res.top_sites && res.top_sites.length > 0 && (
          <div className="top-sites">
            <h3>Top Binding Sites</h3>
            <div className="top-sites-chips">
              {res.top_sites.map((s, i) => (
                <div key={i} className={`site-chip ${s.prob >= 0.5 ? 'hot' : ''}`}>
                  <span className="site-pos">Pos {s.pos}</span>
                  <span className="site-nt">{s.nt}</span>
                  <span className="site-prob">{(s.prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {sites.length > 0 && (
          <div className="binding-sites">
            <h3>All Binding Site Probabilities</h3>
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
                  {sites.map((p, idx) => (
                    <tr key={idx} className={p > 0.5 ? 'high-prob' : ''}>
                      <td>{idx + 1}</td>
                      <td className="residue-cell">{seq[idx] || '-'}</td>
                      <td>{(p * 100).toFixed(1)}%</td>
                      <td>
                        <div className="prob-bar-container">
                          <div
                            className="prob-bar"
                            style={{
                              width: `${p * 100}%`,
                              background: p > 0.5
                                ? 'linear-gradient(90deg, #34a853, #2e7d32)'
                                : 'linear-gradient(90deg, #93c5fd, #60a5fa)'
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
        )}

        <div className="result-actions">
          <button className="btn btn-secondary" onClick={downloadResults}>
            💾 Download Results
          </button>
        </div>
      </motion.div>
    );
  };

  const renderScreeningResult = (res) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="batch-results"
    >
      <div className="batch-summary">
        <div>
          <h3>Candidate Molecules</h3>
          <p className="summary-copy">
            {res.n_screened} screened from {res.library_id} with {res.model_id}
          </p>
        </div>
        <button className="btn btn-secondary" onClick={downloadResults}>
          Download CSV
        </button>
      </div>

      <div className="batch-table-container">
        <table className="sites-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Compound</th>
              <th>SMILES</th>
              <th>Binding Score</th>
            </tr>
          </thead>
          <tbody>
            {(res.candidates || []).map((candidate, idx) => (
              <tr key={`${candidate.smiles}-${idx}`}>
                <td>{idx + 1}</td>
                <td>{candidate.title || '-'}</td>
                <td className="smiles-cell">{candidate.smiles}</td>
                <td>{(candidate.interaction_prob * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </motion.div>
  );

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

              <div className="form-group">
                <label>Model Version</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  {models.length === 0 ? (
                    <option value={selectedModel}>{selectedModel}</option>
                  ) : models.map(model => (
                    <option key={model.id} value={model.id}>
                      {model.label || model.id} ({model.mol_encoder})
                    </option>
                  ))}
                </select>
              </div>

              {/* Mode Tabs */}
              <div className="mode-tabs">
                <button
                  className={`mode-tab ${mode === 'single' ? 'active' : ''}`}
                  onClick={() => setMode('single')}
                >
                  Single Prediction
                </button>
                <button
                  className={`mode-tab ${mode === 'batch' ? 'active' : ''}`}
                  onClick={() => setMode('batch')}
                >
                  Batch (CSV Upload)
                </button>
                <button
                  className={`mode-tab ${mode === 'screen' ? 'active' : ''}`}
                  onClick={() => setMode('screen')}
                >
                  Virtual Screening
                </button>
              </div>

              {mode === 'single' && (
                <>
                  <button onClick={loadExample} className="example-btn">
                    Load Example
                  </button>

                  <form onSubmit={handleSubmit}>
                    <div className={`form-group ${validationErrors.rna ? 'has-error' : ''}`}>
                      <label>
                        RNA Sequence
                        <span className="required">*</span>
                      </label>
                      <textarea
                        value={rnaSequence}
                        onChange={(e) => {
                          setRnaSequence(e.target.value);
                          if (validationErrors.rna) setValidationErrors(v => ({...v, rna: null}));
                        }}
                        placeholder="Enter RNA sequence (A, U, G, C only, max 512 nt)..."
                        rows={6}
                      />
                      <div className="field-footer">
                        <small>{rnaSequence.length} nucleotides</small>
                        {validationErrors.rna && <small className="error-text">{validationErrors.rna}</small>}
                      </div>
                    </div>

                    <div className={`form-group ${validationErrors.smiles ? 'has-error' : ''}`}>
                      <label>
                        Small Molecule SMILES
                        <span className="required">*</span>
                      </label>
                      <input
                        type="text"
                        value={smiles}
                        onChange={(e) => {
                          setSmiles(e.target.value);
                          if (validationErrors.smiles) setValidationErrors(v => ({...v, smiles: null}));
                        }}
                        placeholder="e.g., CC1=NC2=CC=CC=C2N1"
                      />
                      {validationErrors.smiles && <small className="error-text">{validationErrors.smiles}</small>}
                    </div>

                    <div className="form-actions">
                      <button type="submit" className="btn btn-primary" disabled={loading}>
                        {loading ? (
                          <><span className="spinner"></span>Predicting...</>
                        ) : (
                          <>🚀 Run Prediction</>
                        )}
                      </button>
                      <button type="button" className="btn btn-secondary" onClick={clearForm}>
                        Clear
                      </button>
                    </div>
                  </form>
                </>
              )}

              {mode === 'batch' && (
                <div className="batch-upload">
                  <div className="upload-info">
                    <h4>CSV Format Requirements:</h4>
                    <ul>
                      <li>Must contain <code>sequence</code> and <code>smiles</code> columns</li>
                      <li>RNA sequences: A, U, G, C characters only (max 512 nt)</li>
                      <li>SMILES: valid molecular SMILES strings</li>
                    </ul>
                    <div className="csv-example">
                      <code>
                        sequence,smiles<br/>
                        GGCUAGCUAUAGC,CC1=NC2=CC=CC=C2N1<br/>
                        AUGGCCUUACGA,CCO
                      </code>
                    </div>
                  </div>

                  <label className="upload-btn">
                    📁 Choose CSV File
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".csv"
                      onChange={handleCSVUpload}
                      hidden
                    />
                  </label>

                  {loading && (
                    <div className="upload-progress">
                      <span className="spinner"></span>
                      Processing batch predictions...
                    </div>
                  )}
                </div>
              )}

              {mode === 'screen' && (
                <form onSubmit={handleScreening}>
                  <div className={`form-group ${validationErrors.rna ? 'has-error' : ''}`}>
                    <label>
                      RNA Sequence
                      <span className="required">*</span>
                    </label>
                    <textarea
                      value={rnaSequence}
                      onChange={(e) => {
                        setRnaSequence(e.target.value);
                        if (validationErrors.rna) setValidationErrors(v => ({...v, rna: null}));
                      }}
                      placeholder="Enter the RNA sequence to screen against..."
                      rows={6}
                    />
                    <div className="field-footer">
                      <small>{rnaSequence.length} nucleotides</small>
                      {validationErrors.rna && <small className="error-text">{validationErrors.rna}</small>}
                    </div>
                  </div>

                  <div className={`form-group ${validationErrors.library ? 'has-error' : ''}`}>
                    <label>
                      Compound Library
                      <span className="required">*</span>
                    </label>
                    <select
                      value={selectedLibrary}
                      onChange={(e) => {
                        setSelectedLibrary(e.target.value);
                        if (validationErrors.library) setValidationErrors(v => ({...v, library: null}));
                      }}
                    >
                      {libraries.length === 0 ? (
                        <option value="">No libraries found</option>
                      ) : libraries.map(library => (
                        <option key={library.id} value={library.id} disabled={!library.ready}>
                          {library.name} {library.n_rows ? `(${library.n_rows} molecules)` : ''}
                        </option>
                      ))}
                    </select>
                    {validationErrors.library && <small className="error-text">{validationErrors.library}</small>}
                  </div>

                  <div className="form-group">
                    <label>Number of Candidates</label>
                    <input
                      type="number"
                      min="1"
                      max="200"
                      value={topK}
                      onChange={(e) => setTopK(e.target.value)}
                    />
                  </div>

                  <div className="form-actions">
                    <button type="submit" className="btn btn-primary" disabled={loading || !selectedLibrary}>
                      {loading ? (
                        <><span className="spinner"></span>Screening...</>
                      ) : (
                        <>Run Screening</>
                      )}
                    </button>
                    <button type="button" className="btn btn-secondary" onClick={clearForm}>
                      Clear
                    </button>
                  </div>
                </form>
              )}

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

              {!result && !batchResults && !screeningResult && !loading && (
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

              {result && !batchResults && !screeningResult && renderSingleResult(result)}

              {screeningResult && renderScreeningResult(screeningResult)}

              {batchResults && !screeningResult && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="batch-results"
                >
                  <div className="batch-summary">
                    <h3>Batch Results ({batchResults.length} predictions)</h3>
                    <button className="btn btn-secondary" onClick={downloadResults}>
                      💾 Download All (CSV)
                    </button>
                  </div>

                  <div className="batch-table-container">
                    <table className="sites-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Sequence</th>
                          <th>SMILES</th>
                          <th>Prob</th>
                          <th>Confidence</th>
                          <th>Verdict</th>
                        </tr>
                      </thead>
                      <tbody>
                        {batchResults.map((r, idx) => {
                          const prob = r.interaction_prob ?? 0;
                          return (
                            <tr key={idx} className={r.error ? 'error-row' : ''}>
                              <td>{idx + 1}</td>
                              <td className="sequence-cell">{(r.sequence || '').substring(0, 20)}...</td>
                              <td className="smiles-cell">{(r.smiles || '').substring(0, 20)}...</td>
                              <td>{r.error ? 'ERR' : `${(prob * 100).toFixed(1)}%`}</td>
                              <td>
                                {r.confidence && (
                                  <span className={`confidence-badge confidence-${r.confidence}`}>
                                    {r.confidence}
                                  </span>
                                )}
                              </td>
                              <td>
                                {r.error ? (
                                  <span className="score-badge negative">Error</span>
                                ) : prob > 0.5 ? (
                                  <span className="score-badge positive">Binding</span>
                                ) : (
                                  <span className="score-badge negative">No Binding</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
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
