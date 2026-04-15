import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './JobStatus.css';

const API_BASE = '/cocobind/api';

const JobStatus = () => {
  const [jobId, setJobId] = useState('');
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedJob, setSelectedJob] = useState(null);
  const [error, setError] = useState(null);

  const fetchJobs = useCallback(async () => {
    try {
      setLoading(true);
      const res = await axios.get(`${API_BASE}/jobs?limit=50`);
      setJobs(res.data);
    } catch (err) {
      console.error('Failed to fetch jobs:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchJobs();
    // Auto-refresh every 10 seconds
    const interval = setInterval(fetchJobs, 10000);
    return () => clearInterval(interval);
  }, [fetchJobs]);

  const searchJob = async () => {
    if (!jobId.trim()) return;
    setError(null);
    setSelectedJob(null);
    try {
      const res = await axios.get(`${API_BASE}/jobs/${jobId.trim()}`);
      setSelectedJob(res.data);
    } catch (err) {
      if (err.response?.status === 404) {
        setError(`Job "${jobId.trim()}" not found`);
      } else {
        setError('Failed to fetch job details');
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') searchJob();
  };

  const viewJobDetail = async (id) => {
    try {
      const res = await axios.get(`${API_BASE}/jobs/${id}`);
      setSelectedJob(res.data);
    } catch (err) {
      setError('Failed to load job details');
    }
  };

  const formatTime = (isoStr) => {
    if (!isoStr) return '-';
    const date = new Date(isoStr);
    const now = new Date();
    const diffMs = now - date;
    const diffMin = Math.floor(diffMs / 60000);
    const diffHr = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHr / 24);

    if (diffMin < 1) return 'Just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHr < 24) return `${diffHr}h ago`;
    return `${diffDay}d ago`;
  };

  const getStatusBadge = (status) => {
    const statusConfig = {
      completed: { label: 'Completed', class: 'status-completed' },
      running: { label: 'Running', class: 'status-running' },
      failed: { label: 'Failed', class: 'status-failed' },
      pending: { label: 'Pending', class: 'status-pending' },
    };
    const config = statusConfig[status] || statusConfig.pending;
    return <span className={`status-badge ${config.class}`}>{config.label}</span>;
  };

  return (
    <div className="job-status-page">
      <div className="container">
        <div className="page-header">
          <h1>Job Status</h1>
          <p>Track your prediction jobs and view results</p>
        </div>

        <div className="search-section card">
          <h3>Search Job by ID</h3>
          <div className="search-box">
            <input
              type="text"
              value={jobId}
              onChange={(e) => setJobId(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Enter Job ID (e.g., a1b2c3d4)..."
            />
            <button className="btn btn-primary" onClick={searchJob}>Search</button>
          </div>
          {error && <p className="search-error">{error}</p>}
        </div>

        {/* Job detail modal/panel */}
        {selectedJob && (
          <div className="job-detail card">
            <div className="detail-header">
              <h3>Job Details — <code>{selectedJob.id}</code></h3>
              <button className="btn-close" onClick={() => setSelectedJob(null)}>✕</button>
            </div>
            <div className="detail-grid">
              <div className="detail-item">
                <span className="detail-label">Status</span>
                {getStatusBadge(selectedJob.status)}
              </div>
              <div className="detail-item">
                <span className="detail-label">Type</span>
                <span className="detail-value">{selectedJob.type || 'single'}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Created</span>
                <span className="detail-value">{new Date(selectedJob.created_at).toLocaleString()}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Completed</span>
                <span className="detail-value">
                  {selectedJob.completed_at ? new Date(selectedJob.completed_at).toLocaleString() : '-'}
                </span>
              </div>
            </div>
            {selectedJob.input_summary && (
              <div className="detail-section">
                <h4>Input</h4>
                <pre>{JSON.stringify(selectedJob.input_summary, null, 2)}</pre>
              </div>
            )}
            {selectedJob.result && (
              <div className="detail-section">
                <h4>Result</h4>
                {selectedJob.result.error ? (
                  <p className="search-error">{selectedJob.result.error}</p>
                ) : selectedJob.result.interaction_prob !== undefined ? (
                  <div className="detail-result-summary">
                    <span>Interaction: <strong>{(selectedJob.result.interaction_prob * 100).toFixed(2)}%</strong></span>
                    <span>Confidence: <strong className={`confidence-${selectedJob.result.confidence}`}>{selectedJob.result.confidence}</strong></span>
                  </div>
                ) : (
                  <pre>{JSON.stringify(selectedJob.result, null, 2)}</pre>
                )}
              </div>
            )}
          </div>
        )}

        <div className="jobs-section card">
          <div className="jobs-header">
            <h3>Recent Jobs {jobs.length > 0 && <small>({jobs.length})</small>}</h3>
            <button className="btn btn-secondary btn-sm" onClick={fetchJobs} disabled={loading}>
              {loading ? '⟳ Refreshing...' : '⟳ Refresh'}
            </button>
          </div>

          {jobs.length === 0 && !loading ? (
            <div className="empty-jobs">
              <p>No jobs found. Run a prediction to see results here.</p>
            </div>
          ) : (
            <div className="jobs-table-container">
              <table className="jobs-table">
                <thead>
                  <tr>
                    <th>Job ID</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Input</th>
                    <th>Submitted</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((job) => (
                    <tr key={job.id}>
                      <td className="job-id-cell">
                        <code>{job.id}</code>
                      </td>
                      <td>
                        <span className={`type-badge type-${job.type}`}>
                          {job.type}
                        </span>
                      </td>
                      <td>{getStatusBadge(job.status)}</td>
                      <td className="input-cell">
                        {job.input_summary?.sequence
                          ? `${job.input_summary.sequence.substring(0, 20)}...`
                          : job.input_summary?.filename || `${job.input_summary?.n_pairs || 0} pairs`
                        }
                      </td>
                      <td>{formatTime(job.created_at)}</td>
                      <td>
                        <button
                          className="btn-small btn-primary"
                          onClick={() => viewJobDetail(job.id)}
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="info-box">
          <h4>💡 Note</h4>
          <p>Job history is stored in memory and will reset when the server restarts. Download important results promptly.</p>
        </div>
      </div>
    </div>
  );
};

export default JobStatus;
