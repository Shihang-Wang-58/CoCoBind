import React, { useState } from 'react';
import './JobStatus.css';

const JobStatus = () => {
  const [jobId, setJobId] = useState('');
  const [jobs] = useState([
    { id: '1735894234567', status: 'completed', time: '2 hours ago', rna: 'AUGC...', smiles: 'CCO', probability: 85.3 },
    { id: '1735893234567', status: 'completed', time: '5 hours ago', rna: 'GCAU...', smiles: 'CC(C)O', probability: 42.1 },
    { id: '1735892234567', status: 'failed', time: '1 day ago', rna: 'UACG...', smiles: 'Invalid', probability: null },
  ]);

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
              placeholder="Enter Job ID..."
            />
            <button className="btn btn-primary">Search</button>
          </div>
        </div>

        <div className="jobs-section card">
          <h3>Recent Jobs</h3>
          <div className="jobs-table-container">
            <table className="jobs-table">
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Status</th>
                  <th>RNA Sequence</th>
                  <th>SMILES</th>
                  <th>Probability</th>
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
                    <td>{getStatusBadge(job.status)}</td>
                    <td className="sequence-cell">{job.rna}</td>
                    <td className="smiles-cell">{job.smiles}</td>
                    <td className="prob-cell">
                      {job.probability !== null ? `${job.probability}%` : '-'}
                    </td>
                    <td>{job.time}</td>
                    <td>
                      {job.status === 'completed' && (
                        <button className="btn-small btn-primary">View</button>
                      )}
                      {job.status === 'failed' && (
                        <button className="btn-small btn-secondary">Retry</button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="info-box">
          <h4>💡 Note</h4>
          <p>Jobs are automatically deleted after 7 days. Make sure to download your results.</p>
        </div>
      </div>
    </div>
  );
};

export default JobStatus;
