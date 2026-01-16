import React from 'react';
import './About.css';

const About = () => {
  return (
    <div className="about-page">
      <div className="container">
        <div className="page-header">
          <h1>About CoCoBind</h1>
          <p>Cooperative Consistency-constrained Binding Prediction</p>
        </div>

        <div className="content-section card">
          <h2>What is CoCoBind?</h2>
          <p>
            CoCoBind is an advanced deep learning framework designed for predicting RNA-small molecule interactions 
            and identifying binding sites. Our multi-task model leverages cooperative consistency constraints to 
            simultaneously predict binding affinity and per-residue binding probabilities.
          </p>
        </div>

        <div className="content-section card">
          <h2>Key Features</h2>
          <ul className="features-list">
            <li>
              <strong>Multi-task Learning:</strong> Simultaneous prediction of interaction probability and binding site locations
            </li>
            <li>
              <strong>RNA-FM Embeddings:</strong> Leverages pre-trained RNA foundation models for rich sequence representations
            </li>
            <li>
              <strong>Cross-Attention Architecture:</strong> RNA tokens query molecule tokens for enhanced feature fusion
            </li>
            <li>
              <strong>Consistency Constraints:</strong> Novel loss function ensures coherence between interaction and site predictions
            </li>
            <li>
              <strong>ECFP4 Fingerprints:</strong> State-of-the-art molecular representations for small molecules
            </li>
          </ul>
        </div>

        <div className="content-section card">
          <h2>Model Architecture</h2>
          <p>CoCoBind consists of the following components:</p>
          <ol className="architecture-list">
            <li><strong>RNA Encoder:</strong> RNA-FM pre-trained transformer model</li>
            <li><strong>Molecule Encoder:</strong> ECFP4 fingerprint projector</li>
            <li><strong>Cross-Attention Layer:</strong> RNA-molecule feature fusion</li>
            <li><strong>Dual Prediction Heads:</strong>
              <ul>
                <li>Interaction Head: Predicts overall binding probability</li>
                <li>Site Head: Predicts per-residue binding probabilities</li>
              </ul>
            </li>
          </ol>
        </div>

        <div className="content-section card">
          <h2>Performance</h2>
          <p>
            CoCoBind has been evaluated on multiple challenging scenarios including unseen pairs, 
            unseen compounds, unseen RNAs, and unseen both. Our model achieves state-of-the-art 
            performance across all evaluation metrics.
          </p>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-value">~0.85</div>
              <div className="metric-label">AUROC</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">~0.80</div>
              <div className="metric-label">AUPRC</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">Top-10</div>
              <div className="metric-label">Site Recall</div>
            </div>
          </div>
        </div>

        <div className="content-section card">
          <h2>Citation</h2>
          <div className="citation-box">
            <p>If you use CoCoBind in your research, please cite:</p>
            <code>
              @article&#123;cocobind2026,<br/>
              &nbsp;&nbsp;title=&#123;CoCoBind: Cooperative Consistency-constrained Binding Prediction&#125;,<br/>
              &nbsp;&nbsp;author=&#123;Your Team&#125;,<br/>
              &nbsp;&nbsp;journal=&#123;Journal Name&#125;,<br/>
              &nbsp;&nbsp;year=&#123;2026&#125;<br/>
              &#125;
            </code>
          </div>
        </div>

        <div className="content-section card">
          <h2>API Reference</h2>
          <p>CoCoBind provides a RESTful API for programmatic access:</p>
          
          <div className="api-example">
            <h4>POST /predict</h4>
            <p><strong>Request Body:</strong></p>
            <pre>
{`{
  "rna_sequence": "AUGGCCUU...",
  "smiles": "CCO"
}`}
            </pre>
            
            <p><strong>Response:</strong></p>
            <pre>
{`{
  "interaction_probability": 0.85,
  "binding_sites": [0.1, 0.8, 0.9, ...]
}`}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
