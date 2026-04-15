import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import './Home.css';

const Home = () => {
  const features = [
    {
      icon: '🎯',
      title: 'Accurate Predictions',
      description: 'State-of-the-art deep learning for RNA-small molecule binding prediction'
    },
    {
      icon: '⚡',
      title: 'Fast Inference',
      description: 'GPU-accelerated predictions with results in seconds'
    },
    {
      icon: '🔬',
      title: 'Binding Site Analysis',
      description: 'Per-residue binding probability visualization'
    },
    {
      icon: '🧪',
      title: 'Multi-task Learning',
      description: 'Simultaneous interaction and binding site prediction'
    }
  ];

  return (
    <div className="home">
      {/* Hero Section */}
      <section className="hero">
        <div className="container">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="hero-content"
          >
            <h1 className="hero-title">
              <span className="gradient-text">CoCoBind</span>
              <br />
              RNA-Drug Binding Prediction
            </h1>
            <p className="hero-subtitle">
              AI-powered platform for predicting RNA-small molecule interactions and binding sites using cooperative consistency constraints
            </p>
            <div className="hero-buttons">
              <Link to="/run" className="btn btn-primary btn-large">
                Start Prediction
                <span>→</span>
              </Link>
              <Link to="/about" className="btn btn-secondary btn-large">
                Learn More
              </Link>
            </div>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="hero-visual"
          >
            <img 
              src={`${process.env.PUBLIC_URL}/model_architecture.png`} 
              alt="CoCoBind Model Architecture" 
              className="hero-arch-img"
            />
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features">
        <div className="container">
          <div className="section-header">
            <h2>Powerful Features</h2>
            <p>Advanced deep learning models designed for RNA drug discovery</p>
          </div>
          <div className="features-grid">
            {features.map((feature, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: idx * 0.1 }}
                viewport={{ once: true }}
                className="feature-card"
              >
                <div className="feature-icon">{feature.icon}</div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="how-it-works">
        <div className="container">
          <div className="section-header">
            <h2>How It Works</h2>
            <p>Three simple steps to get your predictions</p>
          </div>
          <div className="steps">
            <div className="step">
              <div className="step-number">1</div>
              <h3>Input Data</h3>
              <p>Provide RNA sequence and small molecule SMILES</p>
            </div>
            <div className="step-arrow">→</div>
            <div className="step">
              <div className="step-number">2</div>
              <h3>AI Analysis</h3>
              <p>Deep learning model processes your input</p>
            </div>
            <div className="step-arrow">→</div>
            <div className="step">
              <div className="step-number">3</div>
              <h3>Get Results</h3>
              <p>View binding predictions and site probabilities</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="container">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="cta-box"
          >
            <h2>Ready to Start Predicting?</h2>
            <p>Accelerate your RNA drug discovery with CoCoBind</p>
            <Link to="/run" className="btn btn-primary btn-large">
              Launch Prediction Tool
              <span>→</span>
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;
