import React from 'react';
import { motion } from 'framer-motion';
import { Cpu, Shield, Zap, Mail } from 'lucide-react';

const About = ({ darkMode }) => {
  return (
    <div style={{ padding: '40px', maxWidth: '1200px', margin: '0 auto' }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 style={{
          fontSize: '2.5rem',
          fontWeight: 'bold',
          marginBottom: '20px',
          color: darkMode ? '#ffffff' : '#1f2937'
        }}>
          About VeriPulse
        </h2>

        <p style={{
          fontSize: '1.2rem',
          lineHeight: '1.6',
          marginBottom: '40px',
          color: darkMode ? '#d1d5db' : '#4b5563'
        }}>
          VeriPulse is an advanced AI-powered misinformation detection system that helps identify fake news and unreliable content across social media platforms.
        </p>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '30px',
          marginBottom: '40px'
        }}>
          <motion.div
            whileHover={{ scale: 1.05 }}
            style={{
              padding: '30px',
              borderRadius: '12px',
              background: darkMode ? '#374151' : '#f9fafb',
              border: `1px solid ${darkMode ? '#4b5563' : '#e5e7eb'}`
            }}
          >
            <Cpu style={{
              width: '48px',
              height: '48px',
              color: '#3b82f6',
              marginBottom: '16px'
            }} />
            <h3 style={{
              fontSize: '1.5rem',
              fontWeight: 'bold',
              marginBottom: '12px',
              color: darkMode ? '#ffffff' : '#1f2937'
            }}>
              AI-Powered Detection
            </h3>
            <p style={{
              color: darkMode ? '#d1d5db' : '#4b5563',
              lineHeight: '1.6'
            }}>
              Advanced hybrid transformer model combines DistilBERT with linguistic features for superior accuracy.
            </p>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.05 }}
            style={{
              padding: '30px',
              borderRadius: '12px',
              background: darkMode ? '#374151' : '#f9fafb',
              border: `1px solid ${darkMode ? '#4b5563' : '#e5e7eb'}`
            }}
          >
            <Shield style={{
              width: '48px',
              height: '48px',
              color: '#10b981',
              marginBottom: '16px'
            }} />
            <h3 style={{
              fontSize: '1.5rem',
              fontWeight: 'bold',
              marginBottom: '12px',
              color: darkMode ? '#ffffff' : '#1f2937'
            }}>
              Real-Time Analysis
            </h3>
            <p style={{
              color: darkMode ? '#d1d5db' : '#4b5563',
              lineHeight: '1.6'
            }}>
              Get instant results with confidence scores and detailed explanations for each prediction.
            </p>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.05 }}
            style={{
              padding: '30px',
              borderRadius: '12px',
              background: darkMode ? '#374151' : '#f9fafb',
              border: `1px solid ${darkMode ? '#4b5563' : '#e5e7eb'}`
            }}
          >
            <Zap style={{
              width: '48px',
              height: '48px',
              color: '#f59e0b',
              marginBottom: '16px'
            }} />
            <h3 style={{
              fontSize: '1.5rem',
              fontWeight: 'bold',
              marginBottom: '12px',
              color: darkMode ? '#ffffff' : '#1f2937'
            }}>
              Live Feed Monitoring
            </h3>
            <p style={{
              color: darkMode ? '#d1d5db' : '#4b5563',
              lineHeight: '1.6'
            }}>
              Connect to Reddit and Mastodon APIs to monitor social media content in real-time.
            </p>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          style={{
            padding: '30px',
            borderRadius: '12px',
            background: darkMode ? '#1f2937' : '#ffffff',
            border: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`
          }}
        >
          <h3 style={{
            fontSize: '1.8rem',
            fontWeight: 'bold',
            marginBottom: '20px',
            color: darkMode ? '#ffffff' : '#1f2937'
          }}>
            Technology Stack
          </h3>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '20px'
          }}>
            <div>
              <h4 style={{
                color: '#3b82f6',
                fontWeight: 'bold',
                marginBottom: '8px'
              }}>
                Frontend
              </h4>
              <p style={{
                color: darkMode ? '#d1d5db' : '#4b5563',
                fontSize: '0.9rem'
              }}>
                React, Framer Motion, Tailwind CSS, Lucide Icons
              </p>
            </div>
            <div>
              <h4 style={{
                color: '#10b981',
                fontWeight: 'bold',
                marginBottom: '8px'
              }}>
                Backend
              </h4>
              <p style={{
                color: darkMode ? '#d1d5db' : '#4b5563',
                fontSize: '0.9rem'
              }}>
                Python, Flask, DistilBERT, PyTorch
              </p>
            </div>
            <div>
              <h4 style={{
                color: '#f59e0b',
                fontWeight: 'bold',
                marginBottom: '8px'
              }}>
                AI Models
              </h4>
              <p style={{
                color: darkMode ? '#d1d5db' : '#4b5563',
                fontSize: '0.9rem'
              }}>
                Advanced Hybrid Transformer, BERT, Linguistic Analysis
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          style={{
            textAlign: 'center',
            padding: '40px 20px',
            marginTop: '40px'
          }}
        >
          <h3 style={{
            fontSize: '1.5rem',
            fontWeight: 'bold',
            marginBottom: '20px',
            color: darkMode ? '#ffffff' : '#1f2937'
          }}>
            Contact
          </h3>
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            marginBottom: '20px'
          }}>
            <motion.a
              href="mailto:sathwikjangili@gmail.com"
              whileHover={{ scale: 1.05 }}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '10px',
                padding: '12px 20px',
                borderRadius: '8px',
                background: darkMode ? '#374151' : '#f3f4f6',
                color: darkMode ? '#ffffff' : '#1f2937',
                textDecoration: 'none',
                fontWeight: 600,
                fontSize: '0.95rem'
              }}
            >
              <Mail style={{ width: '22px', height: '22px', flexShrink: 0 }} />
              sathwikjangili@gmail.com
            </motion.a>
          </div>
          <p style={{
            color: darkMode ? '#9ca3af' : '#6b7280',
            fontSize: '0.9rem'
          }}>
            © 2026 VeriPulse. All rights reserved.
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default About;
