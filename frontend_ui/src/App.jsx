import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Moon, Sun, Home, Activity, FileText, Info, Menu, X, LayoutDashboard } from 'lucide-react';
import { Toaster } from 'react-hot-toast';

// Import components with proper ES6 imports
import Hero from './components/Hero';
import AnalysisPanel from './components/AnalysisPanel';
import LiveFeed from './components/LiveFeed';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import About from './components/About';

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [darkMode, setDarkMode] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    try {
      if (darkMode) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    } catch (e) {
      console.error('DOM error:', e);
    }
  }, [darkMode]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const handleAnalyzeNow = () => {
    setActiveTab('analyze');
  };

  const handleViewLiveFeed = () => {
    setActiveTab('live');
  };

  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'analyze', label: 'Analyze', icon: FileText },
    { id: 'live', label: 'Live Feed', icon: Activity },
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'about', label: 'About', icon: Info },
  ];

  const renderActiveTab = () => {
    try {
      switch (activeTab) {
        case 'home':
          return <Hero darkMode={darkMode} onAnalyzeNow={handleAnalyzeNow} onViewLiveFeed={handleViewLiveFeed} />;
        case 'analyze':
          return <AnalysisPanel darkMode={darkMode} onAnalyze={handleAnalyzeNow} />;
        case 'live':
          return <LiveFeed darkMode={darkMode} />;
        case 'dashboard':
          return <AnalyticsDashboard darkMode={darkMode} />;
        case 'about':
          return <About darkMode={darkMode} />;
        default:
          return <Hero darkMode={darkMode} onAnalyzeNow={handleAnalyzeNow} onViewLiveFeed={handleViewLiveFeed} />;
      }
    } catch (e) {
      console.error('Render error:', e);
      return <div style={{padding: '20px', textAlign: 'center'}}>
        <p>Loading component...</p>
      </div>;
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: darkMode
        ? '#0a0a0f'
        : '#f8fafc',
      fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
    }}>
      {/* Sidebar */}
      <div style={{
        position: 'fixed',
        left: 0,
        top: 0,
        height: '100vh',
        width: sidebarOpen ? '260px' : '0',
        background: darkMode ? '#11111a' : '#ffffff',
        borderRight: `1px solid ${darkMode ? '#1e1e2d' : '#e2e8f0'}`,
        zIndex: 1000,
        overflow: 'hidden',
        transition: 'width 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
      }}>
        {/* Sidebar Header */}
        <div style={{
          padding: '24px',
          borderBottom: `1px solid ${darkMode ? '#1e1e2d' : '#e2e8f0'}`,
          whiteSpace: 'nowrap'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <img
              src="/logo.svg"
              alt="VeriPulse"
              style={{
                width: '32px',
                height: '32px',
                objectFit: 'contain'
              }}
            />
            <h2 style={{
              fontSize: '1.25rem',
              fontWeight: 'bold',
              color: darkMode ? '#ffffff' : '#1e293b'
            }}>
              VeriPulse
            </h2>
          </div>
        </div>

        {/* Navigation Items */}
        <nav style={{ padding: '16px' }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            const iconColor = active
              ? (darkMode ? '#60a5fa' : '#1d4ed8')
              : (darkMode ? '#9ca3af' : '#475569');
            return (
            <motion.button
              key={item.id}
              onClick={() => {
                setActiveTab(item.id);
                setSidebarOpen(false);
              }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              style={{
                width: '100%',
                padding: '12px 14px',
                marginBottom: '8px',
                borderRadius: '12px',
                border: 'none',
                background: active
                  ? (darkMode ? 'rgba(59, 130, 246, 0.2)' : 'rgba(59, 130, 246, 0.12)')
                  : 'transparent',
                color: active
                  ? (darkMode ? '#60a5fa' : '#1e40af')
                  : (darkMode ? '#9ca3af' : '#334155'),
                textAlign: 'left',
                fontWeight: active ? '600' : '500',
                transition: 'all 0.2s ease',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '12px'
              }}
            >
              <Icon size={20} strokeWidth={2.25} color={iconColor} aria-hidden />
              <span style={{ flex: 1 }}>{item.label}</span>
            </motion.button>
            );
          })}
        </nav>
      </div>

      {/* Mobile Menu Button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        style={{
          position: 'fixed',
          top: '20px',
          left: '20px',
          zIndex: 1001,
          padding: '12px',
          borderRadius: '12px',
          border: 'none',
          background: darkMode ? 'rgba(17, 24, 39, 0.9)' : 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(20px)',
          color: darkMode ? '#ffffff' : '#1f2937',
          cursor: 'pointer',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)'
        }}
      >
        {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Dark Mode Toggle */}
      <motion.button
        onClick={toggleDarkMode}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        style={{
          position: 'fixed',
          top: '20px',
          right: '20px',
          zIndex: 1001,
          padding: '12px',
          borderRadius: '50%',
          border: 'none',
          background: darkMode ? 'rgba(17, 24, 39, 0.9)' : 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(20px)',
          color: darkMode ? '#ffffff' : '#1f2937',
          cursor: 'pointer',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)'
        }}
      >
        {darkMode ? <Sun size={20} /> : <Moon size={20} />}
      </motion.button>

      {/* Main Content */}
      <div style={{
        marginLeft: sidebarOpen ? '260px' : '0',
        padding: '40px',
        minHeight: '100vh',
        transition: 'margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
      }}>
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderActiveTab()}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.5)',
            zIndex: 999
          }}
        />
      )}

      {/* Toast Container */}
      <Toaster position="top-right" />
    </div>
  );
}

export default App;
