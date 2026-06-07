import React from 'react';
import { motion } from 'framer-motion';
import { Home, Activity, FileText, Info, Moon, Sun } from 'lucide-react';

const Navbar = ({ activeTab, setActiveTab, darkMode, toggleDarkMode }) => {
  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'live', label: 'Live Feed', icon: Activity },
    { id: 'analyze', label: 'Analyse Text', icon: FileText },
    { id: 'about', label: 'About', icon: Info },
  ];

  return (
    <nav className={`sticky top-0 z-50 backdrop-blur-lg ${darkMode ? 'bg-neutral-900/90' : 'bg-white/90'} border-b ${darkMode ? 'border-neutral-800' : 'border-neutral-200'}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="flex items-center space-x-3"
          >
            <div
              className={`w-10 h-10 rounded-xl flex items-center justify-center overflow-hidden border ${
                darkMode ? 'bg-white/10 border-white/20' : 'bg-slate-100 border-slate-200'
              }`}
            >
              <img src="/logo.svg" alt="" className="w-8 h-8" />
            </div>
            <div>
              <h1 className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>VeriPulse</h1>
              <p className={`text-xs ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>AI-Powered Misinformation Detection</p>
            </div>
          </motion.div>

          {/* Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`
                    relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                    ${isActive 
                      ? darkMode 
                        ? 'bg-primary-500/10 text-primary-400' 
                        : 'bg-primary-50 text-primary-600'
                      : darkMode
                        ? 'text-neutral-300 hover:text-white hover:bg-neutral-800'
                        : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100'
                    }
                  `}
                >
                  <div className="flex items-center space-x-2">
                    <Icon className="w-4 h-4" />
                    <span>{item.label}</span>
                  </div>
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-primary-500 to-primary-600 rounded-full"
                      initial={false}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.3 }}
                    />
                  )}
                </button>
              );
            })}
          </div>

          {/* Dark Mode Toggle */}
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
            onClick={toggleDarkMode}
            className={`
              p-2 rounded-lg transition-all duration-200
              ${darkMode 
                ? 'bg-neutral-800 text-yellow-400 hover:bg-neutral-700' 
                : 'bg-neutral-100 text-neutral-600 hover:bg-neutral-200'
              }
            `}
          >
            {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </motion.button>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button className={`p-2 rounded-lg ${darkMode ? 'text-neutral-300' : 'text-neutral-600'}`}>
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        <motion.div 
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="md:hidden overflow-hidden"
        >
          <div className="py-4 space-y-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`
                    w-full px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 flex items-center space-x-3
                    ${isActive 
                      ? darkMode 
                        ? 'bg-primary-500/10 text-primary-400' 
                        : 'bg-primary-50 text-primary-600'
                      : darkMode
                        ? 'text-neutral-300 hover:text-white hover:bg-neutral-800'
                        : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100'
                    }
                  `}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </div>
        </motion.div>
      </div>
    </nav>
  );
};

export default Navbar;
