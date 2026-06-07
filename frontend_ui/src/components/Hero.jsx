import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Star, Zap, Shield, Mic, Radio, Activity, MicOff } from 'lucide-react';
import CountUp from 'react-countup';
import toast from 'react-hot-toast';
import { getApiBase } from '../config/api.js';

const Hero = ({ darkMode, onAnalyzeNow, onViewLiveFeed }) => {
  const [stats, setStats] = useState({ analyzed: 0, unreliable: 0 });
  const [isTyping, setIsTyping] = useState(false);
  const [inputText, setInputText] = useState('');
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef(null);
  const typingTimerRef = useRef(null);

  const refreshStats = useCallback(async () => {
    try {
      const res = await fetch(`${getApiBase()}/dashboard/summary`);
      if (!res.ok) return;
      const data = await res.json();
      const posts = data.totals?.posts ?? 0;
      const preds = data.totals?.predictions ?? 0;
      const pie = data.reliability_pie || [];
      const unrel = pie.find((p) => p.name === 'Unreliable')?.value ?? 0;
      setStats({
        analyzed: Math.max(posts, preds > 0 ? Math.round(preds / 4) : 0),
        unreliable: unrel,
      });
    } catch {
      /* backend may be off */
    }
  }, []);

  useEffect(() => {
    refreshStats();
    const interval = setInterval(refreshStats, 15000);
    return () => clearInterval(interval);
  }, [refreshStats]);

  const handleInputChange = (e) => {
    setInputText(e.target.value);
    setIsTyping(true);
    if (typingTimerRef.current) window.clearTimeout(typingTimerRef.current);
    typingTimerRef.current = window.setTimeout(() => setIsTyping(false), 800);
  };

  const stopListening = () => {
    try {
      recognitionRef.current?.stop?.();
    } catch {
      /* ignore */
    }
    setListening(false);
  };

  const startListening = () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      toast.error('Speech recognition is not supported in this browser. Try Chrome or Edge.');
      return;
    }
    if (listening) {
      stopListening();
      return;
    }
    const rec = new SR();
    rec.lang = 'en-US';
    rec.interimResults = false;
    rec.continuous = false;
    recognitionRef.current = rec;

    rec.onstart = () => {
      setListening(true);
      toast.success('Listening… speak now.', { icon: '🎤' });
    };
    rec.onend = () => setListening(false);
    rec.onerror = (ev) => {
      setListening(false);
      const msg = ev.error === 'not-allowed' ? 'Microphone blocked — allow access in browser settings.' : `Voice error: ${ev.error}`;
      toast.error(msg);
    };
    rec.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((r) => r[0].transcript)
        .join(' ')
        .trim();
      if (transcript) {
        setInputText((prev) => (prev ? `${prev} ${transcript}` : transcript));
        toast.success('Captured speech to text.');
      }
    };
    try {
      rec.start();
    } catch (e) {
      toast.error('Could not start microphone.');
    }
  };

  useEffect(() => () => stopListening(), []);

  const particles = Array.from({ length: 10 }, (_, i) => ({
    id: i,
    x: Math.random() * 100 - 50,
    y: Math.random() * 100 - 50,
    scale: Math.random() * 0.5 + 0.5,
  }));

  return (
    <div className={`relative overflow-hidden min-h-screen ${darkMode ? 'dark' : ''}`}>
      <div
        className={`absolute inset-0 ${darkMode ? 'bg-[#0a0a12]' : 'bg-slate-900'}`}
        aria-hidden
      />
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 opacity-30" />
      <div className="absolute inset-0 bg-gradient-to-tr from-blue-500 via-cyan-500 to-teal-500 opacity-15 animate-pulse" />

      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute w-1 h-1 bg-white rounded-full"
          initial={{
            x: '50%',
            y: '50%',
            scale: 0,
            opacity: 0,
          }}
          animate={{
            x: `calc(50% + ${particle.x}px)`,
            y: `calc(50% + ${particle.y}px)`,
            scale: particle.scale,
            opacity: [0, 1, 0],
            rotate: [0, 360],
          }}
          transition={{
            duration: 3 + particle.id,
            repeat: Infinity,
            ease: 'linear',
          }}
        />
      ))}

      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4">
        <motion.div
          className="mb-8"
          initial={{ opacity: 0, scale: 0.5, rotateY: -180 }}
          animate={{
            opacity: 1,
            scale: 1,
            rotateY: 0,
            rotate: 360,
          }}
          transition={{
            duration: 1,
            rotate: { duration: 2, repeat: Infinity, ease: 'linear' },
            scale: { duration: 0.5, repeat: Infinity, repeatType: 'reverse' },
          }}
          whileHover={{ scale: 1.1, rotateY: 180 }}
        >
          <div className="relative">
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-600 rounded-full blur-xl opacity-50"
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.3, 0.7, 0.3],
              }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <div className="relative bg-[#1a1a2e] dark:bg-gray-900 rounded-full p-8 shadow-2xl border border-blue-500/20">
              <img
                src="/logo.svg"
                alt="VeriPulse Logo"
                style={{
                  width: '80px',
                  height: '80px',
                  objectFit: 'contain',
                  filter: 'drop-shadow(0 0 15px rgba(59, 130, 246, 0.5))',
                }}
              />
            </div>
          </div>
        </motion.div>

        <motion.h1
          className="text-5xl md:text-7xl font-extrabold text-center mb-4 tracking-tight"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          style={{
            color: '#ffffff',
            textShadow: '0 0 30px rgba(59, 130, 246, 0.3)',
          }}
        >
          VeriPulse
        </motion.h1>

        <motion.p
          className="text-xl md:text-2xl text-center mb-12 max-w-2xl mx-auto font-medium"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.7 }}
          style={{
            color: 'rgba(255,255,255,0.85)',
            lineHeight: '1.6',
          }}
        >
          Advanced AI-powered misinformation detection. Verify facts with high-precision hybrid transformer models.
        </motion.p>

        <motion.div
          className="backdrop-blur-xl bg-white/10 dark:bg-black/20 border border-white/20 rounded-2xl p-8 shadow-2xl mb-8 max-w-2xl w-full"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.9 }}
          whileHover={{ scale: 1.02, boxShadow: '0 20px 40px rgba(0,0,0,0.2)' }}
        >
          <div className="space-y-6">
            <div className="relative">
              <textarea
                value={inputText}
                onChange={handleInputChange}
                placeholder="Enter text to analyze… (use the mic for speech-to-text in Chrome/Edge)"
                className="w-full p-4 rounded-xl bg-white/50 dark:bg-black/30 border border-white/20 text-gray-900 dark:text-white placeholder-gray-500 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows={4}
              />
              <AnimatePresence>
                {isTyping && (
                  <motion.div
                    className="absolute top-2 right-2 text-sm text-gray-500 dark:text-gray-400"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  >
                    Typing…
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <div className="flex flex-wrap gap-2 sm:gap-3 justify-center items-center">
              <motion.button
                type="button"
                onClick={() => {
                  try {
                    if (inputText.trim()) {
                      sessionStorage.setItem('veripulse_draft', inputText);
                    }
                  } catch {
                    /* ignore */
                  }
                  onAnalyzeNow();
                  toast.success('Open Analyze to pick text type and quick vs full model.', { icon: '🚀' });
                }}
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium bg-white/90 text-slate-800 border border-white/40 shadow-sm hover:bg-white transition-colors"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Zap className="w-4 h-4 text-emerald-600" />
                Analyze Text
              </motion.button>

              <motion.button
                type="button"
                onClick={() => {
                  onViewLiveFeed();
                  toast('Opening live feed…', { icon: '📡' });
                }}
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium bg-white/15 text-white border border-white/30 hover:bg-white/25 transition-colors"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Radio className="w-4 h-4" />
                Live Feed
              </motion.button>

              <motion.button
                type="button"
                onClick={startListening}
                className={`inline-flex items-center justify-center gap-1.5 min-w-[2.5rem] h-10 px-3 rounded-lg text-sm font-medium border transition-colors ${
                  listening
                    ? 'bg-red-500/30 text-white border-red-400/50'
                    : 'bg-white/15 text-white border-white/30 hover:bg-white/25'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                aria-label={listening ? 'Stop microphone' : 'Start microphone'}
                title={listening ? 'Stop listening' : 'Speak to fill text (Chrome/Edge)'}
              >
                {listening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
              </motion.button>
            </div>
          </div>
        </motion.div>

        <motion.div
          className="grid grid-cols-2 gap-6 max-w-2xl w-full"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.1 }}
        >
          <motion.div
            className="backdrop-blur-xl bg-white/10 dark:bg-black/20 border border-white/20 rounded-xl p-6 text-center"
            whileHover={{ scale: 1.02 }}
          >
            <div className="text-3xl font-bold text-white mb-2 drop-shadow-sm">
              <CountUp end={stats.analyzed} duration={1.2} />
            </div>
            <div className="text-sm text-white/90 font-medium">Unique posts in database</div>
          </motion.div>

          <motion.div
            className="backdrop-blur-xl bg-white/10 dark:bg-black/20 border border-white/20 rounded-xl p-6 text-center"
            whileHover={{ scale: 1.02 }}
          >
            <div className="text-3xl font-bold text-white mb-2 drop-shadow-sm">
              <CountUp end={stats.unreliable} duration={1.2} />
            </div>
            <div className="text-sm text-white/90 font-medium">Unreliable (primary scorer)</div>
          </motion.div>
        </motion.div>

        <motion.div
          className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl w-full mt-8"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.3 }}
        >
          {[
            { icon: Shield, text: 'Real-time Detection' },
            { icon: Star, text: 'Advanced AI Analysis' },
            { icon: Activity, text: 'Live Monitoring' },
            { icon: Zap, text: 'Confidence Scoring' },
          ].map((feature, index) => (
            <motion.div
              key={index}
              className="backdrop-blur-xl bg-white/10 dark:bg-black/20 border border-white/20 rounded-xl p-4 text-center"
              whileHover={{
                scale: 1.05,
                y: -4,
                boxShadow: '0 20px 40px rgba(0,0,0,0.3)',
              }}
              transition={{ duration: 0.2 }}
            >
              <feature.icon className="w-8 h-8 mx-auto mb-2 text-white drop-shadow" />
              <div className="text-sm font-medium text-white">{feature.text}</div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </div>
  );
};

export default Hero;
