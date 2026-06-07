import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Loader2, CheckCircle, XCircle, AlertTriangle, Zap, Target, Type } from 'lucide-react';
import { getApiBase } from '../config/api.js';

const SALIENCE = ['important', 'breaking', 'shocking', 'scientists', 'miracle', 'secret', 'urgent', 'exclusive'];

const DATASETS = [
  {
    id: 'fakeddit',
    label: 'Social & viral posts',
    hint: 'Memes, threads, and everyday headlines. Best when the text feels like a normal feed.',
  },
  {
    id: 'liar',
    label: 'News-style statements',
    hint: 'Short factual claims or political lines. Best when it reads like a quote or headline.',
  },
];

const ARCHS = [
  { id: 'distilbert', label: 'Quick scan', hint: 'Faster run, lighter model. Good for rapid checks.' },
  { id: 'hybrid', label: 'Full model', hint: 'Uses text plus style signals. Usually stronger for tricky wording.' },
];

const AnalysisPanel = ({ darkMode }) => {
  const [text, setText] = useState('');
  const [dataset, setDataset] = useState('fakeddit');
  const [arch, setArch] = useState('hybrid');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    try {
      const d = sessionStorage.getItem('veripulse_draft');
      if (d) {
        setText(d);
        sessionStorage.removeItem('veripulse_draft');
      }
    } catch {
      /* ignore */
    }
  }, []);

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${getApiBase()}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          dataset,
          arch,
          persist: true,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      if (data.error) {
        throw new Error(data.error);
      }

      const words = text.trim().split(/\s+/).filter(Boolean);
      const avgLen = words.length
        ? words.reduce((a, w) => a + w.length, 0) / words.length
        : 0;
      const f = data.features || {};

      setResult({
        label: data.label,
        confidence: data.confidence,
        explanation: data.explanation || '',
        modelLoaded: data.model_loaded,
        fallback: data.fallback || 'none',
        dataset: data.dataset || dataset,
        arch: data.arch || arch,
        persistNote: data.persist_warning,
        features: {
          exclamation_count: (text.match(/!/g) || []).length,
          question_count: (text.match(/\?/g) || []).length,
          uppercase_ratio: typeof f.uppercase_ratio === 'number' ? f.uppercase_ratio : 0,
          avg_word_length: avgLen,
          salience_score: typeof f.salience_norm === 'number' ? f.salience_norm : 0,
        },
        attention: SALIENCE.filter((word) => text.toLowerCase().includes(word)),
      });
    } catch (err) {
      setError(
        err.message ||
          'Analysis failed. From project root run: python backend/app.py (port 5000), then retry.'
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-500';
    if (confidence >= 0.6) return 'text-yellow-500';
    return 'text-red-500';
  };

  const highlightImportantWords = (t, attention) => {
    if (!attention || attention.length === 0) return t;
    let highlightedText = t;
    attention.forEach((word) => {
      const regex = new RegExp(`\\b(${word})\\b`, 'gi');
      highlightedText = highlightedText.replace(
        regex,
        `<mark class="bg-yellow-200 text-yellow-800 px-1 rounded">${word}</mark>`
      );
    });
    return highlightedText;
  };

  const dsMeta = DATASETS.find((d) => d.id === dataset) || DATASETS[0];
  const arMeta = ARCHS.find((a) => a.id === arch) || ARCHS[0];

  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className={`
          rounded-2xl shadow-xl p-8 mb-8
          ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}
        `}
      >
        <h2 className={`text-2xl font-bold mb-2 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
          Analyze text
        </h2>
        <p className={`text-sm mb-6 ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
          Choose what kind of text this is and how deep to run the check. Your run is saved for the dashboard (the server scores with all models when you analyze).
        </p>

        <div className="space-y-6">
          <div>
            <span className={`block text-sm font-medium mb-2 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
              What kind of text
            </span>
            <div className="flex flex-col sm:flex-row gap-2">
              {DATASETS.map((d) => (
                <button
                  key={d.id}
                  type="button"
                  onClick={() => setDataset(d.id)}
                  className={`flex-1 text-left rounded-xl border px-4 py-3 transition-all ${
                    dataset === d.id
                      ? darkMode
                        ? 'border-blue-500 bg-blue-500/15 ring-1 ring-blue-500/50'
                        : 'border-blue-600 bg-blue-50 ring-1 ring-blue-200'
                      : darkMode
                        ? 'border-neutral-600 hover:border-neutral-500'
                        : 'border-neutral-200 hover:border-neutral-300'
                  }`}
                >
                  <div className={`font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>{d.label}</div>
                  <div className={`text-xs mt-1 ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>{d.hint}</div>
                </button>
              ))}
            </div>
          </div>

          <div>
            <span className={`block text-sm font-medium mb-2 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
              How to score it
            </span>
            <div className="flex flex-col sm:flex-row gap-2">
              {ARCHS.map((a) => (
                <button
                  key={a.id}
                  type="button"
                  onClick={() => setArch(a.id)}
                  className={`flex-1 text-left rounded-xl border px-4 py-3 transition-all ${
                    arch === a.id
                      ? darkMode
                        ? 'border-emerald-500 bg-emerald-500/15 ring-1 ring-emerald-500/50'
                        : 'border-emerald-600 bg-emerald-50 ring-1 ring-emerald-200'
                      : darkMode
                        ? 'border-neutral-600 hover:border-neutral-500'
                        : 'border-neutral-200 hover:border-neutral-300'
                  }`}
                >
                  <div className={`font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>{a.label}</div>
                  <div className={`text-xs mt-1 ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>{a.hint}</div>
                </button>
              ))}
            </div>
          </div>

          <div className={`rounded-lg px-3 py-2 text-xs ${darkMode ? 'bg-neutral-700/50 text-neutral-300' : 'bg-neutral-100 text-neutral-600'}`}>
            <strong>Selected:</strong> {dsMeta.label} — {arMeta.label}
          </div>

          <div>
            <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
              Text to analyze
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste or type the text you want to score..."
              className={`
                w-full h-32 px-4 py-3 rounded-xl border resize-none focus:ring-2 focus:ring-primary-500 focus:border-transparent
                transition-all duration-200
                ${darkMode
                  ? 'bg-neutral-700 border-neutral-600 text-white placeholder-neutral-400'
                  : 'bg-white border-neutral-300 text-neutral-900 placeholder-neutral-500'
                }
              `}
            />
          </div>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center space-x-2 text-red-500 text-sm"
            >
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </motion.div>
          )}

          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            className={`
              w-full py-3 px-6 rounded-xl font-semibold transition-all duration-200
              flex items-center justify-center gap-2
              ${isAnalyzing
                ? 'bg-neutral-400 cursor-not-allowed text-white'
                : 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-md hover:shadow-lg border border-emerald-700/30'
              }
            `}
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin text-white shrink-0" />
                <span className="text-white">Analyzing...</span>
              </>
            ) : (
              <>
                <Send className="w-5 h-5 text-white shrink-0" strokeWidth={2.25} />
                <span className="text-white">Analyze</span>
              </>
            )}
          </button>
        </div>
      </motion.div>

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            transition={{ duration: 0.5 }}
            className={`
              rounded-2xl shadow-xl p-8 border-2
              ${result.label === 'Reliable'
                ? 'bg-green-50 border-green-200'
                : 'bg-red-50 border-red-200'
              }
            `}
          >
            <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
              <div className="flex items-center space-x-3">
                <div
                  className={`
                  w-12 h-12 rounded-full flex items-center justify-center
                  ${result.label === 'Reliable' ? 'bg-green-500' : 'bg-red-500'}
                `}
                >
                  {result.label === 'Reliable' ? (
                    <CheckCircle className="w-6 h-6 text-white" />
                  ) : (
                    <XCircle className="w-6 h-6 text-white" />
                  )}
                </div>
                <div>
                  <h3 className={`text-xl font-bold ${result.label === 'Reliable' ? 'text-green-800' : 'text-red-800'}`}>
                    {result.label}
                  </h3>
                  <p className={`text-sm ${result.label === 'Reliable' ? 'text-green-600' : 'text-red-600'}`}>
                    {result.dataset === 'liar' ? 'News-style' : 'Social-style'} ·{' '}
                    {result.arch === 'hybrid' ? 'Full model' : 'Quick scan'}
                  </p>
                  {result.explanation && (
                    <p
                      className={`text-sm mt-3 leading-relaxed max-w-2xl whitespace-pre-line ${
                        darkMode ? 'text-neutral-300' : 'text-neutral-700'
                      }`}
                    >
                      {result.explanation}
                    </p>
                  )}
                  {result.modelLoaded === false && (
                    <p className="text-xs text-amber-700 mt-1 font-medium">
                      Weights missing or fallback — check models/ folders (see MODEL_INTEGRATION.md).
                    </p>
                  )}
                  {result.persistNote && (
                    <p className="text-xs text-amber-600 mt-1">Note: {result.persistNote}</p>
                  )}
                </div>
              </div>

              <div className="text-right">
                <div className="flex items-center space-x-2 justify-end">
                  <Target className="w-5 h-5 text-neutral-500" />
                  <span className={`text-2xl font-bold ${getConfidenceColor(result.confidence)}`}>
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <p className="text-sm text-neutral-500">Confidence</p>
              </div>
            </div>

            <div className={`mb-6 p-4 rounded-xl ${darkMode ? 'bg-neutral-800' : 'bg-white'}`}>
              <h4 className={`text-sm font-semibold mb-2 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
                Analyzed text
              </h4>
              <div
                className={`text-sm leading-relaxed ${darkMode ? 'text-neutral-200' : 'text-neutral-700'}`}
                dangerouslySetInnerHTML={{
                  __html: highlightImportantWords(text, result.attention),
                }}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className={`text-sm font-semibold mb-4 flex items-center space-x-2 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
                  <Type className="w-4 h-4" />
                  Linguistic features
                </h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>Exclamation marks</span>
                    <span className={`text-sm font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
                      {result.features.exclamation_count}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>Question marks</span>
                    <span className={`text-sm font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
                      {result.features.question_count}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>Uppercase ratio</span>
                    <span className={`text-sm font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
                      {(result.features.uppercase_ratio * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>Avg word length</span>
                    <span className={`text-sm font-semibold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
                      {result.features.avg_word_length.toFixed(1)}
                    </span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className={`text-sm font-semibold mb-4 flex items-center space-x-2 ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
                  <Zap className="w-4 h-4" />
                  Attention keywords
                </h4>
                <div className="flex flex-wrap gap-2">
                  {result.attention.map((word, index) => (
                    <span
                      key={index}
                      className={`
                        px-3 py-1 rounded-full text-xs font-medium
                        ${darkMode
                          ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                          : 'bg-yellow-100 text-yellow-800 border border-yellow-300'
                        }
                      `}
                    >
                      {word}
                    </span>
                  ))}
                  {result.attention.length === 0 && (
                    <span className={`text-sm ${darkMode ? 'text-neutral-500' : 'text-neutral-500'}`}>None matched</span>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AnalysisPanel;
