import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import { Activity, Download, Loader2, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import { getApiBase } from '../config/api.js';

const LiveFeed = ({ darkMode }) => {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [platform, setPlatform] = useState('reddit');
  // Mock live data - replace with actual API calls
  const fetchLivePosts = async () => {
    setLoading(true);
    try {
      const q = '?limit=10&min_sentences=1&max_sentences=2';
      const path = platform === 'reddit' ? '/live/reddit' : '/live/mastodon';
      const response = await fetch(`${getApiBase()}${path}${q}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      if (!Array.isArray(data)) {
        const msg = typeof data?.error === 'string' ? data.error : 'Server returned non-array JSON';
        throw new Error(msg);
      }
      if (data.length === 0) {
        toast.error('No posts returned. Check backend logs / network.');
      }

      const formattedPosts = data.map((post, index) => ({
        id: `${platform}-${index}-${Date.now()}`,
        platform: platform,
        author: post.source,
        content: post.text,
        timestamp: 'Just now',
        analyzed: true,
        result: {
          // API: 0 = Reliable, 1 = Unreliable
          label: post.prediction === 0 ? 'Reliable' : 'Unreliable',
          confidence: post.confidence,
          explanation: post.explanation || ''
        }
      }));
      
      setPosts(formattedPosts);
    } catch (error) {
      console.error('Failed to fetch posts:', error);
      const hint =
        platform === 'mastodon'
          ? ' For Mastodon: set MASTODON_ACCESS_TOKEN or MASTODON_INSTANCES in backend env; check firewall.'
          : '';
      toast.error(
        `Backend Error: ${error.message}. Start API: cd project_root then python backend/app.py (port 5000).${hint}`
      );
    } finally {
      setLoading(false);
    }
  };

  const getPlatformIcon = (platform) => {
    switch (platform) {
      case 'reddit':
        return (
          <div className="w-8 h-8 bg-orange-500 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">R</span>
          </div>
        );
      case 'mastodon':
        return (
          <div className="w-8 h-8 bg-purple-500 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">M</span>
          </div>
        );
      default:
        return (
          <div className="w-8 h-8 bg-neutral-500 rounded-lg flex items-center justify-center">
            <Activity className="w-4 h-4 text-white" />
          </div>
        );
    }
  };

  const getResultIcon = (result) => {
    if (!result) return null;
    
    return result.label === 'Reliable' ? (
      <CheckCircle className="w-5 h-5 text-green-500" />
    ) : (
      <XCircle className="w-5 h-5 text-red-500" />
    );
  };

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className={`
          rounded-2xl shadow-xl p-8 mb-8
          ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}
        `}
      >
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <div>
            <h2 className={`text-2xl font-bold mb-2 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
              Live Feed
            </h2>
            <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
              Analyze live posts from <strong>Reddit</strong> and <strong>Mastodon</strong> to check how reliable they look. Each post is scored with{' '}
              <strong>four models</strong> and saved for the dashboard. The first batch can take a minute on CPU.
            </p>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-3">
            <select
              value={platform}
              onChange={(e) => setPlatform(e.target.value)}
              className={`
                px-4 py-2 rounded-lg border focus:ring-2 focus:ring-primary-500 focus:border-transparent
                ${darkMode 
                  ? 'bg-neutral-700 border-neutral-600 text-white' 
                  : 'bg-white border-neutral-300 text-neutral-900'
                }
              `}
            >
              <option value="reddit">Reddit</option>
              <option value="mastodon">Mastodon</option>
            </select>
            
            <button
              onClick={fetchLivePosts}
              disabled={loading}
              className={`
                px-6 py-2 rounded-lg font-medium transition-all duration-200
                flex items-center justify-center space-x-2
                ${loading
                  ? 'bg-neutral-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-600 hover:to-primary-700 text-white'
                }
              `}
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Fetching...</span>
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  <span>Fetch Live Posts</span>
                </>
              )}
            </button>
          </div>
        </div>
      </motion.div>

      {/* Posts Feed */}
      <div className="space-y-4">
        <AnimatePresence>
          {posts.map((post, index) => (
            <motion.div
              key={post.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={`
                rounded-2xl shadow-lg p-6 border
                ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}
              `}
            >
              {/* Post Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  {getPlatformIcon(post.platform)}
                  <div>
                    <p className={`font-medium ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
                      {post.author}
                    </p>
                    <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
                      {post.timestamp}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  {post.upvotes && (
                    <span className={`text-sm font-medium ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
                      ↑ {post.upvotes}
                    </span>
                  )}
                  {post.likes && (
                    <span className={`text-sm font-medium ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
                      ❤ {post.likes}
                    </span>
                  )}
                </div>
              </div>

              {/* Post Content */}
              <div className={`mb-4 ${darkMode ? 'text-neutral-200' : 'text-neutral-700'}`}>
                <p className="leading-relaxed">{post.content}</p>
              </div>

              {/* Analysis Result */}
              {post.analyzed && post.result && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  transition={{ duration: 0.3 }}
                  className={`
                    mt-4 p-4 rounded-xl border
                    ${post.result.label === 'Reliable'
                      ? 'bg-green-50 border-green-200'
                      : 'bg-red-50 border-red-200'
                    }
                  `}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      {getResultIcon(post.result)}
                      <span className={`font-semibold ${
                        post.result.label === 'Reliable' ? 'text-green-800' : 'text-red-800'
                      }`}>
                        {post.result.label}
                      </span>
                    </div>
                    <div className="text-right">
                      <span className={`text-lg font-bold ${
                        post.result.confidence >= 0.8 ? 'text-green-600' :
                        post.result.confidence >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {(post.result.confidence * 100).toFixed(1)}%
                      </span>
                      <p className={`text-xs ${darkMode ? 'text-neutral-500' : 'text-neutral-600'}`}>
                        Confidence
                      </p>
                    </div>
                  </div>
                  {post.result.explanation && (
                    <p
                      className={`mt-3 text-sm leading-relaxed whitespace-pre-line ${
                        darkMode ? 'text-neutral-700' : 'text-neutral-800'
                      }`}
                    >
                      {post.result.explanation}
                    </p>
                  )}
                </motion.div>
              )}

            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Empty State */}
      {posts.length === 0 && !loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <AlertCircle className="w-12 h-12 mx-auto mb-4 text-neutral-400" />
          <h3 className={`text-lg font-medium mb-2 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            No posts yet
          </h3>
          <p className={`${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
            Click &quot;Fetch Live Posts&quot; to start analyzing social media content
          </p>
        </motion.div>
      )}
    </div>
  );
};

export default LiveFeed;
