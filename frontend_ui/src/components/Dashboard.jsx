import React from 'react';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, TrendingDown, Activity, Target, Brain, AlertTriangle } from 'lucide-react';

const Dashboard = ({ darkMode }) => {
  // Mock data - replace with actual analytics
  const stats = {
    totalAnalyzed: 1247,
    unreliableRate: 0.43,
    avgConfidence: 0.76,
    improvementRate: 0.23,
  };

  const confidenceData = [
    { range: '90-100%', count: 234, color: '#10b981' },
    { range: '80-89%', count: 456, color: '#22c55e' },
    { range: '70-79%', count: 312, color: '#eab308' },
    { range: '60-69%', count: 189, color: '#f59e0b' },
    { range: '50-59%', count: 56, color: '#ef4444' },
  ];

  const reliabilityData = [
    { name: 'Reliable', value: 711, color: '#10b981' },
    { name: 'Unreliable', value: 536, color: '#ef4444' },
  ];

  const dailyTrends = [
    { day: 'Mon', analyzed: 145, unreliable: 62 },
    { day: 'Tue', analyzed: 189, unreliable: 78 },
    { day: 'Wed', analyzed: 234, unreliable: 98 },
    { day: 'Thu', analyzed: 167, unreliable: 71 },
    { day: 'Fri', analyzed: 298, unreliable: 134 },
    { day: 'Sat', analyzed: 123, unreliable: 56 },
    { day: 'Sun', analyzed: 91, unreliable: 37 },
  ];

  const StatCard = ({ icon: Icon, title, value, subtitle, trend, color }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`
        rounded-2xl p-6 border
        ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}
      `}
    >
      <div className="flex items-center justify-between mb-4">
        <div className={`
          w-12 h-12 rounded-xl flex items-center justify-center
          ${color === 'green' ? 'bg-green-100 text-green-600' :
            color === 'red' ? 'bg-red-100 text-red-600' :
            color === 'blue' ? 'bg-blue-100 text-blue-600' :
            'bg-purple-100 text-purple-600'
          }
        `}>
          <Icon className="w-6 h-6" />
        </div>
        {trend && (
          <div className={`flex items-center space-x-1 text-sm font-medium ${
            trend === 'up' ? 'text-green-500' : 'text-red-500'
          }`}>
            {trend === 'up' ? (
              <TrendingUp className="w-4 h-4" />
            ) : (
              <TrendingDown className="w-4 h-4" />
            )}
            <span>{Math.abs(stats.improvementRate * 100).toFixed(1)}%</span>
          </div>
        )}
      </div>

      <div>
        <p className={`mb-1 text-xs font-semibold uppercase tracking-wide ${darkMode ? 'text-neutral-500' : 'text-neutral-500'}`}>
          {title}
        </p>
        <p className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
          {typeof value === 'number' ? value.toLocaleString() : value}
        </p>
        <p className={`text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
          {subtitle}
        </p>
      </div>
    </motion.div>
  );

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center mb-8"
      >
        <h1 className={`text-3xl font-bold mb-2 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
          Analytics Dashboard
        </h1>
        <p className={`text-lg ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
          Real-time insights from misinformation detection
        </p>
      </motion.div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          icon={Activity}
          title="Total Analyzed"
          value={stats.totalAnalyzed}
          subtitle="Posts analyzed"
          trend="up"
          color="blue"
        />
        <StatCard
          icon={Target}
          title="Unreliable Rate"
          value={`${(stats.unreliableRate * 100).toFixed(1)}%`}
          subtitle="Of total posts"
          trend={stats.unreliableRate > 0.5 ? 'down' : 'up'}
          color="red"
        />
        <StatCard
          icon={Brain}
          title="Avg Confidence"
          value={`${(stats.avgConfidence * 100).toFixed(1)}%`}
          subtitle="Model confidence"
          trend="up"
          color="green"
        />
        <StatCard
          icon={TrendingUp}
          title="Improvement"
          value={`${(stats.improvementRate * 100).toFixed(1)}%`}
          subtitle="vs last week"
          trend="up"
          color="purple"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Confidence Distribution */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className={`
            rounded-2xl p-6 border
            ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}
          `}
        >
          <h3 className={`text-lg font-semibold mb-6 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            Confidence Distribution
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={confidenceData}>
              <CartesianGrid 
                strokeDasharray="3 3" 
                stroke={darkMode ? '#374151' : '#e5e7eb'} 
              />
              <XAxis 
                dataKey="range" 
                stroke={darkMode ? '#9ca3af' : '#6b7280'}
                tick={{ fill: darkMode ? '#9ca3af' : '#6b7280' }}
              />
              <YAxis 
                stroke={darkMode ? '#9ca3af' : '#6b7280'}
                tick={{ fill: darkMode ? '#9ca3af' : '#6b7280' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: darkMode ? '#1f2937' : '#ffffff',
                  border: 'none',
                  borderRadius: '8px',
                  color: darkMode ? '#f3f4f6' : '#111827',
                }}
              />
              <Bar dataKey="count" fill="#3b82f6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Reliability Pie Chart */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className={`
            rounded-2xl p-6 border
            ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}
          `}
        >
          <h3 className={`text-lg font-semibold mb-6 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            Reliability Breakdown
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={reliabilityData}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {reliabilityData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
            </PieChart>
            <Tooltip
              contentStyle={{
                backgroundColor: darkMode ? '#1f2937' : '#ffffff',
                border: 'none',
                borderRadius: '8px',
                color: darkMode ? '#f3f4f6' : '#111827',
              }}
            />
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Daily Trends */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
        className={`
          rounded-2xl p-6 border
            ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}
        `}
      >
        <h3 className={`text-lg font-semibold mb-6 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
          Weekly Analysis Trends
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={dailyTrends}>
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke={darkMode ? '#374151' : '#e5e7eb'} 
            />
            <XAxis 
              dataKey="day" 
              stroke={darkMode ? '#9ca3af' : '#6b7280'}
              tick={{ fill: darkMode ? '#9ca3af' : '#6b7280' }}
            />
            <YAxis 
              stroke={darkMode ? '#9ca3af' : '#6b7280'}
              tick={{ fill: darkMode ? '#9ca3af' : '#6b7280' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: darkMode ? '#1f2937' : '#ffffff',
                border: 'none',
                borderRadius: '8px',
                color: darkMode ? '#f3f4f6' : '#111827',
              }}
            />
            <Bar dataKey="analyzed" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="unreliable" fill="#ef4444" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Insights */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.5 }}
        className={`
          rounded-2xl p-6 border
            ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}
        `}
      >
        <h3 className={`text-lg font-semibold mb-4 flex items-center space-x-2 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
          <AlertTriangle className="w-5 h-5 text-yellow-500" />
          Key Insights
        </h3>
        <div className="space-y-3">
          <div className={`p-4 rounded-lg ${darkMode ? 'bg-neutral-700' : 'bg-neutral-100'}`}>
            <p className={`text-sm ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
              <strong>Peak Detection Hours:</strong> Most unreliable content is detected between 2-6 PM when social media activity is highest.
            </p>
          </div>
          <div className={`p-4 rounded-lg ${darkMode ? 'bg-neutral-700' : 'bg-neutral-100'}`}>
            <p className={`text-sm ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
              <strong>Model Performance:</strong> Average confidence improved by 23% this week, indicating better training data quality.
            </p>
          </div>
          <div className={`p-4 rounded-lg ${darkMode ? 'bg-neutral-700' : 'bg-neutral-100'}`}>
            <p className={`text-sm ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
              <strong>Top Indicators:</strong> Exclamation marks and sensational keywords remain the strongest predictors of unreliable content.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;
