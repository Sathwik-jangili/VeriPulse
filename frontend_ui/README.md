# 🚀 VeriPulse - AI-Powered Misinformation Detection System

A production-level React frontend for detecting misinformation in social media content using advanced hybrid transformer models.

## 🎯 **Features**

### **Core Functionality**
- **Text Analysis**: Input text for instant AI-powered misinformation detection
- **Live Feed**: Real-time monitoring of Reddit and Mastodon posts
- **Dashboard**: Comprehensive analytics with charts and insights
- **Explainability**: Detailed feature breakdown and attention visualization

### **Technical Stack**
- **Frontend**: React 18 with functional components and hooks
- **Styling**: Tailwind CSS with custom design system
- **Animations**: Framer Motion for smooth transitions
- **Charts**: Recharts for data visualization
- **Icons**: Lucide React for modern iconography

### **UI/UX Features**
- **Modern Design**: Fintech-inspired interface (Stripe/Notion style)
- **Dark Mode**: Complete dark/light theme support
- **Responsive**: Mobile-first responsive design
- **Micro-interactions**: Hover states, loading animations, transitions
- **Accessibility**: Semantic HTML and ARIA support

## 🛠️ **Installation & Setup**

### **Prerequisites**
- Node.js 16+ 
- npm or yarn package manager

### **Installation**
```bash
# Clone the repository
git clone <repository-url>
cd frontend_ui

# Install dependencies
npm install

# Start development server
npm run dev
```

### **Environment Setup**
```bash
# For production build
npm run build

# Preview production build
npm run preview
```

## 📁 **Project Structure**

```
frontend_ui/
├── src/
│   ├── components/
│   │   ├── Navbar.jsx          # Navigation with dark mode toggle
│   │   ├── Hero.jsx            # Landing section with CTAs
│   │   ├── AnalysisPanel.jsx   # Text analysis interface
│   │   ├── LiveFeed.jsx        # Social media feed monitoring
│   │   ├── DashboardFixed.jsx  # Analytics dashboard
│   │   └── About.jsx           # About page with features
│   ├── App.jsx               # Main application component
│   ├── main.jsx              # Application entry point
│   └── index.css             # Custom styles and animations
├── public/
│   └── index.html            # HTML template
├── package.json              # Dependencies and scripts
├── tailwind.config.js        # Tailwind configuration
├── vite.config.js            # Vite build configuration
└── README.md                # This file
```

## 🎨 **Design System**

### **Color Palette**
- **Primary**: Green gradient (#10b981 → #22c55e)
- **Danger**: Red gradient (#ef4444 → #dc2626)
- **Neutral**: Gray scale (#f9fafb → #111827)

### **Typography**
- **Font Family**: Inter (system font stack fallback)
- **Weights**: 300, 400, 500, 600, 700
- **Responsive**: Fluid typography scaling

### **Spacing**
- **Scale**: Tailwind's default spacing system
- **Components**: Consistent padding/margins
- **Responsive**: Mobile-first breakpoints

## 🔧 **Backend Integration**

### **API Endpoints**
```javascript
// Text Analysis
POST /api/predict
{
  "text": "string",
  "response": {
    "label": "Reliable" | "Unreliable",
    "confidence": 0.85,
    "features": {...},
    "attention": ["important", "words"]
  }
}

// Live Data
GET /api/live/reddit
GET /api/live/mastodon
{
  "posts": [
    {
      "id": 1,
      "platform": "reddit",
      "author": "username",
      "content": "post text",
      "timestamp": "2 minutes ago",
      "upvotes": 1234
    }
  ]
}
```

### **Frontend Integration**
```javascript
// API Service
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 10000,
});

export const analyzeText = async (text) => {
  const response = await api.post('/predict', { text });
  return response.data;
};

export const fetchLivePosts = async (platform) => {
  const response = await api.get(`/live/${platform}`);
  return response.data.posts;
};
```

## 📱 **Responsive Design**

### **Breakpoints**
- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: 1024px - 1280px
- **Large**: > 1280px

### **Mobile Optimizations**
- Touch-friendly buttons (44px min height)
- Optimized typography scaling
- Simplified navigation for small screens
- Swipeable cards for mobile feeds

## 🌙 **Dark Mode**

### **Implementation**
- CSS custom properties for theme switching
- Persistent user preference (localStorage)
- Smooth transitions between themes
- Component-level dark mode awareness

### **Usage**
```javascript
const [darkMode, setDarkMode] = useState(false);

// Toggle dark mode
const toggleDarkMode = () => {
  setDarkMode(!darkMode);
  localStorage.setItem('darkMode', !darkMode);
};

// Apply theme class
useEffect(() => {
  document.documentElement.classList.toggle('dark', darkMode);
}, [darkMode]);
```

## 🎭 **Animations & Interactions**

### **Framer Motion Usage**
- Page transitions: slide/fade effects
- Component animations: scale/rotate on hover
- Loading states: skeleton screens and spinners
- Micro-interactions: button feedback and state changes

### **Performance Optimizations**
- Lazy loading for images and components
- Debounced search inputs
- Optimized re-renders with React.memo
- Smooth 60fps animations

## 🔍 **Component Documentation**

### **Navbar Component**
```jsx
<Navbar 
  activeTab={activeTab}
  setActiveTab={setActiveTab}
  darkMode={darkMode}
  toggleDarkMode={toggleDarkMode}
/>
```

### **AnalysisPanel Component**
```jsx
<AnalysisPanel 
  darkMode={darkMode}
  onAnalyze={handleAnalyze}
/>
```

### **LiveFeed Component**
```jsx
<LiveFeed 
  darkMode={darkMode}
  platform={platform}
  onAnalyzePost={handlePostAnalysis}
/>
```

### **Dashboard Component**
```jsx
<Dashboard 
  darkMode={darkMode}
  realTimeData={analyticsData}
/>
```

## 🚀 **Deployment**

### **Development**
```bash
npm run dev
# Starts development server at http://localhost:5173
```

### **Production Build**
```bash
npm run build
# Creates optimized build in dist/ folder
```

### **Preview**
```bash
npm run preview
# Preview production build locally
```

## 🎯 **Key Features Implemented**

### ✅ **Completed**
- [x] Modern React 18 architecture with hooks
- [x] Tailwind CSS styling system
- [x] Framer Motion animations
- [x] Dark mode support
- [x] Responsive design
- [x] Component-based architecture
- [x] Live feed integration
- [x] Analytics dashboard
- [x] Text analysis interface
- [x] Error handling and loading states

### 🚧 **In Progress**
- [ ] Real-time backend integration
- [ ] Advanced error boundaries
- [ ] Performance monitoring
- [ ] A/B testing framework
- [ ] Internationalization (i18n)

## 📊 **Browser Support**

- **Chrome**: 90+ ✅
- **Firefox**: 88+ ✅
- **Safari**: 14+ ✅
- **Edge**: 90+ ✅
- **Mobile Safari**: iOS 14+ ✅
- **Chrome Mobile**: Android 8+ ✅

## 🎨 **Design Principles**

### **Visual Hierarchy**
- Clear typography scale (h1-h6)
- Consistent spacing system
- Color-coded actions and states
- Proper contrast ratios

### **Interaction Design**
- Immediate visual feedback
- Smooth state transitions
- Loading and error states
- Accessible focus indicators

### **Performance**
- Optimized bundle size
- Lazy loading strategies
- Efficient re-rendering
- 60fps animations

## 🔐 **Security Considerations**

### **Frontend Security**
- Input sanitization for user content
- XSS prevention in dynamic content
- Secure API communication
- Content Security Policy headers

### **Data Privacy**
- No unnecessary data collection
- Secure localStorage usage
- API key protection
- GDPR compliance considerations

## 🎓 **Learning Resources**

### **React & Modern JS**
- [React Documentation](https://react.dev/)
- [Framer Motion](https://www.framer.com/motion/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Recharts](https://recharts.org/)

### **UI/UX Design**
- [Tailwind UI Components](https://tailwindui.com/)
- [Figma Design System](https://www.figma.com/)
- [Design Principles](https://refactoringui.com/)

## 🏆 **Project Status**

### **Current Version**: 1.0.0
### **Last Updated**: 2024
### **License**: MIT
### **Repository**: [GitHub Link]

---

**Built with ❤️ using React, Tailwind CSS, and Framer Motion**

**VeriPulse - Detecting misinformation, protecting truth** 🛡️
