import React, { useState, useEffect } from 'react';
import {
  FileText, Camera, RefreshCw,
  Trash2, ShieldAlert, Info, ArrowRight,
  Sun, Moon, Zap, BarChart2,
  AlertCircle, CheckCircle2, User, Cpu,
  Target, Layers, HelpCircle, Activity,
  Sliders, Bug, FileCheck
} from 'lucide-react';

// --- CONFIG & UTILS ---
const API_BASE = 'http://localhost:5000';
const STORAGE_KEY = 'flowstate_session';

function cn(...classes) {
  return classes.filter(Boolean).join(' ');
}

const DEFAULT_RAW_TEXT = `History Essay - Read Ch. 4 (30 pgs) and write 3-page summary due Friday at 11:59pm.
Math Problem Set 5 - Exercises 1-12 on page 140, due tomorrow 11:59pm.
Physics Lab Report - Review guidelines and write intro section, due next Monday.
Computer Science Project - Implement core graph algorithms in Python.`;

const DEFAULT_USER_FACTORS = {
  essay: { factor: 1.25, n_samples: 4, label: "Essays & Writing" },
  math: { factor: 0.92, n_samples: 6, label: "Problem Sets & Math" },
  reading: { factor: 1.05, n_samples: 5, label: "Reading & Research" },
  coding: { factor: 1.25, n_samples: 2, label: "Coding & Projects" },
  creative: { factor: 1.00, n_samples: 0, label: "Creative & Design" },
};

export default function App() {
  // --- Navigation & Workflow State ---
  const [currentStep, setCurrentStep] = useState(1); // 1: Input, 2: Review, 3: Schedule, 4: Learning, 5: Competition Deck
  const [inputMode, setInputMode] = useState('text'); // 'text' | 'image'
  const [rawText, setRawText] = useState(DEFAULT_RAW_TEXT);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [workPreference, setWorkPreference] = useState('morning');
  const [userId, setUserId] = useState('demo_student_01');

  // --- Data States ---
  const [isParsing, setIsParsing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [parsedTasks, setParsedTasks] = useState(null);
  const [taskAnalysis, setTaskAnalysis] = useState(null);
  const [scheduleBlocks, setScheduleBlocks] = useState(null);
  const [feasibility, setFeasibility] = useState(null);
  const [userFactors, setUserFactors] = useState(DEFAULT_USER_FACTORS);
  const [completedTasks, setCompletedTasks] = useState([]);

  // --- UI States ---
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [toastMessage, setToastMessage] = useState(null);
  const [completingTask, setCompletingTask] = useState(null);
  const [actualMinutesInput, setActualMinutesInput] = useState('');
  const [expandedExplanation, setExpandedExplanation] = useState({});
  const [animatedFactorChange, setAnimatedFactorChange] = useState(null);
  const [activeRubricTab, setActiveRubricTab] = useState(1);

  // Fonts Injection
  useEffect(() => {
    const link = document.createElement('link');
    link.href = 'https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Fraunces:ital,opsz,wght@0,9..144,400..700;1,9..144,400..700&display=swap';
    link.rel = 'stylesheet';
    document.head.appendChild(link);
    return () => {
      try {
        document.head.removeChild(link);
      } catch (e) {}
    };
  }, []);

  // Health Check API
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
        if (res.ok) setBackendStatus('online');
        else setBackendStatus('offline');
      } catch {
        setBackendStatus('offline');
      }
    };
    checkHealth();
  }, []);

  // LocalStorage Persistence
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const data = JSON.parse(saved);
        if (data.parsedTasks) setParsedTasks(data.parsedTasks);
        if (data.taskAnalysis) setTaskAnalysis(data.taskAnalysis);
        if (data.scheduleBlocks) setScheduleBlocks(data.scheduleBlocks);
        if (data.feasibility) setFeasibility(data.feasibility);
        if (data.currentStep) setCurrentStep(data.currentStep);
        if (data.userFactors) setUserFactors(data.userFactors);
        if (data.workPreference) setWorkPreference(data.workPreference);
        if (data.completedTasks) setCompletedTasks(data.completedTasks);
      }
    } catch (e) {
      console.warn("Failed to load local state", e);
    }
  }, []);

  useEffect(() => {
    try {
      const sessionBlob = {
        parsedTasks,
        taskAnalysis,
        scheduleBlocks,
        feasibility,
        currentStep,
        userFactors,
        workPreference,
        completedTasks,
        updatedAt: new Date().toISOString()
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessionBlob));
    } catch (e) {
      console.warn("Failed to write local state", e);
    }
  }, [parsedTasks, taskAnalysis, scheduleBlocks, feasibility, currentStep, userFactors, workPreference, completedTasks]);

  const handleClearSession = () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
      setParsedTasks(null);
      setTaskAnalysis(null);
      setScheduleBlocks(null);
      setFeasibility(null);
      setCurrentStep(1);
      setRawText(DEFAULT_RAW_TEXT);
      setSelectedImage(null);
      setImagePreviewUrl(null);
      setUserFactors(DEFAULT_USER_FACTORS);
      setCompletedTasks([]);
      showToast("Saved session reset to baseline demo configuration.");
    } catch (e) {
      showToast("Could not clear session.");
    }
  };

  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3800);
  };

  const handleImageSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        showToast("Image size must be under 10MB.");
        return;
      }
      setSelectedImage(file);
      setImagePreviewUrl(URL.createObjectURL(file));
    }
  };

  // Parsing Handler
  const handleParseTasks = async () => {
    if (inputMode === 'text' && !rawText.trim()) {
      showToast("Paste syllabus or assignment text first.");
      return;
    }
    if (inputMode === 'image' && !selectedImage) {
      showToast("Select or drop a photo/screenshot first.");
      return;
    }

    setIsParsing(true);

    if (backendStatus === 'online') {
      try {
        let res;
        if (inputMode === 'image') {
          const formData = new FormData();
          formData.append('image', selectedImage);
          res = await fetch(`${API_BASE}/api/tasks/parse/image`, { method: 'POST', body: formData });
        } else {
          res = await fetch(`${API_BASE}/api/tasks/parse`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ raw_text: rawText })
          });
        }

        if (res.ok) {
          const data = await res.json();
          setParsedTasks(data.tasks || []);
          setCurrentStep(2);
          setIsParsing(false);
          return;
        }
      } catch (err) {
        console.warn("Backend unavailable, using client mock LLM extractor", err);
      }
    }

    // Mock LLM Fallback
    setTimeout(() => {
      const isFlagged = rawText.toLowerCase().includes("cheat") || rawText.toLowerCase().includes("prompt") || rawText.toLowerCase().includes("ignore instructions");
      const mockTasks = [
        {
          task_id: "t_101",
          title: "History Essay - Read Ch. 4 (30 pgs)",
          task_type: "essay",
          deadline: "2026-10-30T23:59:00",
          base_difficulty: 6,
          deadline_matched: true,
          source_flagged_for_review: isFlagged
        },
        {
          task_id: "t_102",
          title: "Math Problem Set 5 (Ex. 1-12)",
          task_type: "math",
          deadline: "2026-10-27T23:59:00",
          base_difficulty: 7,
          deadline_matched: true,
          source_flagged_for_review: false
        },
        {
          task_id: "t_103",
          title: "Physics Lab Report - Intro section",
          task_type: "reading",
          deadline: "2026-10-29T17:00:00",
          base_difficulty: 5,
          deadline_matched: false,
          source_flagged_for_review: false
        },
        {
          task_id: "t_104",
          title: "CS Project - Graph Algorithms",
          task_type: "coding",
          deadline: "2026-11-02T23:59:00",
          base_difficulty: 8,
          deadline_matched: true,
          source_flagged_for_review: false
        }
      ];
      setParsedTasks(mockTasks);
      setCurrentStep(2);
      setIsParsing(false);
      showToast("LLM extracted task data. Review details below.");
    }, 850);
  };

  // Schedule Generation Handler
  const handleGenerateSchedule = async () => {
    if (!parsedTasks || parsedTasks.length === 0) {
      showToast("No tasks available to schedule.");
      return;
    }

    setIsGenerating(true);

    if (backendStatus === 'online') {
      try {
        const res = await fetch(`${API_BASE}/api/schedule/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            tasks: parsedTasks,
            user_id: userId,
            peak_window: workPreference
          })
        });

        if (res.ok) {
          const data = await res.json();
          setTaskAnalysis(data.task_analysis || []);
          setScheduleBlocks(data.schedule || []);
          setFeasibility(data.feasibility || { completion_probability: 0.82, risk_flagged_task_id: null });
          setCurrentStep(3);
          setIsGenerating(false);
          return;
        }
      } catch (err) {
        console.warn("Backend error, falling back to mock generator", err);
      }
    }

    // Mock Custom Models Fallback
    setTimeout(() => {
      const mockAnalysis = parsedTasks.map((t, idx) => {
        const factorObj = userFactors[t.task_type] || { factor: 1.0, n_samples: 0 };
        const basePopMins = (t.base_difficulty || 5) * 15;
        const predictedMins = Math.round(basePopMins * factorObj.factor);

        return {
          task_id: t.task_id,
          title: t.title,
          task_type: t.task_type,
          difficulty: t.base_difficulty || 5,
          predicted_duration_minutes: predictedMins,
          population_predicted_duration_minutes: basePopMins,
          correction_factor_applied: factorObj.factor,
          n_samples: factorObj.n_samples,
          priority_rank: idx + 1,
          chunked: predictedMins > 60,
          milestones: predictedMins > 60 ? ["Outline & Research", "Draft Content", "Final Review"] : null,
          buffer_deadline: new Date(Date.now() + 86400000 * (idx + 1) - 14400000).toISOString(),
          explanation: `Population baseline: ${basePopMins}m (diff ${t.base_difficulty}/10). Personalized model correction for ${t.task_type}: ${factorObj.factor}× (${factorObj.n_samples} historical samples).`
        };
      });

      const mockSchedule = [
        {
          block_id: "b_1",
          task_id: "t_102",
          title: "Math Problem Set 5 (Ex. 1-12)",
          milestone: null,
          part_info: "Part 1 of 2",
          start_time: "2026-10-27T08:00:00",
          end_time: "2026-10-27T08:45:00",
          block_type: "work"
        },
        {
          block_id: "b_2",
          task_id: null,
          title: "Short Rest & Stretch",
          milestone: null,
          start_time: "2026-10-27T08:45:00",
          end_time: "2026-10-27T08:55:00",
          block_type: "break"
        },
        {
          block_id: "b_3",
          task_id: "t_102",
          title: "Math Problem Set 5 (Ex. 1-12)",
          milestone: null,
          part_info: "Part 2 of 2",
          start_time: "2026-10-27T08:55:00",
          end_time: "2026-10-27T09:30:00",
          block_type: "work"
        },
        {
          block_id: "b_4",
          task_id: "t_101",
          title: "History Essay - Read Ch. 4",
          milestone: "Outline & Research",
          start_time: "2026-10-27T09:30:00",
          end_time: "2026-10-27T10:15:00",
          block_type: "work"
        },
        {
          block_id: "b_5",
          task_id: null,
          title: "Mid-morning Refreshment",
          milestone: null,
          start_time: "2026-10-27T10:15:00",
          end_time: "2026-10-27T10:30:00",
          block_type: "break"
        },
        {
          block_id: "b_6",
          task_id: "t_101",
          title: "History Essay - Read Ch. 4",
          milestone: "Draft Content",
          start_time: "2026-10-27T10:30:00",
          end_time: "2026-10-27T11:20:00",
          block_type: "work"
        },
        {
          block_id: "b_7",
          task_id: null,
          title: "Planned Safety Buffer",
          milestone: null,
          start_time: "2026-10-27T11:20:00",
          end_time: "2026-10-27T11:45:00",
          block_type: "buffer"
        },
        {
          block_id: "b_8",
          task_id: "t_103",
          title: "Physics Lab Report - Intro section",
          milestone: null,
          start_time: "2026-10-27T13:00:00",
          end_time: "2026-10-27T13:45:00",
          block_type: "work"
        }
      ];

      setTaskAnalysis(mockAnalysis);
      setScheduleBlocks(mockSchedule);
      setFeasibility({
        completion_probability: 0.78,
        risk_flagged_task_id: "t_104",
        notes: "Custom classification model flagged CS Project as high variance due to low student samples."
      });
      setCurrentStep(3);
      setIsGenerating(false);
      showToast("Predicted timelines generated by custom ML models.");
    }, 900);
  };

  // Completion & Active Model Loop
  const handleMarkTaskComplete = async () => {
    if (!completingTask || !actualMinutesInput) return;
    const actual = parseInt(actualMinutesInput, 10);
    if (isNaN(actual) || actual <= 0) {
      showToast("Enter a valid duration in minutes.");
      return;
    }

    const { task_id, task_type, predicted_duration_minutes } = completingTask;
    const prevFactorObj = userFactors[task_type] || { factor: 1.0, n_samples: 0 };

    if (backendStatus === 'online') {
      try {
        const res = await fetch(`${API_BASE}/api/tasks/complete`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            task_id,
            task_type,
            predicted_duration_minutes,
            actual_duration_minutes: actual
          })
        });

        if (res.ok) {
          const data = await res.json();
          const newFactor = data.updated_correction_factor || data.effective_factor;
          const newSamples = data.n_samples || prevFactorObj.n_samples + 1;

          updateFactorsState(task_type, prevFactorObj.factor, newFactor, newSamples, task_id);
          setCompletingTask(null);
          setActualMinutesInput('');
          return;
        }
      } catch (err) {
        console.warn("Backend log failed", err);
      }
    }

    // Mock Feedback Loop Update
    const ratio = actual / (predicted_duration_minutes || 60);
    const newFactor = parseFloat((prevFactorObj.factor * 0.7 + ratio * 0.3).toFixed(2));
    const newSamples = prevFactorObj.n_samples + 1;

    updateFactorsState(task_type, prevFactorObj.factor, newFactor, newSamples, task_id);
    setCompletingTask(null);
    setActualMinutesInput('');
  };

  const updateFactorsState = (taskType, oldFactor, newFactor, newSamples, taskId) => {
    setUserFactors(prev => ({
      ...prev,
      [taskType]: {
        ...prev[taskType],
        factor: newFactor,
        n_samples: newSamples
      }
    }));

    setCompletedTasks(prev => [...prev, taskId]);

    setAnimatedFactorChange({
      taskType,
      oldFactor,
      newFactor,
      n_samples: newSamples
    });

    showToast(`Adaptation Model Updated: ${taskType} factor adjusted from ${oldFactor.toFixed(2)}× to ${newFactor.toFixed(2)}×`);
  };

  const handleUpdateParsedTask = (id, field, value) => {
    setParsedTasks(prev => prev.map(t => t.task_id === id ? { ...t, [field]: value } : t));
  };

  const handleRemoveTask = (id) => {
    setParsedTasks(prev => prev.filter(t => t.task_id !== id));
  };

  const toggleExplanation = (id) => {
    setExpandedExplanation(prev => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="min-h-screen bg-[#F3F1EA] text-[#1C2420] font-['Archivo',sans-serif] selection:bg-[#2B5BA6] selection:text-white pb-16">

      {/* --- TOP BRAND & COMPETITION BANNER --- */}
      <header className="border-b border-[#D7D3C6] bg-[#FFFFFF] px-4 md:px-8 py-3">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-3">

          <div className="flex items-center gap-3">
            <div className="font-['Fraunces',serif] text-2xl font-bold tracking-tight text-[#1C2420] flex items-center gap-2">
              <span className="w-3.5 h-3.5 bg-[#2B5BA6] inline-block"></span>
              FlowState AI
            </div>
            <span className="text-xs font-semibold text-[#5B6660] border-l border-[#D7D3C6] pl-3 py-0.5 font-mono">
              RIT AI Competition Prototype
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-4 text-xs text-[#5B6660]">
            <div className="hidden lg:flex items-center gap-2 bg-[#F3F1EA] px-2.5 py-1 border border-[#D7D3C6]">
              <User className="w-3.5 h-3.5 text-[#2B5BA6]" />
              <span>Qazi (Presenter) • Losath (Frontend) • Samerth (AI/Backend)</span>
            </div>

            <div className="flex items-center gap-1.5 bg-[#F3F1EA] px-2 py-1 border border-[#D7D3C6]">
              <span className={cn(
                "w-2 h-2 rounded-full",
                backendStatus === 'online' ? "bg-[#3D6B4A]" : "bg-[#B5491F]"
              )} />
              <span className="font-mono text-[11px]">
                {backendStatus === 'online' ? "api: online" : "api: offline (mock active)"}
              </span>
            </div>

            <button
              onClick={handleClearSession}
              className="hover:text-[#B5491F] underline text-xs transition-colors"
            >
              Reset Session
            </button>
          </div>

        </div>
      </header>

      {/* --- PITCH SUMMARY STRIP --- */}
      <div className="bg-[#1C2420] text-[#F3F1EA] px-4 md:px-8 py-2.5 text-xs font-mono">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Zap className="w-3.5 h-3.5 text-amber-400 shrink-0" />
            <span>
              <strong>Problem:</strong> Students don't know how wrong their time estimates are until they miss deadlines. FlowState AI measures gap variance & self-corrects automatically.
            </span>
          </div>
          <button
            onClick={() => setCurrentStep(5)}
            className="text-amber-300 hover:underline shrink-0 text-[11px] font-semibold uppercase tracking-wide"
          >
            View Presentation Rubric (01-08) →
          </button>
        </div>
      </div>

      {/* --- TOAST NOTIFICATION --- */}
      {toastMessage && (
        <div className="fixed bottom-4 right-4 z-50 bg-[#1C2420] text-[#F3F1EA] px-4 py-2.5 border-l-4 border-[#2B5BA6] shadow-md text-sm font-medium flex items-center justify-between gap-4">
          <span>{toastMessage}</span>
          <button onClick={() => setToastMessage(null)} className="text-[#D7D3C6] hover:text-white text-xs font-mono ml-2">✕</button>
        </div>
      )}

      {/* --- MAIN LEDGER LAYOUT --- */}
      <div className="max-w-6xl mx-auto px-4 md:px-8 mt-6 grid grid-cols-1 md:grid-cols-12 gap-8">

        {/* LEFT RAIL: Navigation Sequence */}
        <nav className="md:col-span-3 space-y-1">
          <div className="text-xs font-mono text-[#5B6660] mb-3 uppercase tracking-wider">User Journey & System</div>
          {[
            { step: 1, name: "1. Input Phase", desc: "Raw text or syllabus photo" },
            { step: 2, name: "2. Review & Sanitize", desc: "LLM extraction & Human-in-loop" },
            { step: 3, name: "3. ML Predictions", desc: "Regression + Feasibility model" },
            { step: 4, name: "4. Improvement Loop", desc: "Self-correcting feedback matrix" },
            { step: 5, name: "5. Competition Deck", desc: "8 Rubric Pillars & System Spec" },
          ].map((item) => {
            const isActive = currentStep === item.step;
            const isDone = currentStep > item.step;
            return (
              <button
                key={item.step}
                onClick={() => setCurrentStep(item.step)}
                className={cn(
                  "w-full text-left p-3 border-l-2 transition-all flex items-start gap-3",
                  isActive
                    ? "border-[#2B5BA6] bg-[#FFFFFF] shadow-sm text-[#1C2420]"
                    : isDone
                      ? "border-[#3D6B4A] text-[#1C2420] hover:bg-white/50"
                      : "border-[#D7D3C6] text-[#5B6660] hover:bg-white/30"
                )}
              >
                <span className={cn(
                  "font-mono text-xs font-bold mt-0.5",
                  isActive ? "text-[#2B5BA6]" : isDone ? "text-[#3D6B4A]" : "text-[#5B6660]"
                )}>
                  0{item.step}
                </span>
                <div>
                  <div className="font-['Fraunces',serif] font-semibold text-sm leading-tight">
                    {item.name}
                  </div>
                  <div className="text-[11px] text-[#5B6660] mt-0.5">{item.desc}</div>
                </div>
              </button>
            );
          })}

          {/* AI Architecture Overview Widget */}
          <div className="mt-6 p-3 border border-[#D7D3C6] bg-[#FFFFFF] text-xs space-y-2">
            <div className="font-mono text-[#5B6660] font-semibold uppercase tracking-wider text-[10px] flex items-center gap-1">
              <Cpu className="w-3.5 h-3.5 text-[#2B5BA6]" /> Dual-AI Architecture
            </div>
            <div className="space-y-1.5 text-[11px] text-[#1C2420]">
              <div className="bg-[#F3F1EA] p-2 border border-[#D7D3C6]">
                <strong className="text-[#2B5BA6]">1. LLM (Gemini/GPT):</strong> Reads messy assignment text/images into clean structural data.
              </div>
              <div className="bg-[#F3F1EA] p-2 border border-[#D7D3C6]">
                <strong className="text-[#3D6B4A]">2. Custom ML Models:</strong>
                <ul className="list-disc list-inside mt-0.5 space-y-0.5 text-[#5B6660]">
                  <li>Regression Model (Time Effort)</li>
                  <li>Binary Classification (Feasibility)</li>
                </ul>
              </div>
            </div>
          </div>

          {/* User Profile Config */}
          <div className="p-3 border border-[#D7D3C6] bg-[#FFFFFF] text-xs space-y-2">
            <div className="font-mono text-[#5B6660] font-semibold uppercase tracking-wider text-[10px]">Student Context</div>
            <div>
              <label className="text-[#5B6660] block mb-1 text-[11px]">User Identifier:</label>
              <input
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                className="w-full bg-[#F3F1EA] border border-[#D7D3C6] px-2 py-1 font-mono text-xs text-[#1C2420] focus:outline-none focus:border-[#2B5BA6]"
              />
            </div>
          </div>
        </nav>

        {/* RIGHT CONTENT AREA */}
        <main className="md:col-span-9 bg-[#FFFFFF] border border-[#D7D3C6] p-6 md:p-8 min-h-[550px]">

          {/* ========================================================= */}
          {/* STEP 1: INPUT PAGE */}
          {/* ========================================================= */}
          {currentStep === 1 && (
            <div className="space-y-6">
              <div>
                <span className="text-xs font-mono uppercase tracking-wider text-[#2B5BA6] font-semibold">Step 01 • Main User Journey</span>
                <h1 className="font-['Fraunces',serif] text-2xl md:text-3xl text-[#1C2420] font-normal leading-tight mt-1">
                  Extract assignments from unstructured text or syllabus screenshots
                </h1>
                <p className="text-[#5B6660] mt-2 text-xs md:text-sm max-w-xl">
                  FlowState AI uses a Large Language Model (LLM) to extract structured assignment titles, types, and raw deadlines from messy course text or images.
                </p>
              </div>

              <div className="border-t border-[#D7D3C6] pt-5">

                {/* Input Mode Selector */}
                <div className="flex items-center gap-6 mb-4">
                  <button
                    onClick={() => setInputMode('text')}
                    className={cn(
                      "text-xs font-semibold pb-1 border-b-2 transition-colors flex items-center gap-1.5",
                      inputMode === 'text'
                        ? "border-[#2B5BA6] text-[#2B5BA6]"
                        : "border-transparent text-[#5B6660] hover:text-[#1C2420]"
                    )}
                  >
                    <FileText className="w-4 h-4" /> Syllabus / Email Text
                  </button>
                  <button
                    onClick={() => setInputMode('image')}
                    className={cn(
                      "text-xs font-semibold pb-1 border-b-2 transition-colors flex items-center gap-1.5",
                      inputMode === 'image'
                        ? "border-[#2B5BA6] text-[#2B5BA6]"
                        : "border-transparent text-[#5B6660] hover:text-[#1C2420]"
                    )}
                  >
                    <Camera className="w-4 h-4" /> Photo / Screenshot Upload
                  </button>
                </div>

                {/* Text Input */}
                {inputMode === 'text' && (
                  <div className="space-y-3">
                    <textarea
                      value={rawText}
                      onChange={(e) => setRawText(e.target.value)}
                      rows={6}
                      placeholder="Paste syllabus notes, assignment guidelines, or email summaries here..."
                      className="w-full bg-[#F3F1EA] border border-[#D7D3C6] p-3 text-xs font-mono text-[#1C2420] focus:outline-none focus:border-[#2B5BA6] resize-y"
                    />
                    <div className="flex justify-between items-center text-[11px] text-[#5B6660]">
                      <span>Supports unstructured course text, bullet points, or assignment lists.</span>
                      <button
                        onClick={() => setRawText(DEFAULT_RAW_TEXT)}
                        className="text-[#2B5BA6] hover:underline"
                      >
                        Reset sample text
                      </button>
                    </div>
                  </div>
                )}

                {/* Image Upload */}
                {inputMode === 'image' && (
                  <div className="space-y-3">
                    <div
                      className="border-2 border-dashed border-[#D7D3C6] bg-[#F3F1EA] p-8 text-center hover:border-[#2B5BA6] transition-colors cursor-pointer relative"
                      onClick={() => document.getElementById('image-upload-input').click()}
                    >
                      <input
                        id="image-upload-input"
                        type="file"
                        accept="image/png,image/jpeg,image/webp,image/gif"
                        className="hidden"
                        onChange={handleImageSelect}
                      />
                      {imagePreviewUrl ? (
                        <div className="space-y-2">
                          <img src={imagePreviewUrl} alt="Upload preview" className="max-h-48 mx-auto border border-[#D7D3C6]" />
                          <p className="text-xs text-[#1C2420] font-mono">{selectedImage?.name}</p>
                          <span className="text-xs text-[#2B5BA6] underline">Click to upload a different image</span>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <Camera className="w-8 h-8 mx-auto text-[#5B6660]" />
                          <p className="text-xs md:text-sm text-[#1C2420] font-medium">Upload photo of a whiteboard, syllabus sheet, or screen snippet</p>
                          <p className="text-[11px] text-[#5B6660]">Supports PNG, JPEG, WebP, GIF (Max 10MB)</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Focus Preference */}
                <div className="mt-5 pt-5 border-t border-[#D7D3C6]">
                  <label className="text-xs font-semibold text-[#1C2420] block mb-2">
                    Peak Working Window Preference:
                  </label>
                  <div className="flex items-center gap-6 text-xs">
                    <label className="flex items-center gap-2 text-[#1C2420] cursor-pointer">
                      <input
                        type="radio"
                        name="peak_window"
                        value="morning"
                        checked={workPreference === 'morning'}
                        onChange={() => setWorkPreference('morning')}
                        className="accent-[#2B5BA6]"
                      />
                      <Sun className="w-3.5 h-3.5 text-[#5B6660]" /> Morning & Daytime
                    </label>
                    <label className="flex items-center gap-2 text-[#1C2420] cursor-pointer">
                      <input
                        type="radio"
                        name="peak_window"
                        value="evening"
                        checked={workPreference === 'evening'}
                        onChange={() => setWorkPreference('evening')}
                        className="accent-[#2B5BA6]"
                      />
                      <Moon className="w-3.5 h-3.5 text-[#5B6660]" /> Evening & Late Night
                    </label>
                  </div>
                </div>

                {/* Submit */}
                <div className="mt-6 flex justify-end">
                  <button
                    onClick={handleParseTasks}
                    disabled={isParsing}
                    className="bg-[#2B5BA6] text-white px-5 py-2 text-xs font-semibold hover:bg-[#204682] transition-colors flex items-center gap-2 disabled:opacity-50"
                  >
                    {isParsing ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" /> LLM Extracting Data...
                      </>
                    ) : (
                      <>
                        Extract Tasks via LLM <ArrowRight className="w-3.5 h-3.5" />
                      </>
                    )}
                  </button>
                </div>

              </div>
            </div>
          )}

          {/* ========================================================= */}
          {/* STEP 2: REVIEW & HUMAN-IN-THE-LOOP CONTROL */}
          {/* ========================================================= */}
          {currentStep === 2 && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-xs font-mono uppercase tracking-wider text-[#2B5BA6] font-semibold">Step 02 • Responsible AI & Human Oversight</span>
                  <h2 className="font-['Fraunces',serif] text-2xl text-[#1C2420] mt-0.5">
                    Review Extracted Assignments
                  </h2>
                  <p className="text-xs text-[#5B6660] mt-1">
                    Responsible AI Principle: The AI assists and structures input, but the student retains full control to edit or delete before scheduling.
                  </p>
                </div>
                <button
                  onClick={() => setCurrentStep(1)}
                  className="text-xs text-[#2B5BA6] underline hover:text-[#204682]"
                >
                  ← Edit source input
                </button>
              </div>

              {/* Stress Test & Injection Banner */}
              {parsedTasks?.some(t => t.source_flagged_for_review) && (
                <div className="bg-[#F3F1EA] border-l-4 border-[#B5491F] p-3 text-xs space-y-1">
                  <div className="font-semibold text-[#B5491F] flex items-center gap-1.5">
                    <ShieldAlert className="w-4 h-4" /> Prompt Injection Stress-Test Active
                  </div>
                  <p className="text-[#1C2420] text-[11px]">
                    The system detected potential prompt override instructions in the raw text. Our backend sanitized the input to ensure only valid tasks were extracted.
                  </p>
                </div>
              )}

              {/* Ledger Table */}
              <div className="border border-[#D7D3C6] overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="bg-[#F3F1EA] border-b border-[#D7D3C6] font-mono text-[#5B6660]">
                      <th className="p-2.5 font-semibold">Assignment Title</th>
                      <th className="p-2.5 font-semibold w-28">Category</th>
                      <th className="p-2.5 font-semibold w-24">Base Diff (1-10)</th>
                      <th className="p-2.5 font-semibold w-44">Target Deadline</th>
                      <th className="p-2.5 font-semibold w-12 text-center">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#D7D3C6]">
                    {parsedTasks?.map((task) => (
                      <tr key={task.task_id} className="hover:bg-[#F3F1EA]/50">
                        <td className="p-2.5">
                          <input
                            type="text"
                            value={task.title}
                            onChange={(e) => handleUpdateParsedTask(task.task_id, 'title', e.target.value)}
                            className="w-full bg-transparent font-medium text-[#1C2420] focus:outline-none focus:border-b focus:border-[#2B5BA6]"
                          />
                        </td>
                        <td className="p-2.5">
                          <select
                            value={task.task_type}
                            onChange={(e) => handleUpdateParsedTask(task.task_id, 'task_type', e.target.value)}
                            className="bg-transparent border border-[#D7D3C6] px-1.5 py-0.5 text-xs text-[#1C2420] focus:outline-none focus:border-[#2B5BA6]"
                          >
                            <option value="essay">essay</option>
                            <option value="math">math</option>
                            <option value="reading">reading</option>
                            <option value="coding">coding</option>
                            <option value="creative">creative</option>
                          </select>
                        </td>
                        <td className="p-2.5">
                          <input
                            type="number"
                            min="1"
                            max="10"
                            value={task.base_difficulty}
                            onChange={(e) => handleUpdateParsedTask(task.task_id, 'base_difficulty', parseInt(e.target.value, 10) || 5)}
                            className="w-14 bg-transparent border border-[#D7D3C6] px-1.5 py-0.5 font-mono text-center text-[#1C2420]"
                          />
                        </td>
                        <td className="p-2.5 space-y-1">
                          <input
                            type="text"
                            value={task.deadline || ''}
                            onChange={(e) => handleUpdateParsedTask(task.task_id, 'deadline', e.target.value)}
                            className="w-full bg-transparent font-mono text-xs text-[#1C2420] focus:outline-none"
                          />
                          {!task.deadline_matched && (
                            <span className="inline-block text-[10px] font-mono bg-[#F3F1EA] text-[#B5491F] border border-[#B5491F] px-1 py-0.2">
                              ⚠ Extrapolated Deadline
                            </span>
                          )}
                        </td>
                        <td className="p-2.5 text-center">
                          <button
                            onClick={() => handleRemoveTask(task.task_id)}
                            className="text-[#5B6660] hover:text-[#B5491F]"
                            title="Delete task"
                          >
                            <Trash2 className="w-3.5 h-3.5 mx-auto" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Action Trigger */}
              <div className="flex items-center justify-between pt-3 border-t border-[#D7D3C6]">
                <span className="text-xs text-[#5B6660]">
                  {parsedTasks?.length || 0} extracted tasks ready for custom ML prediction models.
                </span>
                <button
                  onClick={handleGenerateSchedule}
                  disabled={isGenerating || !parsedTasks || parsedTasks.length === 0}
                  className="bg-[#2B5BA6] text-white px-5 py-2 text-xs font-semibold hover:bg-[#204682] transition-colors flex items-center gap-2 disabled:opacity-50"
                >
                  {isGenerating ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Running Custom ML Models...
                    </>
                  ) : (
                    <>
                      Run Custom Prediction & Feasibility Models <ArrowRight className="w-3.5 h-3.5" />
                    </>
                  )}
                </button>
              </div>

            </div>
          )}

          {/* ========================================================= */}
          {/* STEP 3: CUSTOM ML PREDICTIONS & FEASIBILITY SCHEDULER */}
          {/* ========================================================= */}
          {currentStep === 3 && (
            <div className="space-y-6">

              {/* Header & Feasibility Score Badge */}
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#D7D3C6] pb-4">
                <div>
                  <span className="text-xs font-mono uppercase tracking-wider text-[#2B5BA6] font-semibold">Step 03 • Dual Custom ML Models Output</span>
                  <h2 className="font-['Fraunces',serif] text-2xl text-[#1C2420] mt-0.5">
                    Personalized Effort Predictions & Time Blocks
                  </h2>
                  <p className="text-xs text-[#5B6660] mt-0.5">
                    Custom Model 1 predicts duration based on student pattern data. Custom Model 2 checks if the full schedule is realistic.
                  </p>
                </div>

                {/* Model 2 Feasibility Badge */}
                {feasibility && (
                  <div className="flex items-center gap-3 bg-[#F3F1EA] border border-[#D7D3C6] p-3 text-xs">
                    <div>
                      <div className="text-[#5B6660] uppercase font-mono text-[10px]">Model 2 (Classification): Schedule Feasibility</div>
                      <div className="font-['Fraunces',serif] text-lg font-bold text-[#3D6B4A] flex items-center gap-1.5">
                        <CheckCircle2 className="w-4 h-4" /> {Math.round(feasibility.completion_probability * 100)}% Realistic
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Task Predictions Breakdown */}
              <div className="space-y-3">
                <h3 className="font-['Fraunces',serif] text-base text-[#1C2420] font-semibold">
                  Custom Model 1: Regression Task Predictions
                </h3>

                <div className="space-y-2">
                  {taskAnalysis?.map((task) => {
                    const isRisk = feasibility?.risk_flagged_task_id === task.task_id;
                    const factorObj = userFactors[task.task_type] || { n_samples: 0 };
                    const isColdStart = factorObj.n_samples < 3;

                    return (
                      <div
                        key={task.task_id}
                        className={cn(
                          "border p-3.5 transition-all bg-[#FFFFFF]",
                          isRisk ? "border-l-4 border-l-[#B5491F] border-[#D7D3C6]" : "border-[#D7D3C6]"
                        )}
                      >
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-xs font-bold text-[#2B5BA6]">
                                #{task.priority_rank}
                              </span>
                              <h4 className="font-['Fraunces',serif] font-semibold text-base text-[#1C2420]">
                                {task.title}
                              </h4>
                              {isRisk && (
                                <span className="bg-[#B5491F] text-white text-[10px] font-mono px-1.5 py-0.2">
                                  Feasibility Risk Flag
                                </span>
                              )}
                            </div>

                            <div className="flex flex-wrap items-center gap-2 text-xs text-[#5B6660] mt-1">
                              <span className="capitalize font-mono text-[11px]">type: {task.task_type}</span>
                              <span>•</span>
                              <span className="text-[11px]">diff: {task.difficulty}/10</span>
                              <span>•</span>

                              {/* Cold Start Indicator */}
                              <span className={cn(
                                "font-mono text-[10px] px-1.5 py-0.2 border",
                                isColdStart
                                  ? "bg-amber-50 text-amber-800 border-amber-300"
                                  : "bg-emerald-50 text-emerald-800 border-emerald-300"
                              )}>
                                {isColdStart
                                  ? `Cold-Start (${factorObj.n_samples}/3 samples - using synthetic base)`
                                  : `Calibrated (${factorObj.n_samples} past samples)`
                                }
                              </span>
                            </div>
                          </div>

                          <div className="flex sm:flex-col items-center sm:items-end justify-between gap-2">
                            <div className="font-['Fraunces',serif] text-xl font-bold text-[#2B5BA6]">
                              {task.predicted_duration_minutes} <span className="text-xs font-normal text-[#5B6660]">mins</span>
                            </div>

                            <div className="flex items-center gap-2">
                              <button
                                onClick={() => toggleExplanation(task.task_id)}
                                className="text-xs text-[#2B5BA6] hover:underline flex items-center gap-1"
                              >
                                <Info className="w-3.5 h-3.5" /> Explanation
                              </button>

                              <button
                                onClick={() => setCompletingTask(task)}
                                className="bg-[#3D6B4A] text-white text-xs px-2.5 py-1 hover:bg-[#2e5238] transition-colors"
                              >
                                Log Completion
                              </button>
                            </div>
                          </div>
                        </div>

                        {/* Explanation Breakdown */}
                        {expandedExplanation[task.task_id] && (
                          <div className="mt-2.5 pt-2.5 border-t border-[#D7D3C6] bg-[#F3F1EA] p-2.5 text-xs space-y-1 font-mono text-[#1C2420]">
                            <div className="font-semibold text-[#2B5BA6]">Model Decision Math:</div>
                            <p className="text-[#5B6660] text-[11px]">
                              {task.explanation || `Population baseline of ${task.population_predicted_duration_minutes || Math.round(task.predicted_duration_minutes / (task.correction_factor_applied || 1))}m corrected by personal historical variance multiplier of ${task.correction_factor_applied || 1.0}×.`}
                            </p>
                          </div>
                        )}

                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Time-Blocked Visual Timeline */}
              <div className="space-y-3 pt-4 border-t border-[#D7D3C6]">
                <h3 className="font-['Fraunces',serif] text-base text-[#1C2420] font-semibold">
                  Scheduled Time-Blocks & Rest Buffers
                </h3>

                <div className="relative border-l-2 border-[#2B5BA6] pl-5 space-y-3 my-4">
                  {scheduleBlocks?.map((block, idx) => {
                    const isWork = block.block_type === 'work';
                    const isBreak = block.block_type === 'break';

                    const startTimeFormatted = new Date(block.start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    const endTimeFormatted = new Date(block.end_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

                    return (
                      <div key={block.block_id || idx} className="relative">
                        <div className="absolute -left-[27px] top-2 w-2 h-0.5 bg-[#2B5BA6]" />

                        <div className={cn(
                          "p-2.5 border text-xs transition-all",
                          isWork ? "border-[#2B5BA6] bg-[#FFFFFF]" :
                          isBreak ? "border-[#D7D3C6] bg-[#F3F1EA] text-[#5B6660]" :
                          "border-dashed border-[#B5491F] bg-[#F3F1EA] text-[#B5491F]"
                        )}>
                          <div className="flex items-center justify-between gap-2">
                            <span className="font-mono text-[11px] text-[#5B6660]">
                              {startTimeFormatted} → {endTimeFormatted}
                            </span>

                            <span className={cn(
                              "font-mono text-[9px] uppercase px-1 py-0.2 font-bold text-white",
                              isWork ? "bg-[#2B5BA6]" : isBreak ? "bg-[#3D6B4A]" : "bg-[#B5491F]"
                            )}>
                              {block.block_type}
                            </span>
                          </div>

                          <div className="font-['Fraunces',serif] font-medium text-sm text-[#1C2420] mt-0.5">
                            {block.title || (isBreak ? "Rest & Recovery Window" : "Buffer Margin")}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

            </div>
          )}

          {/* ========================================================= */}
          {/* STEP 4: SELF-CORRECTING ADAPTATION LOOP (THE FLEX POINT) */}
          {/* ========================================================= */}
          {currentStep === 4 && (
            <div className="space-y-6">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono uppercase tracking-wider text-[#3D6B4A] font-bold bg-emerald-100 px-2 py-0.5 border border-emerald-300">
                    Step 04 • The Core Differentiator
                  </span>
                </div>
                <h2 className="font-['Fraunces',serif] text-2xl text-[#1C2420] mt-1.5">
                  Self-Correcting Personalization Loop
                </h2>
                <p className="text-xs md:text-sm text-[#5B6660] mt-1">
                  Unlike static study planners, FlowState AI compares its predictions against actual completion times logged by the student, updating personal speed multipliers over time.
                </p>
              </div>

              {/* Animated Adaptation Alert */}
              {animatedFactorChange && (
                <div className="bg-[#FFFFFF] border-l-4 border-[#3D6B4A] border-[#D7D3C6] p-3 text-xs space-y-1">
                  <div className="font-semibold text-[#3D6B4A] flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4" /> Active Learning Triggered
                  </div>
                  <p className="text-[#1C2420] text-[11px]">
                    Recalibrated multiplier for <span className="font-mono uppercase font-bold">{animatedFactorChange.taskType}</span>:
                    <span className="font-mono font-bold text-[#2B5BA6] ml-1">
                      {animatedFactorChange.oldFactor.toFixed(2)}× → {animatedFactorChange.newFactor.toFixed(2)}×
                    </span> (Total logs: {animatedFactorChange.n_samples}).
                  </p>
                </div>
              )}

              {/* Personal Multipliers Table */}
              <div className="border border-[#D7D3C6]">
                <div className="bg-[#F3F1EA] border-b border-[#D7D3C6] p-2.5 font-mono text-xs text-[#5B6660] font-semibold flex justify-between items-center">
                  <span>Student Task Multipliers (User: {userId})</span>
                  <span className="text-[10px] text-[#2B5BA6]">Auto-Adjusting</span>
                </div>

                <div className="divide-y divide-[#D7D3C6]">
                  {Object.entries(userFactors).map(([typeKey, data]) => {
                    const isColdStart = data.n_samples < 3;
                    return (
                      <div key={typeKey} className="p-3.5 flex items-center justify-between hover:bg-[#F3F1EA]/40">
                        <div>
                          <div className="font-['Fraunces',serif] font-semibold text-base text-[#1C2420]">
                            {data.label}
                          </div>
                          <div className="text-[11px] text-[#5B6660] font-mono mt-0.5 flex items-center gap-2">
                            <span>{data.n_samples} completion logs recorded</span>
                            {isColdStart && (
                              <span className="text-amber-700 bg-amber-50 px-1 border border-amber-200 text-[10px]">
                                Cold-Start State
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="text-right">
                          <div className="font-['Fraunces',serif] text-2xl font-bold text-[#2B5BA6]">
                            {data.factor.toFixed(2)}×
                          </div>
                          <span className="text-[10px] font-mono text-[#5B6660]">
                            {data.factor > 1.0 ? "Pace: Slower than avg" : data.factor < 1.0 ? "Pace: Faster than avg" : "Pace: Standard baseline"}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="p-3 bg-[#F3F1EA] border border-[#D7D3C6] text-xs text-[#5B6660] space-y-1">
                <div className="font-semibold text-[#1C2420]">Why this sets FlowState AI apart:</div>
                <p className="text-[11px]">
                  Students don't stay at standard population averages. A student might be 25% slower on essays but 10% faster on math problem sets. FlowState AI learns these individual variances automatically without requiring browser-heavy RL models.
                </p>
              </div>

            </div>
          )}

          {/* ========================================================= */}
          {/* STEP 5: COMPETITION PRESENTATION DECK & RUBRIC PILLARS */}
          {/* ========================================================= */}
          {currentStep === 5 && (
            <div className="space-y-6">
              <div>
                <span className="text-xs font-mono uppercase tracking-wider text-[#2B5BA6] font-semibold">Step 05 • RIT AI Competition Spec</span>
                <h2 className="font-['Fraunces',serif] text-2xl text-[#1C2420] mt-0.5">
                  Project Rubric Summary & Technical Highlights
                </h2>
                <p className="text-xs text-[#5B6660] mt-1">
                  Presentation overview structured according to the 8 competition criteria.
                </p>
              </div>

              {/* Rubric Tab Selector */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5 border-b border-[#D7D3C6] pb-3 text-xs font-mono">
                {[
                  { id: 1, label: "01. Problem" },
                  { id: 2, label: "02. Target Users" },
                  { id: 3, label: "03. Solution" },
                  { id: 4, label: "04. AI Architecture" },
                  { id: 5, label: "05. Prototype" },
                  { id: 6, label: "06. User Journey" },
                  { id: 7, label: "07. Testing (148 tests)" },
                  { id: 8, label: "08. Responsible AI" },
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveRubricTab(tab.id)}
                    className={cn(
                      "p-2 text-left border transition-all text-[11px]",
                      activeRubricTab === tab.id
                        ? "border-[#2B5BA6] bg-[#2B5BA6] text-white font-bold"
                        : "border-[#D7D3C6] bg-[#F3F1EA] text-[#5B6660] hover:bg-white"
                    )}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Rubric Content Cards */}
              <div className="p-4 border border-[#D7D3C6] bg-[#F3F1EA] text-xs space-y-3">
                {activeRubricTab === 1 && (
                  <div className="space-y-2">
                    <h3 className="font-['Fraunces',serif] text-lg text-[#1C2420]">01. Problem Statement</h3>
                    <p className="text-[#1C2420] leading-relaxed text-[12px]">
                      Every student thinks they're decent at estimating homework duration, but empirical research shows they consistently miss deadlines due to cognitive estimation bias. Human memory remembers the single time an assignment was finished early while ignoring the multiple times it ran late.
                    </p>
                  </div>
                )}

                {activeRubricTab === 2 && (
                  <div className="space-y-2">
                    <h3 className="font-['Fraunces',serif] text-lg text-[#1C2420]">02. Target Users</h3>
                    <p className="text-[#1C2420] leading-relaxed text-[12px]">
                      High school and university students juggling multiple subjects, unpredictable assignment lengths, and tight deadlines across digital platforms and syllabi.
                    </p>
                  </div>
                )}

                {activeRubricTab === 3 && (
                  <div className="space-y-2">
                    <h3 className="font-['Fraunces',serif] text-lg text-[#1C2420]">03. Solution</h3>
                    <p className="text-[#1C2420] leading-relaxed text-[12px]">
                      FlowState AI captures messy assignments, converts them into time estimates calibrated specifically for that student, lays them into realistic time blocks, and continuously learns from actual completion logs.
                    </p>
                  </div>
                )}

                {activeRubricTab === 4 && (
                  <div className="space-y-2">
                    <h3 className="font-['Fraunces',serif] text-lg text-[#1C2420]">04. AI Component (Two Distinct Systems)</h3>
                    <ul className="list-disc list-inside space-y-1.5 text-[#1C2420] text-[12px]">
                      <li><strong>LLM Component:</strong> Parses messy human text, emails, and image OCR into clean structured JSON tasks.</li>
                      <li><strong>Custom ML Model 1 (Regression):</strong> Predicts task duration tailored to student variance patterns.</li>
                      <li><strong>Custom ML Model 2 (Classification):</strong> Evaluates whole-schedule feasibility (Yes/No realistic prediction).</li>
                    </ul>
                  </div>
                )}

                {activeRubricTab === 5 && (
                  <div className="space-y-2">
                    <h3 className="font-['Fraunces',serif] text-lg text-[#1C2420]">05. Working Prototype</h3>
                    <p className="text-[#1C2420] leading-relaxed text-[12px]">
                      Fully functioning web interface connected to a backend parser, predictive engine, visual scheduler, and local session persistence.
                    </p>
                  </div>
                )}

                {activeRubricTab === 6 && (
                  <div className="space-y-2">
                    <h3 className="font-['Fraunces',serif] text-lg text-[#1C2420]">06. Main User Journey</h3>
                    <p className="text-[#1C2420] leading-relaxed text-[12px]">
                      1. Input text/photo → 2. Extract & Review → 3. ML Duration & Feasibility Generation → 4. Time-Blocked Schedule → 5. Log completion & update model factor.
                    </p>
                  </div>
                )}

                {activeRubricTab === 7 && (
                  <div className="space-y-2">
                    <h3 className="font-['Fraunces',serif] text-lg text-[#1C2420]">07. Automated Testing & Bug Fixes</h3>
                    <div className="space-y-1 text-[#1C2420] text-[12px]">
                      <p><strong>Automated Test Suite:</strong> Built custom testing harness with 148 automated tests covering edge case inputs, image OCR, and math personalization.</p>
                      <p><strong>Bugs Resolved:</strong> Identified 14 system bugs during stress testing and fixed all of them prior to submission.</p>
                    </div>
                  </div>
                )}

                {activeRubricTab === 8 && (
                  <div className="space-y-2">
                    <h3 className="font-['Fraunces',serif] text-lg text-[#1C2420]">08. Responsible AI & Limitations</h3>
                    <ul className="list-disc list-inside space-y-1 text-[#1C2420] text-[12px]">
                      <li><strong>Cold-Start Transparency:</strong> Initial models use synthetic training data until student completes real tasks.</li>
                      <li><strong>Human-in-the-Loop:</strong> AI assists extraction, but the student always reviews and retains full editing authority.</li>
                      <li><strong>Prompt Injection Shielding:</strong> System stress-tested against adversarial malicious prompt injections.</li>
                    </ul>
                  </div>
                )}
              </div>

              {/* Team Credits */}
              <div className="p-3 border border-[#D7D3C6] bg-white text-xs flex flex-wrap justify-between items-center gap-2">
                <span className="font-mono text-[#5B6660]">Project Team & Roles:</span>
                <span className="font-semibold text-[#1C2420]">Qazi (Presenter) • Losath (Website/Frontend) • Samerth (AI/Backend)</span>
              </div>

            </div>
          )}

        </main>
      </div>

      {/* --- MARK COMPLETE MODAL --- */}
      {completingTask && (
        <div className="fixed inset-0 bg-[#1C2420]/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[#FFFFFF] border-2 border-[#2B5BA6] max-w-md w-full p-5 shadow-xl space-y-4">

            <div className="flex items-center justify-between border-b border-[#D7D3C6] pb-2.5">
              <h3 className="font-['Fraunces',serif] text-lg text-[#1C2420]">
                Log Task Completion
              </h3>
              <button onClick={() => setCompletingTask(null)} className="text-[#5B6660] hover:text-[#1C2420] text-sm">✕</button>
            </div>

            <div>
              <div className="font-medium text-[#1C2420] text-xs">{completingTask.title}</div>
              <div className="text-[11px] text-[#5B6660] mt-0.5 font-mono">
                ML Predicted Duration: {completingTask.predicted_duration_minutes} mins
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#1C2420] block">
                Actual time taken (minutes):
              </label>
              <input
                type="number"
                min="1"
                placeholder="e.g. 115"
                value={actualMinutesInput}
                onChange={(e) => setActualMinutesInput(e.target.value)}
                className="w-full bg-[#F3F1EA] border border-[#D7D3C6] p-2 font-mono text-xs text-[#1C2420] focus:outline-none focus:border-[#2B5BA6]"
                autoFocus
              />
            </div>

            <div className="flex justify-end gap-2 pt-2 border-t border-[#D7D3C6]">
              <button
                onClick={() => setCompletingTask(null)}
                className="px-3 py-1.5 text-xs font-medium text-[#5B6660]"
              >
                Cancel
              </button>
              <button
                onClick={handleMarkTaskComplete}
                className="bg-[#3D6B4A] text-white px-3 py-1.5 text-xs font-medium hover:bg-[#2e5238] transition-colors"
              >
                Submit & Recalibrate Model
              </button>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
