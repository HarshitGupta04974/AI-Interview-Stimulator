import React, { useState, useEffect, useRef } from 'react';
import { 
  Mic, Loader2, Home, Target, Code, BookOpen, 
  ChevronRight, AlertCircle, Award, Type
} from 'lucide-react';

import './App.css';
import apiService from './api_service';

export default function App() {
  const [screen, setScreen] = useState('setup'); 
  const [toast, setToast] = useState({ msg: '', visible: false });
  const [sessionId, setSessionId] = useState(null);
  const [mode, setMode] = useState('full');
  const [resume, setResume] = useState('');
  const [planData, setPlanData] = useState(null);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [currentQ, setCurrentQ] = useState(null);
  const [timerSec, setTimerSec] = useState(300);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [lastEval, setLastEval] = useState(null);

  // "speak" | "type"
  const [inputMode, setInputMode] = useState('speak');

  const timerRef = useRef(null);
  const recognitionRef = useRef(null);
  const finalTranscriptRef = useRef('');
  const shouldBeRecordingRef = useRef(false);
  const transcriptRef = useRef('');

  const updateTranscript = (val) => {
    transcriptRef.current = val;
    setTranscript(val);
  };

  const showToast = (msg) => {
    setToast({ msg, visible: true });
    setTimeout(() => setToast({ msg: '', visible: false }), 3000);
  };

  const speak = (text) => {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 0.95;
    window.speechSynthesis.speak(utt);
  };

  const handleStart = async () => {
    if (!resume.trim()) return showToast('Please paste your resume first.');
    setScreen('loading');
    try {
      const data = await apiService.startSession(resume, mode);
      setSessionId(data.session_id);
      setPlanData(data);
      setScreen('plan');
    } catch (err) {
      showToast('Error: Backend connection failed.');
      setScreen('setup');
    }
  };

  const loadQuestion = async (idx) => {
    setScreen('question');
    setCurrentIdx(idx);
    finalTranscriptRef.current = '';
    updateTranscript('');
    setInterimTranscript('');
    try {
      const data = await apiService.getQuestion(sessionId, idx);
      setCurrentQ(data);
      setTimerSec(data.time_limit_seconds);
      speak(data.question);
    } catch (err) {
      showToast('Error loading question.');
    }
  };

  const submitAnswer = async (manualTranscript) => {
    const finalAnswer = manualTranscript !== undefined
      ? manualTranscript
      : transcriptRef.current;
    clearInterval(timerRef.current);
    stopRecording();
    setScreen('eval');
    setLastEval(null);
    try {
      const data = await apiService.evaluateAnswer(sessionId, currentIdx, finalAnswer);
      setLastEval(data);
    } catch (err) {
      setLastEval({ error: err.message });
    }
  };

  // ── Input mode switching ───────────────────────────────────────────────────
  const switchInputMode = (next) => {
    if (next === inputMode) return;
    // Stop mic if switching away from speak mode
    if (inputMode === 'speak') stopRecording();
    setInputMode(next);
    setInterimTranscript('');
    // Don't clear the transcript — user may have a partial spoken answer
    // they want to continue in text, or vice versa.
  };

  // ── Speech recognition ─────────────────────────────────────────────────────
  const toggleRecording = () => {
    if (isRecording) stopRecording();
    else startRecording();
  };

  const startRecording = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return showToast('Speech recognition not supported in this browser.');

    shouldBeRecordingRef.current = true;

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = (e) => {
      let interim = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const text = e.results[i][0].transcript;
        if (e.results[i].isFinal) {
          finalTranscriptRef.current += text + ' ';
        } else {
          interim += text;
        }
      }
      updateTranscript(finalTranscriptRef.current);
      setInterimTranscript(interim);
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error', event.error);
      if (event.error !== 'no-speech') {
        showToast(`Mic error: ${event.error}`);
      }
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        shouldBeRecordingRef.current = false;
        setIsRecording(false);
      }
    };

    recognition.onend = () => {
      if (shouldBeRecordingRef.current) {
        try {
          recognition.start();
          return;
        } catch (err) {
          console.error('Failed to auto-restart recognition', err);
        }
      }
      setIsRecording(false);
    };

    try {
      recognition.start();
      recognitionRef.current = recognition;
      setIsRecording(true);
    } catch (err) {
      console.error(err);
    }
  };

  const stopRecording = () => {
    shouldBeRecordingRef.current = false;
    if (recognitionRef.current) {
      recognitionRef.current.onend = null;
      recognitionRef.current.stop();
    }
    setIsRecording(false);
    setInterimTranscript('');
  };

  useEffect(() => {
    if (screen === 'question' && currentQ) {
      timerRef.current = setInterval(() => {
        setTimerSec((prev) => {
          if (prev <= 1) {
            clearInterval(timerRef.current);
            submitAnswer(transcriptRef.current);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(timerRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [screen, currentQ]);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="app-container">
      <header className="header no-print">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Target color="#4f46e5" />
          <h1 style={{ fontSize: '18px', fontWeight: 'bold' }}>AI Interview Simulator</h1>
        </div>
        {screen !== 'setup' && (
          <button className="btn btn-outline" style={{ marginLeft: 'auto' }} onClick={() => setScreen('setup')}>
            <Home size={18} />
          </button>
        )}
      </header>

      <main className="main-content">

        {/* ── Setup ── */}
        {screen === 'setup' && (
          <div style={{ textAlign: 'center' }}>
            <h2 style={{ fontSize: '28px', marginBottom: '8px' }}>Prepare for Tech Interviews</h2>
            <p style={{ color: 'var(--text-muted)', marginBottom: '32px' }}>Choose a mode and paste your resume to start.</p>
            <div className="card">
              <h3 style={{ fontSize: '12px', color: 'var(--text-muted)', textAlign: 'left', marginBottom: '16px', letterSpacing: '1px' }}>1. INTERVIEW FOCUS</h3>
              <div className="grid-3">
                {[
                  { id: 'full', icon: Award, label: 'Full Technical' },
                  { id: 'dsa_only', icon: Code, label: 'DSA Intensive' },
                  { id: 'core_only', icon: BookOpen, label: 'Core Subjects' }
                ].map(m => (
                  <div key={m.id} className={`mode-btn ${mode === m.id ? 'active' : ''}`} onClick={() => setMode(m.id)}>
                    <m.icon size={32} color={mode === m.id ? '#6366f1' : '#475569'} style={{ marginBottom: '12px' }} />
                    <span style={{ fontWeight: 'bold' }}>{m.label}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="card" style={{ textAlign: 'left' }}>
              <h3 style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '16px', letterSpacing: '1px' }}>2. YOUR RESUME TEXT</h3>
              <textarea className="textarea" rows="10" value={resume} onChange={(e) => setResume(e.target.value)} placeholder="Paste your professional experience and projects here..." />
              <div style={{ display: 'flex', marginTop: '20px' }}>
                <button className="btn btn-primary" style={{ marginLeft: 'auto' }} onClick={handleStart}>
                  Start Now <ChevronRight size={18} />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── Loading ── */}
        {screen === 'loading' && (
          <div style={{ textAlign: 'center', padding: '100px 0' }}>
            <Loader2 className="animate-spin" size={48} color="#4f46e5" />
            <h2 style={{ marginTop: '24px' }}>Analyzing Resume...</h2>
          </div>
        )}

        {/* ── Plan ── */}
        {screen === 'plan' && planData && (
          <div className="card">
            <h2 style={{ fontSize: '24px' }}>Welcome, {planData.candidate}</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>We've generated a custom roadmap for you.</p>
            <div style={{ marginTop: '24px' }}>
              {planData.plan.map((item, i) => (
                <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: '4px', padding: '12px 0', borderBottom: '1px solid var(--border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <span style={{ color: 'var(--text-muted)', fontFamily: 'monospace' }}>0{i+1}</span>
                    <span className="badge" style={{ backgroundColor: '#1e293b', color: '#818cf8', border: '1px solid #312e81' }}>{item.subject}</span>
                    <span style={{ fontWeight: '500' }}>{item.topic}</span>
                  </div>
                  {item.routing_reason && (
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: '40px' }}>
                      {item.routing_reason}
                    </span>
                  )}
                </div>
              ))}
            </div>
            <button className="btn btn-primary" style={{ width: '100%', marginTop: '32px', justifyContent: 'center', padding: '16px' }} onClick={() => loadQuestion(0)}>
              Begin First Question
            </button>
          </div>
        )}

        {/* ── Question ── */}
        {screen === 'question' && currentQ && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                QUESTION {currentIdx + 1} OF {currentQ.total_questions}
              </span>
              <span className="badge" style={{ backgroundColor: '#1e1b4b', color: '#c7d2fe' }}>
                {currentQ.subject_label || currentQ.subject}
              </span>
            </div>

            <div className="progress-bar" style={{ marginBottom: '24px' }}>
              <div className="progress-fill" style={{ width: `${((currentIdx + 1) / currentQ.total_questions) * 100}%` }} />
            </div>

            <div className="card" style={{ borderLeft: '4px solid var(--primary)' }}>
              <h3 style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>{currentQ.topic}</h3>
              {currentQ.routing_reason && (
                <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>{currentQ.routing_reason}</p>
              )}
              <p style={{ fontSize: '20px', lineHeight: '1.5', color: 'white' }}>{currentQ.question}</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
              <div className="card">

                {/* ── Speak / Type toggle ── */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                  <h3 style={{ fontSize: '12px', color: 'var(--text-muted)' }}>YOUR ANSWER</h3>
                  <div style={{ display: 'flex', gap: '0', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border)' }}>
                    <button
                      onClick={() => switchInputMode('speak')}
                      style={{
                        display: 'flex', alignItems: 'center', gap: '6px',
                        padding: '6px 14px', fontSize: '12px', fontWeight: 'bold',
                        cursor: 'pointer', border: 'none',
                        backgroundColor: inputMode === 'speak' ? '#4f46e5' : 'transparent',
                        color: inputMode === 'speak' ? 'white' : 'var(--text-muted)',
                        transition: 'background 0.15s',
                      }}
                    >
                      <Mic size={13} /> Speak
                    </button>
                    <button
                      onClick={() => switchInputMode('type')}
                      style={{
                        display: 'flex', alignItems: 'center', gap: '6px',
                        padding: '6px 14px', fontSize: '12px', fontWeight: 'bold',
                        cursor: 'pointer', border: 'none',
                        borderLeft: '1px solid var(--border)',
                        backgroundColor: inputMode === 'type' ? '#4f46e5' : 'transparent',
                        color: inputMode === 'type' ? 'white' : 'var(--text-muted)',
                        transition: 'background 0.15s',
                      }}
                    >
                      <Type size={13} /> Type
                    </button>
                  </div>
                  {isRecording && (
                    <div className="wave-container">
                      {[1,2,3,4].map(i => <div key={i} className="wave-bar" />)}
                    </div>
                  )}
                </div>

                {/* ── Speak mode ── */}
                {inputMode === 'speak' && (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'center', padding: '20px 0' }}>
                      <button className={`record-btn ${isRecording ? 'active' : ''}`} onClick={toggleRecording}>
                        <Mic color={isRecording ? 'white' : '#64748b'} />
                        <span style={{ fontSize: '10px', fontWeight: 'bold', marginTop: '4px', color: isRecording ? 'white' : '#64748b' }}>
                          {isRecording ? 'STOP' : 'START'}
                        </span>
                      </button>
                    </div>
                    <div className="transcript-box">
                      {transcript || interimTranscript ? (
                        <p>
                          {transcript}
                          <span style={{ opacity: 0.5 }}>{interimTranscript}</span>
                        </p>
                      ) : (
                        <p style={{ color: '#475569', fontStyle: 'italic' }}>
                          Click the mic and start speaking...
                        </p>
                      )}
                    </div>
                  </>
                )}

                {/* ── Type mode ── */}
                {inputMode === 'type' && (
                  <>
                    <textarea
                      className="textarea"
                      rows="8"
                      autoFocus
                      value={transcript}
                      onChange={(e) => {
                        // In type mode the textarea IS the accumulator.
                        // Keep the ref in sync so timer-expiry submit picks it up.
                        finalTranscriptRef.current = e.target.value;
                        updateTranscript(e.target.value);
                      }}
                      placeholder="Type your answer here — explain your reasoning, approach, and trade-offs..."
                      style={{ resize: 'vertical', minHeight: '160px' }}
                    />
                    <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px', textAlign: 'right' }}>
                      {transcript.trim().split(/\s+/).filter(Boolean).length} words
                    </p>
                  </>
                )}

                <div style={{ display: 'flex', marginTop: '24px', gap: '12px' }}>
                  <button className="btn btn-outline" onClick={() => submitAnswer('')}>Skip</button>
                  <button
                    className="btn btn-primary"
                    style={{ marginLeft: 'auto' }}
                    disabled={transcript.trim().length < 5}
                    onClick={() => submitAnswer()}
                  >
                    Submit Answer
                  </button>
                </div>
              </div>

              {/* ── Timer ── */}
              <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <div className="timer-circle">
                  <svg className="timer-svg">
                    <circle cx="56" cy="56" r="50" fill="none" stroke="#1e293b" strokeWidth="6" />
                    <circle
                      cx="56" cy="56" r="50" fill="none"
                      stroke={timerSec < 30 ? '#e11d48' : '#4f46e5'}
                      strokeWidth="6" strokeDasharray="314"
                      strokeDashoffset={314 * (1 - timerSec / currentQ.time_limit_seconds)}
                      strokeLinecap="round"
                    />
                  </svg>
                  <span style={{ fontSize: '24px', fontWeight: 'bold' }}>
                    {Math.floor(timerSec / 60)}:{(timerSec % 60).toString().padStart(2, '0')}
                  </span>
                </div>
                <p style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '8px', fontWeight: 'bold' }}>TIME REMAINING</p>
                <p style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '12px', textAlign: 'center' }}>
                  {inputMode === 'type' ? 'Typing mode active' : 'Mic mode active'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* ── Eval ── */}
        {screen === 'eval' && (
          <div className="card" style={{ maxWidth: '600px', margin: '40px auto' }}>
            {!lastEval ? (
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <Loader2 className="animate-spin" size={32} color="#4f46e5" />
                <p style={{ marginTop: '16px' }}>AI is analyzing your response...</p>
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '32px' }}>
                  <div>
                    <h3 style={{ fontSize: '12px', color: 'var(--text-muted)' }}>SCORE</h3>
                    <div style={{ fontSize: '48px', fontWeight: '900' }}>
                      {lastEval.evaluation.score}
                      <span style={{ fontSize: '20px', color: '#475569' }}>/{lastEval.evaluation.max_score}</span>
                    </div>
                  </div>
                  <span className="badge" style={{ backgroundColor: '#064e3b', color: '#6ee7b7' }}>
                    {lastEval.evaluation.approach || 'Balanced'}
                  </span>
                </div>
                <div style={{ backgroundColor: 'var(--bg-dark)', padding: '20px', borderRadius: '12px', border: '1px solid var(--border)', marginBottom: '32px' }}>
                  <h4 style={{ color: '#818cf8', fontSize: '14px', marginBottom: '8px' }}>Feedback</h4>
                  <p style={{ fontSize: '14px', lineHeight: '1.6' }}>
                    {lastEval.evaluation.feedback || lastEval.evaluation.justification}
                  </p>
                </div>
                <button
                  className="btn btn-primary"
                  style={{ width: '100%', justifyContent: 'center' }}
                  onClick={() => lastEval.has_next ? loadQuestion(lastEval.next_index) : setScreen('report')}
                >
                  {lastEval.has_next ? 'Next Question' : 'View Final Report'} <ChevronRight size={18} />
                </button>
              </div>
            )}
          </div>
        )}

        {/* ── Report ── */}
        {screen === 'report' && (
          <div className="card" style={{ textAlign: 'center' }}>
            <Award size={64} color="#0d9488" style={{ marginBottom: '24px' }} />
            <h2 style={{ fontSize: '32px' }}>Interview Complete</h2>
            <p style={{ color: 'var(--text-muted)', marginBottom: '32px' }}>Your performance report is ready.</p>
            <button className="btn btn-primary" onClick={() => window.print()}>Print Scorecard</button>
            <button className="btn btn-outline" style={{ marginLeft: '12px' }} onClick={() => setScreen('setup')}>Start New Session</button>
          </div>
        )}
      </main>

      {toast.visible && (
        <div className="toast"><AlertCircle size={18} /> {toast.msg}</div>
      )}
    </div>
  );
}