const API_BASE = 'http://localhost:8000';

const apiService = {
  async startSession(resume, mode) {
    const res = await fetch(`${API_BASE}/api/session/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ resume, mode })
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async getQuestion(sessionId, index) {
    const res = await fetch(`${API_BASE}/api/question/${sessionId}/${index}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async evaluateAnswer(sessionId, index, transcript) {
    const res = await fetch(`${API_BASE}/api/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        question_index: index,
        transcript: transcript || '(no answer provided)'
      })
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async getReport(sessionId) {
    const res = await fetch(`${API_BASE}/api/report/${sessionId}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
};

export default apiService;