# AI-Interview-Stimulator
A high-performance, Mixture of Experts (MoE) backend designed to conduct FAANG-level technical interviews. It parses resumes, dynamically retrieves scenario-based questions from a vector database, and evaluates candidate responses across core domains (DSA, OS, CN, DBMS, SD) in real-time.


---

## 🏗️ System Architecture

Rather than acting as a standard monolithic LLM API wrapper, this system separates semantic processing, text classification, and adaptive routing loops into decoupled algorithmic boundaries.

* **Resume Parser Pass:** Extracts engineering domains and professional milestones via targeted JSON extraction layers.
* **Dynamic MoE Generation Loop:** Maps core systems topics (`OS`, `CN`, `DBMS`, `SD`) directly to the candidate's custom codebase context.
* **Conditional Infrastructure Routing:** Completely bypasses live LLM inference overhead by deploying local database traps if domain familiarity isn't flagged.
* **Semantic Scoring Engine:** Utilizes local embeddings and cosine distance metrics to categorize algorithmic code blocks mathematically against reference patterns.

---

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI (Python 3.10+)
* **Vector Database Engine:** ChromaDB (Embedded Persistent Local Instance)
* **Embedding Model Vectorization:** `all-MiniLM-L6-v2` (Sentence Transformers via HuggingFace)
* **LLM Orchestration Interface:** LangChain Engine + OpenRouter API / Local Ollama Configurations
* **Frontend User Interface:** React, Tailwind CSS, Lucide Icons, Web Speech API

---

## 🚀 Local Installation & Deployment

### 1. Backend Setup

Clone the repository and move into the server workspace directory:
```bash
git clone [[https://github.com/yourusername/ai-interview-simulator.git](https://github.com/yourusername/ai-interview-simulator.git)](https://github.com/HarshitGupta04974/AI-Interview-Stimulator)
cd AI Interview Simulator/backend
