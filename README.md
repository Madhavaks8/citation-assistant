# 🎓 Generative AI Source Citation Assistant

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/Citation_Assistant_Colab.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic, modular Python framework designed for **Google Colab** that grounds Generative AI responses in verifiable scholarly evidence. The system extracts atomic factual claims, searches authoritative sources (`.gov`, `.edu`, peer-reviewed journals), evaluates natural language claim entailment, attaches in-text citations (`[1]`, `[2]`), and produces formatted bibliographies in **APA**, **MLA**, or **IEEE** formats.

---

## 🎯 Conceptual Pipeline

```text
User Question
     │
     ▼
Generative AI Synthesis (Google Gemini / OpenAI)
     │
     ▼
Atomic Claim Extraction (Decomposes answer into verifiable factual assertions)
     │
     ▼
Authoritative Evidence Retrieval (.gov, .edu, UNESCO, IEEE, Nature, etc.)
     │
     ▼
Source Validation & NLI Entailment (Supported, Partially Supported, Insufficient, Unsupported)
     │
     ▼
In-Text Citation Injection ([1], [2]) & Style Formatter (APA 7th, MLA 9th, IEEE)
     │
     ▼
Interactive Gradio Web Interface
```

---

## 🚀 Key Features

* **Atomic Claim Decomposition**: Breaks long-form AI answers into individual verifiable claims.
* **Authoritative Domain Scoring**: Ranks and boosts reliable institutional sources (`.gov`, `.edu`, `who.int`, `unesco.org`, `nature.com`, `ieee.org`, `arxiv.org`, `sciencedirect.com`).
* **Entailment Verification**: Validates whether retrieved text snippets actually support each claim (`Supported`, `Partially Supported`, `Insufficient Evidence`, `Unsupported`).
* **Multi-Format Reference Formatter**: Formats bibliographies in **APA 7th**, **MLA 9th**, and **IEEE**, with direct hyperlinks and BibTeX export.
* **Google Colab Native**: Zero local setup required; runs directly in Google Colab with Gradio UI sharing.
* **Zero-Setup Search Engine**: Uses DuckDuckGo and Wikipedia APIs out-of-the-box (no mandatory search API keys needed).

---

## 📁 Project Structure

```text
citation-assistant/
├── app.py                      # Gradio web interface application
├── pipeline.py                 # Core citation pipeline orchestrator
├── llm.py                      # Unified LLM client (Gemini / OpenAI / Mock)
├── claim_extractor.py          # Atomic claim extraction & search query generator
├── source_search.py            # Reliable source search & domain authority scoring
├── source_validator.py         # Claim-evidence entailment & verification
├── citation_engine.py          # In-text citation injector & style formatter
├── config.py                   # Configuration & domain authority scores
├── requirements.txt            # Package dependencies
├── README.md                   # Documentation & guide
├── notebooks/
│   └── Citation_Assistant_Colab.ipynb  # Step-by-step Google Colab demo
└── tests/
    └── test_citations.py       # Unit and integration test suite
```

---

## ⚡ Quickstart: Running in Google Colab

1. Open [`notebooks/Citation_Assistant_Colab.ipynb`](notebooks/Citation_Assistant_Colab.ipynb) in [Google Colab](https://colab.research.google.com).
2. (Optional) In Google Colab, add your `GEMINI_API_KEY` under the 🔑 **Secrets** tab and enable notebook access.
3. Run all cells sequentially.
4. Interact with the live Gradio interface directly in the notebook output cell or click the public `gradio.live` link.

---

## 💻 Local Installation & Execution

```bash
# Clone the repository
git clone https://github.com/yourusername/citation-assistant.git
cd citation-assistant

# Install dependencies
pip install -r requirements.txt

# Run the Gradio Web App
python app.py
```

---

## 🧪 Running Automated Tests

```bash
# Run pytest test suite
pytest tests/ -v
```

---

## 📚 Citation Styles Supported

| Style | In-Text Citation | Reference List Example |
| :--- | :--- | :--- |
| **APA 7th** | `...personalized learning [1].` | `[1] UNESCO. (2023). *AI in Education*. UNESCO. https://...` |
| **MLA 9th** | `...personalized learning [1].` | `[1] UNESCO. "AI in Education." *UNESCO*, 2023, https://...` |
| **IEEE** | `...personalized learning [1].` | `[1] UNESCO, "AI in Education," *UNESCO*, 2023. [Online]. Available: https://...` |
