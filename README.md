# 🧠 LLM-Powered Legal Document Q&A System

A full-stack, containerized Retrieval-Augmented Generation (RAG) system for answering questions from legal PDFs.

Upload any legal document (e.g., GDPR), and ask questions in natural language. The app uses semantic search to find relevant parts of the document and a language model to generate grounded answers.

---

## 🔍 What It Does

- 📄 Upload legal PDFs
- 🧩 Extract and chunk document into searchable sections
- 🔍 Use FAISS and embeddings for semantic retrieval
- 🧠 Answer questions using HuggingFace's `flan-t5-base`
- 🧪 Built-in test pipeline to verify logic
- 🌐 Web UI with Streamlit (2-page interface)
- 🐳 Fully Dockerized
- ⚙️ CI with GitHub Actions

---

## 🛠 Tech Stack

| Layer            | Tool                                       |
|------------------|---------------------------------------------|
| LLM              | `gemma-2b-it` (Hugging Face Transformers)  |
| Embeddings       | `all-MiniLM-L6-v2` via SentenceTransformers |
| Vector DB        | FAISS                                       |
| PDF Parsing      | PyMuPDF                                     |
| Backend Logic    | Custom Python RAG pipeline                  |
| Frontend         | Streamlit (multipage UI)                    |
| Deployment       | Docker                                      |
| CI/CD            | GitHub Actions                              |

---

## 🚀 Getting Started
### Hugging Face Access Token
You'll need a token to access Gemma models from Hugging Face.

    -Create a .env file in the root directory:
```
HF_TOKEN=hf_your_token_here
```
    -Add it to GitHub Secrets for CI:

    	-Go to GitHub → Settings → Secrets → Actions

    	-Add secret: HF_TOKEN
### Local Run

```
python3 -m venv llm.venv
source llm.venv/bin/activate  # or llm.venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app/main.py
```
	
	
### Docker Run

```
docker build -f docker/Dockerfile -t llm-qa .
docker run --gpus all ^
  -v "C:\Users\USER\.cache\huggingface:/root/.cache/huggingface" ^
  --env-file .env ^
  -p 8501:8501 ^
  llm-qa

```
---

## 🧪 Run Tests

Run the `test_app.py` file to test the full end-to-end pipeline.  
GitHub Actions CI also runs this test on every push.

---

## ✨ Example Questions

After uploading a document like the EU GDPR:

- “What are the rights of data subjects?”
- “How does GDPR regulate cross-border data transfers?”
- “Can a company process data without consent?”

---

## 📄 License

MIT License — free to use, modify, and build on.

---