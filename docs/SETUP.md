# ⚙️ Setup & Deployment Guide

This guide covers running the Nexus Equity Terminal locally for development, and options for cloud deployment.

## Prerequisites
- Python 3.10 or higher
- A HuggingFace account with an active access token.

## 1. Local Development Setup

Clone the repository and navigate into it:
```bash
git clone https://github.com/khushi2704rj-sephora/eq-analytics-terminal.git
cd eq-analytics-terminal
```

Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required heavy-weight ML dependencies:
```bash
pip install -r requirements.txt
```

Set your HuggingFace environment variable. The app uses the `novita` provider for `meta-llama/Llama-3.1-8B-Instruct`.
```bash
export HF_TOKEN="your_huggingface_token_here"
```

Run the Streamlit server:
```bash
streamlit run app.py
```
The terminal will be accessible at `http://localhost:8501`.

## 2. Cloud Deployment (Streamlit Community Cloud)

Due to the memory requirements of the `sentence-transformers` embedding model and the `FAISS` vector indexer, deploying to low-tier providers (like free HuggingFace spaces, which only provide 16GB RAM) often results in server timeouts during dependency building.

**The recommended deployment path is Streamlit Community Cloud:**

1. Commit and push this repository to your GitHub account.
2. Log into [Streamlit Share](https://share.streamlit.io).
3. Click "New App" and select this repository and the `app.py` file.
4. Open the "Advanced Settings" menu during setup.
5. In the secrets manager, add your HuggingFace token:
   ```toml
   HF_TOKEN = "your_token_here"
   ```
6. Deploy. The platform will take approximately 3-5 minutes to install `torch` and `sentence-transformers` before launching the terminal.
