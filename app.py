import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json, os, re, time
import pandas as pd
import numpy as np
import sqlite3
from huggingface_hub import InferenceClient
from pydantic import BaseModel, field_validator
from typing import List
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from fpdf import FPDF
import base64
import random

# ─────────────────────────────────────────────
# PAGE CONFIG & ULTRA-PREMIUM STYLING
# ─────────────────────────────────────────────
st.set_page_config(page_title='Nexus Equity Terminal | Capstone', page_icon='⚡', layout='wide', initial_sidebar_state='collapsed')

st.markdown('''<style>
/* Core Terminal Reset */
.stApp {background: #050505; color: #d1d5db; font-family: "Inter", -apple-system, sans-serif;}
header {visibility: hidden;}
.st-emotion-cache-16txtl3 {padding-top: 0rem;}
.st-emotion-cache-1jicfl2 {padding: 1rem 1rem;}

/* Persistent Top Navigation Bar (Bloomberg Style) */
.terminal-nav {
    background: #0a0a0a; border-bottom: 1px solid #1f2937; padding: 12px 24px; 
    display: flex; justify-content: space-between; align-items: center; 
    position: sticky; top: 0; z-index: 999; margin-bottom: 20px;
}
.nav-brand {font-family: "JetBrains Mono", monospace; font-size: 16px; font-weight: 700; color: #fbbf24; letter-spacing: 2px;}
.nav-status {font-size: 11px; color: #10b981; font-weight: 600; font-family: "JetBrains Mono", monospace; display: flex; align-items: center;}
.nav-status::before {content: ''; display: inline-block; width: 8px; height: 8px; background-color: #10b981; border-radius: 50%; margin-right: 8px; box-shadow: 0 0 8px #10b981;}

/* Custom Tab Styling (Hide Default Streamlit Tabs, make them look like terminal panes) */
div[data-testid="stTabs"] {background: transparent;}
div[data-testid="stTabs"] button {
    background-color: #0f0f0f !important; color: #6b7280 !important; 
    border: 1px solid #1f2937 !important; border-bottom: none !important; 
    border-radius: 4px 4px 0 0 !important; padding: 8px 16px !important; 
    font-family: "JetBrains Mono", monospace; font-size: 12px !important; font-weight: 600 !important; 
    letter-spacing: 1px; margin-right: 4px;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #1a1a1a !important; color: #e5e7eb !important; 
    border-top: 2px solid #3b82f6 !important; border-bottom: 1px solid #1a1a1a !important;
}

/* Base Panel Container */
.t-panel {
    background: #111111; border: 1px solid #1f2937; border-radius: 6px; 
    padding: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5); margin-bottom: 16px;
}
.t-panel-header {
    font-size: 10px; color: #9ca3af; text-transform: uppercase; letter-spacing: 2px; 
    margin-bottom: 12px; font-weight: 700; font-family: "JetBrains Mono", monospace;
    border-bottom: 1px solid #1f2937; padding-bottom: 8px;
}

/* Metric Ticker Cards (Dense Layout) */
.metric-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px;}
.m-card {background: #0a0a0a; border: 1px solid #1f2937; border-left: 3px solid #3b82f6; border-radius: 4px; padding: 12px;}
.m-title {font-size: 10px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px;}
.m-val {font-size: 24px; font-weight: 700; color: #f3f4f6; font-family: "JetBrains Mono", monospace; margin: 4px 0;}
.m-sub {font-size: 11px; color: #10b981; display: flex; align-items: center;}
.m-sub.warn {color: #fbbf24;}
.m-sub.danger {color: #ef4444;}
.m-card.bull {border-left-color: #10b981;}
.m-card.bear {border-left-color: #ef4444;}

/* Insight Pills & Verdicts */
.verdict-box {
    background: rgba(59, 130, 246, 0.05); border: 1px solid rgba(59, 130, 246, 0.2); 
    border-left: 4px solid #3b82f6; padding: 16px; border-radius: 4px; margin-bottom: 16px;
}
.verdict-text {font-size: 15px; color: #e5e7eb; line-height: 1.6; font-weight: 500;}

.insight-list {list-style: none; padding: 0; margin: 0;}
.insight-item {
    font-size: 13px; color: #d1d5db; padding: 8px 12px; margin-bottom: 8px; 
    background: #1a1a1a; border-left: 2px solid #4b5563; border-radius: 2px;
}
.insight-item::before {content: '>'; color: #6b7280; margin-right: 8px; font-family: monospace;}

/* Streamlit Widget Overrides */
.stButton>button {
    background: #1d4ed8 !important; color: white !important; border: none !important; 
    border-radius: 4px !important; font-weight: 600 !important; font-family: "JetBrains Mono" !important; 
    text-transform: uppercase; letter-spacing: 1px; width: 100%; transition: all 0.2s;
}
.stButton>button:hover {background: #2563eb !important; box-shadow: 0 0 10px rgba(37,99,235,0.4) !important;}

/* File Uploader and Inputs */
.stFileUploader>div>div, .stTextInput>div>div>input, .stSelectbox>div>div>div {
    background: #0a0a0a !important; color: #e5e7eb !important; border: 1px solid #374151 !important; border-radius: 4px !important;
}
.stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus {border-color: #3b82f6 !important; box-shadow: none !important;}
</style>''', unsafe_allow_html=True)

# Fake top navigation bar
st.markdown('''
<div class="terminal-nav">
    <div class="nav-brand">NEXUS // EQ.ANALYTICS.TERMINAL [V3]</div>
    <div class="nav-status">SYSTEM ONLINE // LATENCY: 12ms</div>
</div>
'''  , unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ENVIRONMENT & STATE
# ─────────────────────────────────────────────
TOKEN = os.environ.get('HF_TOKEN', '')
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nexus.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS companies (
        ticker TEXT PRIMARY KEY,
        sector TEXT,
        analyst_verdict TEXT,
        business_model TEXT,
        management_tone TEXT,
        management_tone_score REAL,
        innovation REAL,
        market_position REAL,
        financial_health REAL,
        risk_profile REAL,
        wacc REAL,
        growth_rate REAL,
        fcf_base REAL,
        bull_case TEXT,
        bear_case TEXT,
        financial_highlights TEXT,
        risk_factors TEXT,
        full_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def persist_to_db(ticker, data_dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    scores = data_dict.get('competitive_scores', {})
    if not isinstance(scores, dict):
        scores = scores if isinstance(scores, dict) else {}
    c.execute('''INSERT OR REPLACE INTO companies 
        (ticker, sector, analyst_verdict, business_model, management_tone, management_tone_score,
         innovation, market_position, financial_health, risk_profile,
         wacc, growth_rate, fcf_base, bull_case, bear_case, financial_highlights, risk_factors, full_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (ticker, data_dict.get('sector', 'Unknown'),
         data_dict.get('analyst_verdict', ''), data_dict.get('business_model', ''),
         data_dict.get('management_tone', ''), data_dict.get('management_tone_score', 5),
         scores.get('innovation', 5), scores.get('market_position', 5),
         scores.get('financial_health', 5), scores.get('risk_profile', 5),
         data_dict.get('wacc', 0.08), data_dict.get('growth_rate', 0.03),
         data_dict.get('fcf_base', 1000),
         data_dict.get('bull_case', ''), data_dict.get('bear_case', ''),
         json.dumps(data_dict.get('financial_highlights', [])),
         json.dumps([r if isinstance(r, str) else r for r in data_dict.get('risk_factors', [])]),
         json.dumps(data_dict, default=str)))
    conn.commit()
    conn.close()

def load_from_db():
    if not os.path.exists(DB_PATH):
        return {}
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('SELECT ticker, full_json FROM companies')
        rows = c.fetchall()
        db = {}
        for ticker, full_json in rows:
            try:
                db[ticker] = json.loads(full_json)
            except:
                pass
        return db
    except:
        return {}
    finally:
        conn.close()

init_db()

if 'briefs' not in st.session_state: st.session_state.briefs={}
if 'contexts' not in st.session_state: st.session_state.contexts={}
if 'macro_db' not in st.session_state: st.session_state.macro_db={}

# Load from SQLite on first run (structured data pipeline)
if not st.session_state.macro_db:
    db_data = load_from_db()
    if db_data:
        st.session_state.macro_db.update(db_data)

# Load simulated universe once
if not st.session_state.macro_db and os.path.exists('macro_universe.json'):
    with open('macro_universe.json', 'r') as f:
        st.session_state.macro_db = json.load(f)

# ─────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────
class Risk(BaseModel):
    risk: str
    description: str
    severity: str
    likelihood: int
    impact: int
    @field_validator('severity')
    def sv(cls,v): return v if v in ['High','Medium','Low'] else 'Medium'

class StrategicPriority(BaseModel):
    priority: str
    detail: str
    time_horizon: str

class CompetitiveScore(BaseModel):
    innovation: int
    market_position: int
    financial_health: int
    risk_profile: int

class AnalystBrief(BaseModel):
    company_name: str
    fiscal_year: str
    sector: str
    business_model: str
    key_segments: List[str]
    financial_highlights: List[str]
    top_risks: List[Risk]
    strategic_priorities: List[StrategicPriority]
    competitive_scores: CompetitiveScore
    management_tone: str
    management_tone_score: int
    bull_case: str
    bear_case: str
    analyst_verdict: str
    wacc: float = 0.08
    growth_rate: float = 0.03
    fcf_base: float = 5000.0

# ─────────────────────────────────────────────
# BACKEND: INGESTION & LLM
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vs():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def ingest_pdf(uploaded_files, company_name, prog_func):
    emb = load_vs()
    text = ""
    total_files = len(uploaded_files)
    
    prog_func(0.1, f"Initializing Deep Scan on {total_files} documents...")
    
    for f_idx, file in enumerate(uploaded_files):
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        for i, page in enumerate(reader.pages):
            # Enforce 1.5M Char Limit to prevent local OOM
            if len(text) > 1500000:
                prog_func(0.4, f"WARNING: Context limit reached (1.5M chars). Truncating remaining pages.")
                break
                
            text += f"\n\n--- PAGE {i} ---\n\n{page.extract_text()}"
            
            # Granular Progress Math (scales from 10% to 40% of overall bar)
            base_prog = 0.1
            file_progress = (f_idx / total_files) * 0.3
            page_progress = (i / total_pages) * (0.3 / total_files)
            current_prog = base_prog + file_progress + page_progress
            prog_func(current_prog, f"Extracting File {f_idx+1}/{total_files} | Page {i}/{total_pages} ...")
            
    prog_func(0.4, "Text chunking (1000 char windows)...")
    sp = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = sp.create_documents([text], metadatas=[{'company': company_name}])
    
    prog_func(0.5, f"Vectorizing {len(docs)} text chunks into FAISS memory...")
    return FAISS.from_documents(docs, emb)

def retr(vs, co, q, k=5):
    res = vs.similarity_search(q, k=k)
    return '\n\n'.join([x.page_content for x in res])

def prompt(cn, sector, ctx):
    p = 'You are a Senior Equity Research Analyst for a top-tier investment bank.\n'
    p += f'Analyze {cn}. Return ONLY valid JSON matching this schema exactly.\n'
    p += '{\n'
    p += f'  "company_name":"{cn}","fiscal_year":"Latest","sector":"{sector}",\n'
    p += '  "business_model":"Detailed 3 sentence breakdown",\n'
    p += '  "key_segments":["segment 1","segment 2","segment 3"],\n'
    p += '  "financial_highlights":["Detail 1","Detail 2"],\n'
    p += '  "top_risks":[{"risk":"Name","description":"Detail","severity":"High","likelihood":4,"impact":5}],\n'
    p += '  "strategic_priorities":[{"priority":"Name","detail":"Detail","time_horizon":"Short-term"}],\n'
    p += '  "competitive_scores":{"innovation":8,"market_position":9,"financial_health":8,"risk_profile":5},\n'
    p += '  "management_tone":"Optimistic","management_tone_score":7,\n'
    p += '  "bull_case":"2 sentences","bear_case":"2 sentences",\n'
    p += '  "analyst_verdict":"1 sentence decisive verdict",\n'
    p += '  "wacc": 0.08, "growth_rate": 0.03, "fcf_base": 5000\n'
    p += '}\n'
    p += f'=== CONTEXT ===\n{ctx[:4000]}\n'
    return p

def run_llm(co, sector, vs, prog):
    prog(0.6, f'Retrieving top-k semantic context for {co}...')
    ctx = retr(vs, co, 'business model revenue segments strategy risks financial performance metrics')
    prog(0.7, 'Synthesizing with Llama-3.1-8B...')
    
    cl = InferenceClient(provider='novita', api_key=TOKEN)
    last_err = ""
    for attempt in range(3):
        try:
            r = cl.chat.completions.create(model='meta-llama/Llama-3.1-8B-Instruct', messages=[{'role':'user','content':prompt(co, sector, ctx)}], max_tokens=2500, temperature=0.1)
            raw = r.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*', '', raw); raw = re.sub(r'^```\s*', '', raw); raw = re.sub(r'\s*```$', '', raw)
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                raise ValueError("No valid JSON block detected from LLM.")
            b = AnalystBrief(**json.loads(m.group()))
            prog(0.9, 'Validated Schema.')
            return b, ctx
        except Exception as e:
            last_err = str(e)
            prog(0.7, f'Synthesizing... Retry {attempt+1}/3 failed ({last_err[:30]}...)')
            time.sleep(2)
            
    st.error(f"LLM Generation Failed consistently. Last Error: {last_err}")
    return None, None

def simulate_dcf(fcf, wacc, g, years=5):
    vals, pv = [], 0
    for i in range(1, years+1):
        cf = fcf * ((1 + g) ** i)
        dcf = cf / ((1 + wacc) ** i)
        vals.append(dcf)
        pv += dcf
    tv = (fcf * ((1 + g) ** years) * (1 + g)) / (wacc - g) if wacc > g else 0
    pv_tv = tv / ((1 + wacc) ** years)
    return pv + pv_tv, vals

# ─────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ─────────────────────────────────────────────
BASE_CHART = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af', family='"JetBrains Mono", monospace'))

def c_dcf_sim(vals, tv, pv, years):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f'Year {i}' for i in range(1, years+1)], y=vals, name='Projected Cash Flow', marker_color='#3b82f6', text=[f'${v:,.0f}M' for v in vals], textposition='outside', textfont=dict(color='#9ca3af', size=10)))
    fig.add_trace(go.Bar(x=['Terminal Value'], y=[tv], name='Terminal Value (Present)', marker_color='#8b5cf6', text=[f'${tv:,.0f}M'], textposition='outside', textfont=dict(color='#9ca3af', size=10)))
    fig.update_layout(**BASE_CHART, title=dict(text=f"TOTAL ESTIMATED VALUE: ${pv:,.0f}M", font=dict(color='#fbbf24', size=16)), height=400, 
                      margin=dict(l=0, r=0, t=50, b=0), yaxis=dict(gridcolor='#1f2937', zerolinecolor='#374151'), xaxis=dict(gridcolor='#1f2937'),
                      barmode='group')
    return fig

def c_radar_chart(company_name, scores_dict):
    categories = ['Innovation', 'Market Position', 'Financial Health', 'Risk Profile', 'Mgmt Tone']
    vals = [scores_dict.get('innovation', 5), scores_dict.get('market_position', 5), scores_dict.get('financial_health', 5), scores_dict.get('risk_profile', 5), scores_dict.get('management_tone_score', 5)]
    vals += vals[:1]  # close the polygon
    cats = categories + categories[:1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself', name=company_name,
                                   fillcolor='rgba(59, 130, 246, 0.2)', line=dict(color='#3b82f6', width=2)))
    fig.update_layout(**BASE_CHART, polar=dict(radialaxis=dict(visible=True, range=[0, 10], gridcolor='#1f2937', tickfont=dict(size=10, color='#6b7280')),
                                                angularaxis=dict(gridcolor='#1f2937', tickfont=dict(size=11, color='#d1d5db'))),
                      title=dict(text=f"{company_name}: COMPETITIVE PROFILE", font=dict(color='#e5e7eb', size=14)),
                      height=420, margin=dict(l=40, r=40, t=60, b=40), showlegend=False)
    return fig

def c_comparison_bars(db):
    companies = list(db.keys())
    metrics = ['Innovation', 'Financial Health', 'Risk Profile', 'Market Position']
    colors = ['#3b82f6', '#10b981', '#ef4444', '#8b5cf6']
    
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        metric_key = metric.lower().replace(' ', '_')
        vals = []
        for co in companies:
            v = db[co]
            scores = v.get('competitive_scores', {}) if isinstance(v, dict) else {}
            if isinstance(scores, dict):
                vals.append(scores.get(metric_key, 5))
            else:
                vals.append(getattr(scores, metric_key, 5))
        fig.add_trace(go.Bar(name=metric, x=companies, y=vals, marker_color=colors[i]))
        
    fig.update_layout(**BASE_CHART, barmode='group', title=dict(text="HEAD-TO-HEAD: COMPANY COMPARISON", font=dict(color='#e5e7eb', size=14)),
                      height=400, margin=dict(l=0, r=0, t=50, b=0), yaxis=dict(gridcolor='#1f2937', range=[0, 10], title='Score (out of 10)'),
                      xaxis=dict(gridcolor='#1f2937'), legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)))
    return fig



# ─────────────────────────────────────────────
# FRONTEND LAYOUT
# ─────────────────────────────────────────────
tabs = st.tabs(["📄 Upload Reports", "📊 AI Analysis", "🌍 Market Comparison", "💰 Valuation Model", "🏆 Industry Ranking"])

# --- TAB 1: INGESTION ---
with tabs[0]:
    # Welcome / How It Works Panel
    st.markdown('''
    <div class="t-panel" style="border-left: 3px solid #3b82f6;">
        <div class="t-panel-header" style="color: #3b82f6; border-bottom-color: #1e3a5f;">WELCOME — HOW THIS TOOL WORKS</div>
        <div style="color: #d1d5db; font-size: 14px; line-height: 1.8;">
            This tool uses <b>Artificial Intelligence</b> to read company financial reports (PDFs) and automatically generate professional investment analysis — no finance expertise required.<br><br>
            <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                <div style="flex:1; min-width: 140px; background:#0a0a0a; border:1px solid #1f2937; border-radius:6px; padding:12px; text-align:center;">
                    <div style="font-size: 28px;">①</div>
                    <div style="font-size: 11px; color: #9ca3af; margin-top: 6px;"><b>UPLOAD</b><br>Drop company PDF reports below</div>
                </div>
                <div style="flex:1; min-width: 140px; background:#0a0a0a; border:1px solid #1f2937; border-radius:6px; padding:12px; text-align:center;">
                    <div style="font-size: 28px;">②</div>
                    <div style="font-size: 11px; color: #9ca3af; margin-top: 6px;"><b>TAG</b><br>Name each file with a ticker & industry</div>
                </div>
                <div style="flex:1; min-width: 140px; background:#0a0a0a; border:1px solid #1f2937; border-radius:6px; padding:12px; text-align:center;">
                    <div style="font-size: 28px;">③</div>
                    <div style="font-size: 11px; color: #9ca3af; margin-top: 6px;"><b>ANALYZE</b><br>AI reads the reports & generates insights</div>
                </div>
                <div style="flex:1; min-width: 140px; background:#0a0a0a; border:1px solid #1f2937; border-radius:6px; padding:12px; text-align:center;">
                    <div style="font-size: 28px;">④</div>
                    <div style="font-size: 11px; color: #9ca3af; margin-top: 6px;"><b>EXPLORE</b><br>Switch to the other tabs to view results</div>
                </div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="t-panel">', unsafe_allow_html=True)
    st.markdown('<div class="t-panel-header">STEP 1: DOCUMENT BATCH UPLOAD</div>', unsafe_allow_html=True)
    st.info("Upload SEC 10-K or Annual Report PDFs. You can upload files for multiple different companies. In Step 2, you will map each file to its respective Asset Ticker.")
    uploaded_files = st.file_uploader("DROP PDFs HERE", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files:
        st.markdown('<div class="t-panel">', unsafe_allow_html=True)
        st.markdown('<div class="t-panel-header">STEP 2: BULK METADATA MAPPING</div>', unsafe_allow_html=True)
        
        st.markdown('''
        <div style="display:flex; font-size: 10px; color:#6b7280; margin-bottom: 8px;">
            <div style="flex:2;">DOCUMENT FILENAME</div>
            <div style="flex:1;">TICKER (Required)</div>
            <div style="flex:1;">SECTOR</div>
        </div>
        ''', unsafe_allow_html=True)
        
        file_configs = []
        for i, file in enumerate(uploaded_files):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f'<div style="font-size: 13px; color: #d1d5db; padding-top: 8px; font-family: \'JetBrains Mono\', monospace;">📄 {file.name[:40]}</div>', unsafe_allow_html=True)
            with col2:
                ticker = st.text_input("TICKER", key=f"t_{i}", placeholder="e.g. AAPL", label_visibility="collapsed")
            with col3:
                sector = st.selectbox("SECTOR", ["Technology", "Healthcare", "Financials", "Consumer", "Energy", "Industrials", "Real Estate", "Materials"], key=f"s_{i}", label_visibility="collapsed")
            file_configs.append({'file': file, 'ticker': ticker.upper().strip() if ticker else "", 'sector': sector})
            
        st.markdown("<hr style='border-color: #1f2937; margin: 20px 0;'>", unsafe_allow_html=True)
        process_btn = st.button("▶ EXECUTE BULK PIPELINE")
        st.markdown('</div>', unsafe_allow_html=True)
        
        valid_configs = [c for c in file_configs if c['ticker']]
        
        if process_btn and valid_configs:
            # Group files by ticker
            groups = {}
            for c in valid_configs:
                t = c['ticker']
                if t not in groups:
                    groups[t] = {'sector': c['sector'], 'files': []}
                groups[t]['files'].append(c['file'])
                # If sectors differ for the same ticker, the first one assigned is used.
                
            with st.container():
                st.markdown('<div class="t-panel">', unsafe_allow_html=True)
                st.markdown('<div class="t-panel-header">PIPELINE EXECUTION LOG (SEQUENTIAL BATCH)</div>', unsafe_allow_html=True)
                
                pb = st.progress(0); st_txt = st.empty()
                total_groups = len(groups)
                
                for idx, (ticker_name, data) in enumerate(groups.items()):
                    group_files = data['files']
                    group_sector = data['sector']
                    
                    def upd(v, m): 
                        # Scale local progress (v) to overall progress across all companies in batch
                        overall_v = (idx + v) / total_groups
                        pb.progress(min(overall_v, 1.0))
                        st_txt.markdown(f"<span style='color:#10b981; font-family:monospace;'>`[{overall_v*100:02.0f}%]` <b>[{ticker_name}]</b> {m}</span>", unsafe_allow_html=True)
                    
                    upd(0.05, f"Initializing pipeline for {ticker_name}...")
                    vs = ingest_pdf(group_files, ticker_name, upd)
                    st.session_state[f"vs_{ticker_name}"] = vs
                    
                    b, ctx = run_llm(ticker_name, group_sector, vs, upd)
                    if b:
                        st.session_state.briefs[ticker_name] = b
                        b_dict = b.model_dump()
                        st.session_state.macro_db[ticker_name] = b_dict
                        st.session_state.macro_db[ticker_name]['sensitivity'] = {'rates': -0.2, 'inflation': -0.4, 'supply_chain': -0.1}
                        # Persist to SQLite (Structured Data Pipeline)
                        persist_to_db(ticker_name, st.session_state.macro_db[ticker_name])
                        upd(1.0, "PROFILE GENERATED & PERSISTED TO SQL.")
                    else:
                        st.error(f"❌ PIPELINE FAILED FOR [{ticker_name}]: The LLM could not generate a valid profile.")
                        
                st.success(f"BULK INGESTION COMPLETE FOR {total_groups} ASSETS. Data persisted to nexus.db. SWITCH TO TAB [2] OR [5].")
                st.markdown(f'<div style="font-size: 11px; color: #6b7280; font-family: monospace; margin-top: 4px;">📁 SQL Database: {DB_PATH}</div>', unsafe_allow_html=True)
                time.sleep(2.5)
                st_txt.empty(); pb.empty()

                st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: TERMINAL ---
with tabs[1]:
    if not st.session_state.briefs:
        st.markdown('''
        <div class="t-panel" style="text-align:center; padding: 60px 40px; color: #6b7280;">
            <div style="font-size: 40px; margin-bottom: 16px;">📊</div>
            <div style="font-size: 16px; color: #9ca3af; font-weight: 600; margin-bottom: 12px;">AI-Generated Analyst Brief</div>
            <div style="font-size: 13px; line-height: 1.6; max-width: 500px; margin: 0 auto;">
                Once you upload a company's annual report in Tab 1, this screen will display an auto-generated investment analysis including:<br><br>
                • <b>Management Conviction Score</b> — how optimistic is the C-suite?<br>
                • <b>Financial Health & Risk Scores</b> — competitive positioning metrics<br>
                • <b>Bull & Bear Case Scenarios</b> — best and worst case outlook<br>
                • <b>Analyst Verdict</b> — a one-sentence recommendation<br><br>
                <span style="color: #fbbf24;">→ Go to Tab 1 to upload your first report.</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('<div style="display:flex; justify-content:flex-end; margin-bottom: 10px;">', unsafe_allow_html=True)
        target = st.selectbox("ACTIVE ASSET", list(st.session_state.briefs.keys()), key="t2_target", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        b = st.session_state.briefs[target]
        
        # Dense Metric Grid
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        html = f'''
        <div class="m-card bull">
            <div class="m-title">MGMT CONVICTION</div>
            <div class="m-val">{b.management_tone_score}<span style="font-size:14px;color:#6b7280">/10</span></div>
            <div class="m-sub">▲ {b.management_tone} Sentiment</div>
        </div>
        <div class="m-card">
            <div class="m-title">FINANCIAL HEALTH</div>
            <div class="m-val">{b.competitive_scores.financial_health}<span style="font-size:14px;color:#6b7280">/10</span></div>
            <div class="m-sub">Sector Benchmarked</div>
        </div>
        <div class="m-card bear">
            <div class="m-title">RISK PROFILE</div>
            <div class="m-val">{b.competitive_scores.risk_profile}<span style="font-size:14px;color:#6b7280">/10</span></div>
            <div class="m-sub danger">▼ Volatility Active</div>
        </div>
        <div class="m-card">
            <div class="m-title">INNOVATION SCORE</div>
            <div class="m-val">{b.competitive_scores.innovation}<span style="font-size:14px;color:#6b7280">/10</span></div>
            <div class="m-sub">R&D Output Index</div>
        </div>
        '''
        st.markdown(html + '</div>', unsafe_allow_html=True)
        
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="t-panel">', unsafe_allow_html=True)
            st.markdown('<div class="t-panel-header">ANALYST VERDICT // EXEC SUMMARY</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="verdict-box"><div class="verdict-text">{b.analyst_verdict}</div></div>', unsafe_allow_html=True)
            
            st.markdown('<div class="t-panel-header" style="margin-top:20px;">CORE BUSINESS MODEL</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color:#d1d5db; font-size:14px; line-height:1.6; margin-bottom: 20px;">{b.business_model}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="t-panel-header">FINANCIAL HIGHLIGHTS</div>', unsafe_allow_html=True)
            hls = ''.join([f'<li class="insight-item">{h}</li>' for h in b.financial_highlights])
            st.markdown(f'<ul class="insight-list">{hls}</ul>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with c_right:
            st.markdown('<div class="t-panel" style="border-top: 2px solid #10b981;">', unsafe_allow_html=True)
            st.markdown('<div class="t-panel-header" style="color: #10b981; border-bottom-color: #064e3b;">BULL CASE SCENARIO</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:13px; color:#d1d5db; line-height:1.5;">{b.bull_case}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="t-panel" style="border-top: 2px solid #ef4444;">', unsafe_allow_html=True)
            st.markdown('<div class="t-panel-header" style="color: #ef4444; border-bottom-color: #7f1d1d;">BEAR CASE SCENARIO</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:13px; color:#d1d5db; line-height:1.5;">{b.bear_case}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: MACRO UNIVERSE ---
with tabs[2]:
    if not st.session_state.macro_db:
        st.markdown('''
        <div class="t-panel" style="text-align:center; padding: 60px 40px; color: #6b7280;">
            <div style="font-size: 40px; margin-bottom: 16px;">🌍</div>
            <div style="font-size: 16px; color: #9ca3af; font-weight: 600; margin-bottom: 12px;">Market Stress Test Simulator</div>
            <div style="font-size: 13px; line-height: 1.6; max-width: 500px; margin: 0 auto;">
                This tab lets you simulate how real-world economic events (like interest rate hikes or inflation spikes) would impact the companies you've analyzed.<br><br>
                • A scatter plot will map all companies by <b>Innovation vs. Risk</b><br>
                • Interactive sliders let you model "what if" macro scenarios<br>
                • The estimated <b>Market Cap Variance</b> updates live<br><br>
                <span style="color: #fbbf24;">→ Upload at least one report in Tab 1 to activate this module.</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('<div class="t-panel">', unsafe_allow_html=True)
        st.markdown('<div class="t-panel-header">COMPANY COMPETITIVE PROFILE</div>', unsafe_allow_html=True)
        st.info("Select a company to see its strengths and weaknesses visualized as a radar chart. If you have analyzed 2+ companies, a side-by-side comparison chart will also appear below.")
        
        target_t3 = st.selectbox("Select Company", list(st.session_state.macro_db.keys()), key="t3_target")
        
        # Build scores dict for the selected company
        v_data = st.session_state.macro_db[target_t3]
        scores = v_data.get('competitive_scores', {}) if isinstance(v_data, dict) else {}
        mgmt = v_data.get('management_tone_score', 5) if isinstance(v_data, dict) else 5
        
        if isinstance(scores, dict):
            scores_flat = {**scores, 'management_tone_score': mgmt}
        else:
            scores_flat = {'innovation': 5, 'market_position': 5, 'financial_health': 5, 'risk_profile': 5, 'management_tone_score': mgmt}
        
        rc1, rc2 = st.columns([1, 1])
        with rc1:
            st.plotly_chart(c_radar_chart(target_t3, scores_flat), use_container_width=True)
        with rc2:
            # Interpretation
            st.markdown('<div class="t-panel" style="margin-top: 10px;">', unsafe_allow_html=True)
            st.markdown('<div class="t-panel-header">WHAT THIS MEANS</div>', unsafe_allow_html=True)
            inn_v = scores_flat.get('innovation', 5)
            fin_v = scores_flat.get('financial_health', 5)
            risk_v = scores_flat.get('risk_profile', 5)
            mkt_v = scores_flat.get('market_position', 5)
            
            def interpret(val): return '🟢 Strong' if val >= 7 else ('🟡 Average' if val >= 4 else '🔴 Weak')
            def risk_interpret(val): return '🟢 Low Risk' if val <= 3 else ('🟡 Moderate' if val <= 6 else '🔴 High Risk')
            
            st.markdown(f'''
            <div style="font-size: 13px; color: #d1d5db; line-height: 2;">
                <b>Innovation ({inn_v}/10):</b> {interpret(inn_v)} — R&D and product pipeline strength<br>
                <b>Market Position ({mkt_v}/10):</b> {interpret(mkt_v)} — competitive market dominance<br>
                <b>Financial Health ({fin_v}/10):</b> {interpret(fin_v)} — balance sheet and cash reserves<br>
                <b>Risk Profile ({risk_v}/10):</b> {risk_interpret(risk_v)} — exposure to external threats<br>
                <b>Mgmt Tone ({mgmt}/10):</b> {interpret(mgmt)} — how confident is the leadership team<br>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Comparison Chart (only if 2+ companies)
        if len(st.session_state.macro_db) >= 2:
            st.markdown('<div class="t-panel">', unsafe_allow_html=True)
            st.markdown('<div class="t-panel-header">SIDE-BY-SIDE COMPANY COMPARISON</div>', unsafe_allow_html=True)
            st.info("This chart compares all analyzed companies across 4 key dimensions. Taller bars = stronger performance (except Risk, where lower is better).")
            st.plotly_chart(c_comparison_bars(st.session_state.macro_db), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # CLASSICAL ANALYTICS: Pearson Correlation Heatmap
        if len(st.session_state.macro_db) >= 2:
            st.markdown('<div class="t-panel">', unsafe_allow_html=True)
            st.markdown('<div class="t-panel-header">CLASSICAL ANALYTICS: PEARSON CORRELATION MATRIX</div>', unsafe_allow_html=True)
            st.info("This is a statistical correlation heatmap (not AI-generated). It uses the Pearson coefficient to measure linear relationships between competitive metrics across all analyzed companies. Values close to +1 indicate strong positive correlation; values close to -1 indicate inverse correlation.")
            
            corr_data = []
            for k, v in st.session_state.macro_db.items():
                scores = v.get('competitive_scores', {}) if isinstance(v, dict) else {}
                if isinstance(scores, dict):
                    corr_data.append({
                        'Innovation': scores.get('innovation', 5),
                        'Market Position': scores.get('market_position', 5),
                        'Financial Health': scores.get('financial_health', 5),
                        'Risk Profile': scores.get('risk_profile', 5),
                        'Mgmt Tone': v.get('management_tone_score', 5) if isinstance(v, dict) else 5
                    })
            
            if len(corr_data) >= 2:
                corr_df = pd.DataFrame(corr_data)
                corr_matrix = corr_df.corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.columns.tolist(),
                    colorscale=[[0, '#ef4444'], [0.5, '#111827'], [1, '#10b981']],
                    zmin=-1, zmax=1,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont=dict(size=14, color='#e5e7eb'),
                    hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>'
                ))
                fig_corr.update_layout(**BASE_CHART, height=400, margin=dict(l=0, r=0, t=20, b=0),
                                        xaxis=dict(side='bottom'), yaxis=dict(autorange='reversed'))
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Interpretation
                st.markdown('''
                <div style="font-size: 12px; color: #9ca3af; line-height: 1.6; padding: 8px; background: #0a0a0a; border: 1px solid #1f2937; border-radius: 4px;">
                    <b style="color: #fbbf24;">How to read this:</b> Each cell shows the Pearson correlation coefficient (-1 to +1) between two metrics.
                    <b style="color: #10b981;">Green = positively correlated</b> (when one goes up, the other tends to go up).
                    <b style="color: #ef4444;">Red = negatively correlated</b> (when one goes up, the other tends to go down).
                    This is a classical statistical method, not AI-generated.
                </div>
                ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: DCF SIMULATOR ---
with tabs[3]:
    if not st.session_state.briefs:
        st.markdown('''
        <div class="t-panel" style="text-align:center; padding: 60px 40px; color: #6b7280;">
            <div style="font-size: 40px; margin-bottom: 16px;">💰</div>
            <div style="font-size: 16px; color: #9ca3af; font-weight: 600; margin-bottom: 12px;">Intrinsic Value Calculator (DCF Model)</div>
            <div style="font-size: 13px; line-height: 1.6; max-width: 500px; margin: 0 auto;">
                This tab calculates how much a company is truly worth using a <b>Discounted Cash Flow</b> model — one of the most popular methods used by Wall Street analysts.<br><br>
                • The AI auto-extracts financial inputs (cash flow, growth rate, cost of capital) from the report<br>
                • You can override any number and watch the <b>intrinsic value chart update in real-time</b><br>
                • A bar chart visualizes future cash flows and the terminal value<br><br>
                <span style="color: #fbbf24;">→ Upload at least one report in Tab 1 to activate this module.</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('<div style="display:flex; justify-content:flex-end; margin-bottom: 10px;">', unsafe_allow_html=True)
        target_dcf = st.selectbox("ACTIVE ASSET", list(st.session_state.briefs.keys()), key="t4_target", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        b = st.session_state.briefs[target_dcf]
        
        dc1, dc2 = st.columns([1, 2])
        with dc1:
            st.markdown('<div class="t-panel">', unsafe_allow_html=True)
            st.markdown('<div class="t-panel-header">DCF VARIABLES</div>', unsafe_allow_html=True)
            st.info("The Baseline Variables below were dynamically extracted by the LLM from the 10-K. Override them to model intrinsic value changes.")
            
            st.markdown('<div style="font-size: 10px; color: #6b7280; margin-bottom: 4px; margin-top: 12px;">DISCOUNT RATE (WACC)</div>', unsafe_allow_html=True)
            wacc = st.number_input("WACC", min_value=0.01, max_value=0.30, value=b.wacc, step=0.005, format="%.3f", label_visibility="collapsed", help="Weighted Average Cost of Capital. Determines the discount applied to future cash flows.")
            
            st.markdown('<div style="font-size: 10px; color: #6b7280; margin: 16px 0 4px 0;">TERMINAL GROWTH RATE</div>', unsafe_allow_html=True)
            g = st.number_input("Growth", min_value=-0.05, max_value=0.10, value=b.growth_rate, step=0.005, format="%.3f", label_visibility="collapsed", help="The expected perpetual growth rate of the asset after the projection horizon.")
            
            st.markdown('<div style="font-size: 10px; color: #6b7280; margin: 16px 0 4px 0;">BASE FCF (MILLIONS)</div>', unsafe_allow_html=True)
            fcf = st.number_input("FCF", value=float(b.fcf_base), step=100.0, label_visibility="collapsed", help="Trailing Twelve Months (TTM) Free Cash Flow baseline.")
            
            st.markdown('<div style="font-size: 10px; color: #6b7280; margin: 16px 0 4px 0;">PROJECTION HORIZON (YRS)</div>', unsafe_allow_html=True)
            years = st.slider("Years", 3, 10, 5, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with dc2:
            st.markdown('<div class="t-panel">', unsafe_allow_html=True)
            st.markdown('<div class="t-panel-header">INTRINSIC VALUE MODELER</div>', unsafe_allow_html=True)
            pv, vals = simulate_dcf(fcf, wacc, g, years)
            tv = pv - sum(vals)
            st.plotly_chart(c_dcf_sim(vals, tv, pv, years), use_container_width=True)
            
            # Plain English Interpretation
            st.markdown(f'''
            <div style="font-size: 13px; color: #d1d5db; line-height: 1.8; padding: 12px; background: #0a0a0a; border: 1px solid #1f2937; border-radius: 4px; margin-top: 8px;">
                <b style="color: #fbbf24;">What this means:</b> Based on the current inputs, the AI estimates that <b>{target_dcf}</b> is worth approximately <b style="color: #10b981;">${pv:,.0f}M</b>. 
                This is calculated by projecting {years} years of future cash flows (blue bars) and adding the present value of all cash flows beyond that horizon (purple bar = Terminal Value). 
                Try adjusting the sliders on the left to see how different assumptions change the valuation.
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # CLASSICAL ANALYTICS: Monte Carlo Simulation
        st.markdown('<div class="t-panel">', unsafe_allow_html=True)
        st.markdown('<div class="t-panel-header">CLASSICAL ANALYTICS: MONTE CARLO SIMULATION (1,000 TRIALS)</div>', unsafe_allow_html=True)
        st.info("This is a classical statistical method, not AI. It runs 1,000 randomized simulations of the DCF model by varying the WACC and Growth Rate within ±20% of your current inputs. The resulting distribution shows the range of possible valuations and the probability of each outcome.")
        
        np.random.seed(42)
        n_sims = 1000
        wacc_range = np.random.normal(wacc, wacc * 0.15, n_sims)
        wacc_range = np.clip(wacc_range, 0.01, 0.30)
        g_range = np.random.normal(g, abs(g) * 0.20 + 0.005, n_sims)
        g_range = np.clip(g_range, -0.05, 0.10)
        
        mc_results = []
        for w_i, g_i in zip(wacc_range, g_range):
            if w_i > g_i:
                pv_i, _ = simulate_dcf(fcf, w_i, g_i, years)
                mc_results.append(pv_i)
        
        mc_results = np.array(mc_results)
        mc_results = mc_results[mc_results > 0]  # filter invalid
        
        if len(mc_results) > 10:
            p5 = np.percentile(mc_results, 5)
            p50 = np.percentile(mc_results, 50)
            p95 = np.percentile(mc_results, 95)
            
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(x=mc_results, nbinsx=50, marker_color='#3b82f6', opacity=0.7, name='Simulated Valuations'))
            fig_mc.add_vline(x=p50, line_dash='dash', line_color='#fbbf24', annotation_text=f'Median: ${p50:,.0f}M', annotation_font_color='#fbbf24')
            fig_mc.add_vline(x=p5, line_dash='dot', line_color='#ef4444', annotation_text=f'5th %: ${p5:,.0f}M', annotation_font_color='#ef4444')
            fig_mc.add_vline(x=p95, line_dash='dot', line_color='#10b981', annotation_text=f'95th %: ${p95:,.0f}M', annotation_font_color='#10b981')
            fig_mc.update_layout(**BASE_CHART, height=350, margin=dict(l=0, r=0, t=20, b=0),
                                xaxis=dict(title='Estimated Valuation ($M)', gridcolor='#1f2937'),
                                yaxis=dict(title='Frequency', gridcolor='#1f2937'),
                                showlegend=False)
            st.plotly_chart(fig_mc, use_container_width=True)
            
            st.markdown(f'''
            <div style="display: flex; gap: 16px; margin-top: 8px;">
                <div style="flex:1; background: #0a0a0a; border: 1px solid #7f1d1d; border-radius: 6px; padding: 12px; text-align: center;">
                    <div style="font-size: 10px; color: #ef4444;">BEARISH CASE (5th PERCENTILE)</div>
                    <div style="font-size: 22px; color: #ef4444; font-family: 'JetBrains Mono', monospace; font-weight: bold;">${p5:,.0f}M</div>
                </div>
                <div style="flex:1; background: #0a0a0a; border: 1px solid #92400e; border-radius: 6px; padding: 12px; text-align: center;">
                    <div style="font-size: 10px; color: #fbbf24;">MEDIAN VALUATION</div>
                    <div style="font-size: 22px; color: #fbbf24; font-family: 'JetBrains Mono', monospace; font-weight: bold;">${p50:,.0f}M</div>
                </div>
                <div style="flex:1; background: #0a0a0a; border: 1px solid #064e3b; border-radius: 6px; padding: 12px; text-align: center;">
                    <div style="font-size: 10px; color: #10b981;">BULLISH CASE (95th PERCENTILE)</div>
                    <div style="font-size: 22px; color: #10b981; font-family: 'JetBrains Mono', monospace; font-weight: bold;">${p95:,.0f}M</div>
                </div>
            </div>
            <div style="font-size: 12px; color: #9ca3af; margin-top: 12px; padding: 8px; background: #0a0a0a; border: 1px solid #1f2937; border-radius: 4px;">
                <b style="color: #fbbf24;">What this means:</b> Out of 1,000 randomized scenarios, {target_dcf}'s valuation falls between 
                <b style="color: #ef4444;">${p5:,.0f}M</b> (pessimistic) and <b style="color: #10b981;">${p95:,.0f}M</b> (optimistic) with 90% confidence.
                The median estimate is <b style="color: #fbbf24;">${p50:,.0f}M</b>. This is a classical Monte Carlo simulation, not AI-generated.
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 5: INDUSTRY BENCHMARK ---
with tabs[4]:
    if not st.session_state.macro_db:
        st.markdown('''
        <div class="t-panel" style="text-align:center; padding: 60px 40px; color: #6b7280;">
            <div style="font-size: 40px; margin-bottom: 16px;">🏆</div>
            <div style="font-size: 16px; color: #9ca3af; font-weight: 600; margin-bottom: 12px;">Cross-Industry Benchmark Rankings</div>
            <div style="font-size: 13px; line-height: 1.6; max-width: 500px; margin: 0 auto;">
                This tab compares companies within the same industry to find out which ones are <b>outperforming their peers</b>.<br><br>
                • Select an industry (e.g., Technology) to see all analyzed companies in that sector<br>
                • The system automatically computes sector-wide averages for key metrics<br>
                • The <b>Quantitative Sector Matrix</b> visually highlights relative strength using gradient heatmaps<br><br>
                <span style="color: #fbbf24;">→ Upload reports for 2+ companies in Tab 1 to see meaningful comparisons.</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('<div class="t-panel">', unsafe_allow_html=True)
        st.markdown('<div class="t-panel-header">INDUSTRY CROSS-SECTION BENCHMARKING</div>', unsafe_allow_html=True)
        st.info("Select an industry to algorithmically benchmark all ingested companies against their sector averages. The Quantitative Sector Matrix uses color gradients to highlight outperformance (darker green) versus underperformance.")
        
        # Get unique sectors
        sectors = set()
        for v in st.session_state.macro_db.values():
            sect = v.get('sector', 'Unknown') if isinstance(v, dict) else getattr(v, 'sector', 'Unknown')
            sectors.add(sect)
            
        c1, c2 = st.columns([1, 4])
        with c1:
            st.markdown('<div style="font-size: 10px; color: #6b7280; margin-bottom: 8px; margin-top: 12px;">TARGET INDUSTRY</div>', unsafe_allow_html=True)
            target_sector = st.selectbox("Sector", sorted(list(sectors)), label_visibility="collapsed", key="t5_sector")
            
        # Filter DB by sector
        sector_db = {k: v for k, v in st.session_state.macro_db.items() if (v.get('sector', 'Unknown') if isinstance(v, dict) else getattr(v, 'sector', 'Unknown')) == target_sector}
        
        if sector_db:
            # Calculate Averages safely
            def get_s(comp_data, metric_name):
                scores = comp_data.get('competitive_scores', {})
                metric_key = metric_name.lower().replace(' ', '_')
                if hasattr(scores, 'model_dump'):
                    return getattr(scores, metric_key, 0)
                elif isinstance(scores, dict):
                    return scores.get(metric_key, 0)
                return 0
            def get_m(comp_data):
                m = comp_data.get('management_tone_score', 0) if isinstance(comp_data, dict) else getattr(comp_data, 'management_tone_score', 0)
                return m

            avg_inn = sum([get_s(v, 'innovation') for v in sector_db.values()]) / len(sector_db)
            avg_risk = sum([get_s(v, 'risk_profile') for v in sector_db.values()]) / len(sector_db)
            avg_fin = sum([get_s(v, 'financial_health') for v in sector_db.values()]) / len(sector_db)
            avg_mgmt = sum([get_m(v) for v in sector_db.values()]) / len(sector_db)
            
            with c2:
                st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
                html = f'''
                <div class="m-card bull">
                    <div class="m-title">IND: INNO AVG</div>
                    <div class="m-val">{avg_inn:.1f}<span style="font-size:14px;color:#6b7280">/10</span></div>
                </div>
                <div class="m-card">
                    <div class="m-title">IND: FIN HEALTH AVG</div>
                    <div class="m-val">{avg_fin:.1f}<span style="font-size:14px;color:#6b7280">/10</span></div>
                </div>
                <div class="m-card bear">
                    <div class="m-title">IND: RISK AVG</div>
                    <div class="m-val">{avg_risk:.1f}<span style="font-size:14px;color:#6b7280">/10</span></div>
                </div>
                <div class="m-card">
                    <div class="m-title">IND: MGMT TONE AVG</div>
                    <div class="m-val">{avg_mgmt:.1f}<span style="font-size:14px;color:#6b7280">/10</span></div>
                </div>
                '''
                st.markdown(html + '</div>', unsafe_allow_html=True)
                
            st.markdown("<hr style='border-color: #1f2937; margin: 20px 0;'>", unsafe_allow_html=True)
            
            # Build DataFrame
            df_data = []
            for k, v in sector_db.items():
                inn = get_s(v, 'innovation')
                fin = get_s(v, 'financial_health')
                risk = get_s(v, 'risk_profile')
                mgmt = get_m(v)
                
                df_data.append({
                    "Asset Ticker": k,
                    "Innovation": inn,
                    "Financial Health": fin,
                    "Risk Profile": risk,
                    "Mgmt Tone": mgmt
                })
            
            df = pd.DataFrame(df_data).set_index("Asset Ticker")
            
            st.markdown('<div style="font-size: 11px; color: #9ca3af; margin-bottom: 8px;">QUANTITATIVE SECTOR MATRIX</div>', unsafe_allow_html=True)
            
            # Professional Screener Aesthetic using Pandas Styling
            styled_df = df.style.background_gradient(cmap='Greens', subset=['Innovation', 'Financial Health', 'Mgmt Tone'], vmin=1, vmax=10)\
                                .background_gradient(cmap='Reds', subset=['Risk Profile'], vmin=1, vmax=10)\
                                .format(precision=1)
            
            st.dataframe(styled_df, use_container_width=True, height=250)
            
        else:
            st.warning("No companies found in this sector yet.")
        st.markdown('</div>', unsafe_allow_html=True)
