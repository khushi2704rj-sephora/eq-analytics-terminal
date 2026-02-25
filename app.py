import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json, os, re, time
import pandas as pd
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
    <div class="nav-brand">NEXUS // EQ.ANALYTICS.TERMINAL [V2]</div>
    <div class="nav-status">SYSTEM ONLINE // LATENCY: 12ms</div>
</div>
''', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ENVIRONMENT & STATE
# ─────────────────────────────────────────────
TOKEN = os.environ.get('HF_TOKEN', '')
if 'briefs' not in st.session_state: st.session_state.briefs={}
if 'contexts' not in st.session_state: st.session_state.contexts={}
if 'macro_db' not in st.session_state: st.session_state.macro_db={}

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

def c_dcf_sim(vals, tv, pv):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f'Y{i}' for i in range(1,6)], y=vals, name='Discounted FCF', marker_color='#3b82f6'))
    fig.add_trace(go.Bar(x=['Terminal Value'], y=[tv], name='TV (Present)', marker_color='#8b5cf6'))
    fig.update_layout(**BASE_CHART, title=dict(text=f"VALUATION: ${pv:,.0f}M", font=dict(color='#fbbf24', size=16)), height=350, 
                      margin=dict(l=0, r=0, t=40, b=0), yaxis=dict(gridcolor='#1f2937', zerolinecolor='#374151'), xaxis=dict(gridcolor='#1f2937'))
    return fig

def c_macro_scatter(db, tgt_company, x_metric, y_metric):
    # FIXED: The competitive_scores was sometimes a dict (from mock json) and sometimes a Pydantic object (if generated). Handled safely.
    def get_score(comp_data, metric_name):
        scores = comp_data.get('competitive_scores', {})
        metric_key = metric_name.lower().replace(' ', '_')
        if hasattr(scores, 'model_dump'): # It's a populated AnalystBrief
            return getattr(scores, metric_key, random.randint(3,10))
        elif isinstance(scores, dict): # It's from the raw JSON
            return scores.get(metric_key, random.randint(3,10))
        return random.randint(3,10)

    df_data = []
    for k, v in db.items():
        # Handle dict or AnalystBrief conversion safely for Pandas
        sector = v.get('sector', 'Unknown') if isinstance(v, dict) else getattr(v, 'sector', 'Unknown')
        
        df_data.append({
            'Company': k, 
            'Sector': sector, 
            x_metric: get_score(v, x_metric),
            y_metric: get_score(v, y_metric),
            'Target': k == tgt_company
        })
        
    df = pd.DataFrame(df_data)
    
    fig = px.scatter(df, x=x_metric, y=y_metric, color='Sector', size_max=20, hover_name='Company',
                     color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'])
    
    tgt = df[df['Target']]
    if not tgt.empty:
        fig.add_trace(go.Scatter(x=tgt[x_metric], y=tgt[y_metric], mode='markers', 
                                 marker=dict(size=18, color='rgba(0,0,0,0)', line=dict(color='#fbbf24', width=3)),
                                 name=f"&gt;&gt; {tgt_company} &lt;&lt;", showlegend=True))
        
    fig.update_layout(**BASE_CHART, title=dict(text=f"MACRO CROSS-SECTION: {x_metric.upper()} v {y_metric.upper()}", font=dict(size=14, color='#e5e7eb')), 
                      height=500, xaxis=dict(gridcolor='#1f2937'), yaxis=dict(gridcolor='#1f2937'), 
                      legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)))
    return fig

# ─────────────────────────────────────────────
# FRONTEND LAYOUT
# ─────────────────────────────────────────────
tabs = st.tabs(["[1] DATA_INGEST", "[2] EQ_TERMINAL", "[3] MACRO_UNIVERSE", "[4] VALUATION_DCF", "[5] INDUSTRY_BENCHMARK"])

# --- TAB 1: INGESTION ---
with tabs[0]:
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown('<div class="t-panel">', unsafe_allow_html=True)
        st.markdown('<div class="t-panel-header">STEP 1: DOCUMENT BATCH UPLOAD</div>', unsafe_allow_html=True)
        st.info("Upload 1-3 SEC 10-K or Annual Report PDFs for the SAME company. The engine will merge them into a single intelligence profile.")
        uploaded_files = st.file_uploader("DROP PDFs HERE", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="t-panel">', unsafe_allow_html=True)
        st.markdown('<div class="t-panel-header">STEP 2: METADATA CONFIGURATION</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 10px; color: #6b7280; margin-bottom: 4px;">TARGET ASSET TICKER (Required)</div>', unsafe_allow_html=True)
        custom_name = st.text_input("TICKER", placeholder="e.g. AAPL", label_visibility="collapsed")
        
        st.markdown('<div style="font-size: 10px; color: #6b7280; margin: 12px 0 4px 0;">INDUSTRY CATEGORIZATION (For Benchmark Overlays)</div>', unsafe_allow_html=True)
        sector_input = st.selectbox("SECTOR", ["Technology", "Healthcare", "Financials", "Consumer", "Energy", "Industrials", "Real Estate", "Materials"], label_visibility="collapsed")
        
        st.markdown("<hr style='border-color: #1f2937; margin: 20px 0;'>", unsafe_allow_html=True)
        process_btn = st.button("▶ EXECUTE PIPELINE")
        st.markdown('</div>', unsafe_allow_html=True)
        
    if process_btn and uploaded_files and custom_name:
        with st.container():
            st.markdown('<div class="t-panel">', unsafe_allow_html=True)
            st.markdown('<div class="t-panel-header">PIPELINE EXECUTION LOG</div>', unsafe_allow_html=True)
            pb = st.progress(0); st_txt = st.empty()
            def upd(v, m): pb.progress(v); st_txt.markdown(f"<span style='color:#10b981; font-family:monospace;'>`[{v*100:02.0f}%]` {m}</span>", unsafe_allow_html=True)
            
            vs = ingest_pdf(uploaded_files, custom_name, upd)
            st.session_state[f"vs_{custom_name}"] = vs
            
            b, ctx = run_llm(custom_name, sector_input, vs, upd)
            if b:
                st.session_state.briefs[custom_name] = b
                b_dict = b.model_dump()
                st.session_state.macro_db[custom_name] = b_dict
                st.session_state.macro_db[custom_name]['sensitivity'] = {'rates': -0.2, 'inflation': -0.4, 'supply_chain': -0.1}
                upd(1.0, "TERMINAL SYNCED.")
                st.success(f"[{custom_name}] ASSET REGISTERED. SWITCH TO TAB [2].")
                time.sleep(1.5)
                st_txt.empty(); pb.empty()
            else:
                st.error("❌ PIPELINE FAILED: The LLM could not generate a valid profile. You likely uploaded conflicting files for multiple companies, breaking the strict schema requirement.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: TERMINAL ---
with tabs[1]:
    if not st.session_state.briefs:
        st.markdown('<div class="t-panel" style="text-align:center; padding: 100px 0; color: #4b5563;">AWAITING ASSET INGESTION. PROCEED TO TAB [1].</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="t-panel" style="text-align:center; padding: 100px 0; color: #4b5563;">MACRO DATABASE OFFLINE.</div>', unsafe_allow_html=True)
    else:
        mc1, mc2 = st.columns([1, 4])
        
        with mc1:
            st.markdown('<div class="t-panel">', unsafe_allow_html=True)
            st.markdown('<div class="t-panel-header">MACRO SHOCK SIMULATOR</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 10px; color: #6b7280; margin-bottom: 8px;">TARGET ASSET</div>', unsafe_allow_html=True)
            target_t3 = st.selectbox("Highlight", list(st.session_state.macro_db.keys()), key="t3_target", label_visibility="collapsed")
            
            st.markdown('<div style="font-size: 10px; color: #6b7280; margin: 16px 0 8px 0;">INTEREST RATE DELTA (BPS)</div>', unsafe_allow_html=True)
            rates = st.slider("Interest Rate Delta (bps)", -100, 200, 0, step=25, label_visibility="collapsed")
            
            st.markdown('<div style="font-size: 10px; color: #6b7280; margin: 16px 0 8px 0;">INFLATION SPIKE (%)</div>', unsafe_allow_html=True)
            inf = st.slider("Inflation Spike (%)", 0.0, 10.0, 0.0, label_visibility="collapsed")
            
            st.markdown("<hr style='border-color: #1f2937; margin: 20px 0;'>", unsafe_allow_html=True)
            
            impact = 0
            if target_t3 in st.session_state.macro_db:
                v_data = st.session_state.macro_db[target_t3]
                sens = v_data.get('sensitivity', {}) if isinstance(v_data, dict) else getattr(v_data, 'sensitivity', {'rates': 0, 'inflation': 0})
                impact = (rates/100 * sens.get('rates', 0)) + (inf * sens.get('inflation', 0))
            
            color = "#ef4444" if impact < 0 else "#10b981"
            st.markdown('<div class="t-panel-header" style="color: #9ca3af; border: none; padding: 0;">EST. SHOCK IMPACT</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 28px; color: {color}; font-family: \"JetBrains Mono\", monospace; font-weight: bold;'>{impact:.2f}%</div>", unsafe_allow_html=True)
            st.markdown('<div style="font-size: 10px; color: #6b7280;">MARKET CAP VARIANCE</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with mc2:
            st.markdown('<div class="t-panel">', unsafe_allow_html=True)
            st.plotly_chart(c_macro_scatter(st.session_state.macro_db, target_t3, 'Innovation', 'Risk Profile'), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: DCF SIMULATOR ---
with tabs[3]:
    if not st.session_state.briefs:
        st.markdown('<div class="t-panel" style="text-align:center; padding: 100px 0; color: #4b5563;">AWAITING ASSET INGESTION.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="display:flex; justify-content:flex-end; margin-bottom: 10px;">', unsafe_allow_html=True)
        target_dcf = st.selectbox("ACTIVE ASSET", list(st.session_state.briefs.keys()), key="t4_target", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        b = st.session_state.briefs[target_dcf]
        
        dc1, dc2 = st.columns([1, 2])
        with dc1:
            st.markdown('<div class="t-panel">', unsafe_allow_html=True)
            st.markdown('<div class="t-panel-header">DCF VARIABLES</div>', unsafe_allow_html=True)
            
            st.markdown('<div style="font-size: 10px; color: #6b7280; margin-bottom: 4px;">DISCOUNT RATE (WACC)</div>', unsafe_allow_html=True)
            wacc = st.number_input("WACC", min_value=0.01, max_value=0.30, value=b.wacc, step=0.005, format="%.3f", label_visibility="collapsed")
            
            st.markdown('<div style="font-size: 10px; color: #6b7280; margin: 16px 0 4px 0;">TERMINAL GROWTH RATE</div>', unsafe_allow_html=True)
            g = st.number_input("Growth", min_value=-0.05, max_value=0.10, value=b.growth_rate, step=0.005, format="%.3f", label_visibility="collapsed")
            
            st.markdown('<div style="font-size: 10px; color: #6b7280; margin: 16px 0 4px 0;">BASE FCF (MILLIONS)</div>', unsafe_allow_html=True)
            fcf = st.number_input("FCF", value=float(b.fcf_base), step=100.0, label_visibility="collapsed")
            
            st.markdown('<div style="font-size: 10px; color: #6b7280; margin: 16px 0 4px 0;">PROJECTION HORIZON (YRS)</div>', unsafe_allow_html=True)
            years = st.slider("Years", 3, 10, 5, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with dc2:
            st.markdown('<div class="t-panel">', unsafe_allow_html=True)
            st.markdown('<div class="t-panel-header">INTRINSIC VALUE MODELER</div>', unsafe_allow_html=True)
            pv, vals = simulate_dcf(fcf, wacc, g, years)
            tv = pv - sum(vals)
            st.plotly_chart(c_dcf_sim(vals, tv, pv), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 5: INDUSTRY BENCHMARK ---
with tabs[4]:
    if not st.session_state.macro_db:
        st.markdown('<div class="t-panel" style="text-align:center; padding: 100px 0; color: #4b5563;">MACRO DATABASE OFFLINE.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="t-panel">', unsafe_allow_html=True)
        st.markdown('<div class="t-panel-header">INDUSTRY CROSS-SECTION BENCHMARKING</div>', unsafe_allow_html=True)
        
        # Get unique sectors
        sectors = set()
        for v in st.session_state.macro_db.values():
            sect = v.get('sector', 'Unknown') if isinstance(v, dict) else getattr(v, 'sector', 'Unknown')
            sectors.add(sect)
            
        c1, c2 = st.columns([1, 4])
        with c1:
            st.markdown('<div style="font-size: 10px; color: #6b7280; margin-bottom: 8px;">TARGET INDUSTRY</div>', unsafe_allow_html=True)
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
                    "Ticker": k,
                    "Innovation": f"{inn} {'🟢' if inn >= avg_inn else '🔴'}",
                    "Financial Health": f"{fin} {'🟢' if fin >= avg_fin else '🔴'}",
                    "Risk Profile (Lower is Better)": f"{risk} {'🟢' if risk <= avg_risk else '🔴'}",
                    "Mgmt Tone": f"{mgmt} {'🟢' if mgmt >= avg_mgmt else '🔴'}"
                })
            
            df = pd.DataFrame(df_data)
            # Custom styling for dataframe
            html_table = df.to_html(classes='table table-dark', index=False, escape=False)
            st.markdown(f'''
            <style>
                .table-dark {{width: 100%; color: #d1d5db; border-collapse: collapse; font-family: "JetBrains Mono", monospace; font-size: 12px;}}
                .table-dark th {{background: #1f2937; padding: 12px; text-align: left; color: #9ca3af; border-bottom: 2px solid #374151;}}
                .table-dark td {{padding: 12px; border-bottom: 1px solid #1f2937;}}
                .table-dark tr:hover {{background: #1a1a1a;}}
            </style>
            {html_table}
            ''', unsafe_allow_html=True)
            
        else:
            st.warning("No companies found in this sector yet.")
        st.markdown('</div>', unsafe_allow_html=True)
