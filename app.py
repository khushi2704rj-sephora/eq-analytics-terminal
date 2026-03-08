import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json, os, re, time
import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
import requests
import urllib3
from huggingface_hub import InferenceClient
from pydantic import BaseModel, field_validator, Field
from typing import List
import torch
import torch.nn as nn
from fpdf import FPDF
import base64
import random
import textwrap

# Polyfill for st.html in older Streamlit versions
# Strips ALL leading whitespace per line to prevent markdown code-block rendering
def _st_html_polyfill(html_str):
    cleaned = "\n".join(line.lstrip() for line in html_str.split("\n"))
    st.markdown(cleaned, unsafe_allow_html=True)

if not hasattr(st, "html"):
    st.html = _st_html_polyfill

# ─────────────────────────────────────────────
# PAGE CONFIG & ULTRA-PREMIUM STYLING
# ─────────────────────────────────────────────
st.set_page_config(page_title='Nexus Equity Terminal | Capstone', page_icon='⚡', layout='wide', initial_sidebar_state='collapsed')
st.html('''<style>
/* ─────────────────────────────────────────────
 * ULTRA-PREMIUM CONSUMER UI V5 (CRED / SUI INSPIRED)
 * ───────────────────────────────────────────── */

/* Core Typography & Base */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

.stApp {
    background: #030303; 
    color: #e4e4e7; /* zinc-200 */
    font-family: "Inter", -apple-system, sans-serif;
    background-image: radial-gradient(circle at 50% 0%, rgba(59, 130, 246, 0.08) 0%, rgba(3, 3, 3, 1) 40%);
    background-attachment: fixed;
}
header {visibility: hidden;}
.st-emotion-cache-16txtl3 {padding-top: 0rem;}
.st-emotion-cache-1jicfl2 {padding: 1rem 1rem;}

/* Top Nav (Minimalist Premium) */
.terminal-nav {
    display: flex; justify-content: space-between; align-items: center; 
    padding: 16px 32px; margin-bottom: 24px;
    background: rgba(10, 10, 10, 0.4);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    position: sticky; top: 0; z-index: 999;
}
.nav-brand {
    font-family: "Inter", sans-serif; font-size: 15px; font-weight: 800; 
    color: #ffffff; letter-spacing: 1.5px; text-transform: uppercase;
    background: linear-gradient(90deg, #ffffff, #a1a1aa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nav-status {
    font-size: 11px; color: #10b981; font-weight: 600; font-family: "Inter", sans-serif; 
    display: flex; align-items: center; letter-spacing: 1px; text-transform: uppercase;
    background: rgba(16, 185, 129, 0.1); padding: 4px 12px; border-radius: 20px;
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.nav-status::before {
    content: ''; display: inline-block; width: 6px; height: 6px; 
    background-color: #10b981; border-radius: 50%; margin-right: 8px; box-shadow: 0 0 8px #10b981;
}

/* Custom Tabs (Pill Segments) */
div[data-testid="stTabs"] {background: transparent;}
div[data-testid="stTabs"] button {
    background-color: transparent !important; color: #71717a !important; 
    border: none !important; border-radius: 100px !important; padding: 10px 20px !important; 
    font-family: "Inter", sans-serif; font-size: 13px !important; font-weight: 600 !important; 
    transition: all 0.3s ease; margin-right: 8px;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: rgba(255, 255, 255, 0.08) !important; color: #ffffff !important; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
}

/* Glassmorphic Panels */
.t-panel {
    background: rgba(15, 15, 15, 0.6); 
    border: 1px solid rgba(255, 255, 255, 0.05); 
    border-radius: 24px; padding: 28px; 
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    margin-bottom: 24px;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
}
.t-panel:hover {
    border-color: rgba(255, 255, 255, 0.08);
}
.t-panel-header {
    font-size: 11px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 2.5px; 
    margin-bottom: 20px; font-weight: 700; font-family: "Inter", sans-serif;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04); padding-bottom: 12px;
}

/* UI Typography Classes */
@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}
@keyframes fadeUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

.hero-title {
    font-size: clamp(2.5rem, 6vw, 4.5rem); font-weight: 800;
    line-height: 1.1; letter-spacing: -2px; margin-bottom: 24px;
    background: linear-gradient(90deg, #ffffff 0%, #a1a1aa 20%, #10b981 50%, #a1a1aa 80%, #ffffff 100%);
    background-size: 200% auto;
    color: transparent; -webkit-background-clip: text; background-clip: text;
    animation: shimmer 8s linear infinite, fadeUp 1s ease-out forwards;
}
.hero-subtitle {
    font-size: 18px; color: #a1a1aa; line-height: 1.6; font-weight: 400; max-width: 650px; margin-bottom: 40px;
    animation: fadeUp 1s ease-out 0.2s forwards; opacity: 0;
}

/* Peak Interactive Feature Cards */
.feature-card {
    background: rgba(15, 15, 15, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 24px; padding: 32px 24px;
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    animation: fadeUp 1s ease-out 0.4s forwards, float 6s ease-in-out infinite;
    opacity: 0; cursor: pointer;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
.feature-card:nth-child(2) { animation-delay: 0.5s, 0.5s; }
.feature-card:nth-child(3) { animation-delay: 0.6s, 1s; }

.feature-card:hover {
    transform: translateY(-12px) scale(1.02);
    border-color: rgba(16, 185, 129, 0.4);
    box-shadow: 0 20px 40px rgba(16, 185, 129, 0.15), 0 0 20px inset rgba(16, 185, 129, 0.05);
}
.f-icon { font-size: 40px; margin-bottom: 16px; transition: transform 0.3s ease; }
.feature-card:hover .f-icon { transform: scale(1.2) rotate(5deg); }
.f-title { font-size: 14px; font-weight: 700; color: #ffffff; letter-spacing: 2px; margin-bottom: 12px; }
.f-desc { font-size: 13px; color: #a1a1aa; line-height: 1.5; font-weight: 400; }

/* Glass Metric Cards */
.metric-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px;}
.m-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.m-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); border-color: rgba(255,255,255,0.1); }
.m-title {font-size: 11px; color: #71717a; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 8px;}
.m-val {font-size: 32px; font-weight: 800; color: #ffffff; font-family: "Inter", sans-serif; letter-spacing: -1px; line-height: 1;}
.m-sub {font-size: 12px; color: #10b981; display: flex; align-items: center; margin-top: 12px; font-weight: 500;}
.m-sub.warn {color: #fbbf24;}
.m-sub.danger {color: #ef4444;}
.m-card.bull {border-top: 2px solid #10b981;}
.m-card.bear {border-top: 2px solid #ef4444;}

/* Premium Insight Pills & Verdicts */
.verdict-box {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.05)); 
    border: 1px solid rgba(59, 130, 246, 0.2); 
    padding: 24px; border-radius: 16px; margin-bottom: 24px;
    box-shadow: inset 0 0 20px rgba(59, 130, 246, 0.05);
}
.verdict-text {font-size: 18px; color: #ffffff; line-height: 1.6; font-weight: 500; font-family: "Inter", sans-serif;}

.insight-list {list-style: none; padding: 0; margin: 0;}
.insight-item {
    font-size: 14px; color: #d4d4d8; padding: 12px 16px; margin-bottom: 12px; 
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); border-radius: 12px;
    display: flex; align-items: flex-start;
}
.insight-item::before {
    content: '✦'; color: #3b82f6; margin-right: 12px; font-size: 14px; margin-top: 2px;
}

/* Streamlit Widget Overrides (Buttons & Inputs) */
.stButton>button {
    background: #ffffff !important; color: #000000 !important; border: none !important; 
    border-radius: 100px !important; font-weight: 700 !important; font-family: "Inter", sans-serif !important; 
    letter-spacing: 0.5px; width: 100%; padding: 12px 24px !important; transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(255,255,255,0.1); text-transform: none !important; font-size: 14px !important;
}
.stButton>button:hover {background: #f4f4f5 !important; box-shadow: 0 8px 24px rgba(255,255,255,0.2) !important; transform: translateY(-1px);}

div[data-baseweb="input"] {
    background: rgba(15, 15, 15, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2), 0 0 20px rgba(16, 185, 129, 0.1) !important;
}
div[data-baseweb="input"] input {
    color: #10b981 !important;
    font-size: 20px !important;
    font-family: "JetBrains Mono", monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 14px !important;
}
div[data-baseweb="input"] input::placeholder {
    color: rgba(16, 185, 129, 0.3) !important;
    font-weight: 400 !important;
}

.stTextInput>div>div>input, .stSelectbox>div>div>div {
    background: rgba(0,0,0,0.5) !important; color: #ffffff !important; 
    border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important;
    padding: 12px 16px !important; font-size: 15px !important; font-family: "Inter", sans-serif !important;
    min-height: 48px !important;
}
.stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
    border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}

/* Selectbox & Dropdown Full Override */
div[data-baseweb="select"] { background: rgba(0,0,0,0.5) !important; border-radius: 12px !important; min-height: 48px !important;}
div[data-baseweb="select"] > div { background: transparent !important; color: #ffffff !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important; min-height: 48px !important;}
div[data-baseweb="select"] span { color: #ffffff !important; padding: 0 !important; font-size: 15px !important; }
div[data-baseweb="select"] svg { fill: #ffffff !important; }
ul[role="listbox"] { background: #0a0a0a !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important; }
li[role="option"] { color: #ffffff !important; background: transparent !important; padding: 12px 16px !important; }
li[role="option"]:hover { background: rgba(255,255,255,0.08) !important; }
li[role="option"][aria-selected="true"] { background: rgba(16,185,129,0.15) !important; }
</style>''')

st.html('''
<div class="terminal-nav">
    <div class="nav-brand">NEXUS // EQ.ANALYTICS.TERMINAL [V4]</div>
    <div class="nav-status">SYSTEM ONLINE</div>
</div>
'''  )

# ─────────────────────────────────────────────
# ENVIRONMENT & STATE
# ─────────────────────────────────────────────
TOKEN = os.environ.get('HF_TOKEN', '')
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nexus_v2.db')

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
    management_tone_score: float = Field(description="Score from 1 to 10 evaluating tone (1 = highly pessimistic, 10 = highly confident).")
    bull_case: str = Field(description="2-3 sentence optimistic scenario.")
    bear_case: str
    analyst_verdict: str
    wacc: float = Field(description="Weighted Average Cost of Capital (WACC) as a decimal (e.g. 0.08). MUST be unique to this company based on risk profile.")
    growth_rate: float = Field(description="Terminal growth rate as a decimal (e.g. 0.025). MUST be unique based on company maturity.")
    fcf_base: float = Field(description="Base Free Cash Flow in MILLIONS USD. MUST extract or estimate exactly for this specific company. Do NOT default to 5000.")

# ─────────────────────────────────────────────
# BACKEND: INGESTION & LLM
# ─────────────────────────────────────────────

import urllib.request
import ssl

_YF_SSL_CTX = ssl.create_default_context()
_YF_SSL_CTX.check_hostname = False
_YF_SSL_CTX.verify_mode = ssl.CERT_NONE
_YF_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

def get_yf_session():
    """Legacy helper — still used as fallback."""
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    session = requests.Session()
    session.verify = False
    session.headers.update({"User-Agent": _YF_UA})
    return session

@st.cache_data(ttl=3600, show_spinner=False)
def _get_yahoo_crumb():
    """Fetch a valid Yahoo Finance cookie + crumb pair for API auth."""
    try:
        req1 = urllib.request.Request("https://fc.yahoo.com", headers={'User-Agent': _YF_UA})
        cookie = ""
        try:
            with urllib.request.urlopen(req1, context=_YF_SSL_CTX, timeout=5): pass
        except urllib.error.HTTPError as e:
            sc = e.headers.get('Set-Cookie', '')
            if sc: cookie = sc.split(';')[0]
        except Exception:
            pass
        if not cookie:
            return None, None
        req2 = urllib.request.Request(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers={'User-Agent': _YF_UA, 'Cookie': cookie}
        )
        with urllib.request.urlopen(req2, context=_YF_SSL_CTX, timeout=5) as r:
            return cookie, r.read().decode()
    except Exception:
        return None, None

def yahoo_api_request(url):
    """Make an authenticated request to any Yahoo Finance API endpoint."""
    req = urllib.request.Request(url, headers={'User-Agent': _YF_UA, 'Accept': '*/*'})
    cookie, crumb = _get_yahoo_crumb()
    if cookie:
        req.add_header('Cookie', cookie)
    with urllib.request.urlopen(req, context=_YF_SSL_CTX, timeout=10) as resp:
        return json.loads(resp.read().decode())

def yahoo_quote(ticker):
    """Fetch full quote data for a ticker via v7/finance/quote with crumb auth."""
    cookie, crumb = _get_yahoo_crumb()
    if not cookie or not crumb:
        return {}
    url = f"https://query2.finance.yahoo.com/v7/finance/quote?symbols={ticker}&crumb={crumb}"
    req = urllib.request.Request(url, headers={'User-Agent': _YF_UA, 'Cookie': cookie})
    with urllib.request.urlopen(req, context=_YF_SSL_CTX, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    results = data.get('quoteResponse', {}).get('result', [])
    return results[0] if results else {}

def yahoo_chart(ticker, range_str='1y', interval='1d'):
    """Fetch historical chart data for a ticker."""
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={range_str}"
    req = urllib.request.Request(url, headers={'User-Agent': _YF_UA, 'Accept': '*/*'})
    with urllib.request.urlopen(req, context=_YF_SSL_CTX, timeout=10) as resp:
        return json.loads(resp.read().decode())

def yahoo_modules(ticker, modules="defaultKeyStatistics"):
    """Fetch specialized module data (like Beta, EV/EBITDA) from v10 API."""
    cookie, crumb = _get_yahoo_crumb()
    if not cookie or not crumb:
        return {}
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules={modules}&crumb={crumb}"
    req = urllib.request.Request(url, headers={'User-Agent': _YF_UA, 'Cookie': cookie})
    try:
        with urllib.request.urlopen(req, context=_YF_SSL_CTX, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        result = data.get('quoteSummary', {}).get('result', [])
        return result[0] if result else {}
    except Exception:
        return {}

def yahoo_search(query):
    """Search for a ticker by company name."""
    import urllib.parse
    encoded = urllib.parse.quote(query)
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={encoded}&quotesCount=5&newsCount=0"
    req = urllib.request.Request(url, headers={'User-Agent': _YF_UA, 'Accept': '*/*'})
    with urllib.request.urlopen(req, context=_YF_SSL_CTX, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    return data.get('quotes', [])

@st.cache_data(ttl=86400, show_spinner=False)
def resolve_ticker(query):
    """Resolve a company name or ticker to a valid ticker symbol."""
    query = query.strip()
    if not query:
        return None, None
    
    # First, try as a direct ticker via quote API
    try:
        info = yahoo_quote(query.upper())
        if info and (info.get('regularMarketPrice') or info.get('marketCap')):
            return query.upper(), info.get('longName') or info.get('shortName') or query.upper()
    except Exception:
        pass
    
    # Search by name
    try:
        results = yahoo_search(query)
        if results:
            for q in results:
                symbol = q.get('symbol', '')
                name = q.get('longname') or q.get('shortname') or symbol
                exchange = q.get('exchange', '')
                if exchange in ('NMS', 'NYQ', 'NGM', 'PCX', 'BTS', 'NAS'):
                    return symbol, name
            first = results[0]
            return first.get('symbol', query.upper()), first.get('longname') or first.get('shortname') or query.upper()
    except Exception:
        pass
    
    return None, None

@st.cache_data(ttl=900, show_spinner=False)
def fetch_company_context(ticker):
    """Pull comprehensive financial data from Yahoo Finance APIs and format as context string."""
    try:
        # Primary: Use our crumb-authenticated v7 quote API
        info = yahoo_quote(ticker)
        
        if not info or not (info.get('regularMarketPrice') or info.get('marketCap')):
            return None, None, None
        
        # Basic info
        name = info.get('longName') or info.get('shortName') or ticker
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        summary = info.get('longBusinessSummary', 'Financial data retrieved from market feeds.')[:800]
        employees = info.get('fullTimeEmployees', 'N/A')
        country = info.get('country', 'N/A') if 'country' in info else 'N/A'
        
        ctx = f"COMPANY: {name} ({ticker})\n"
        ctx += f"SECTOR: {sector} | INDUSTRY: {industry}\n"
        ctx += f"COUNTRY: {country} | EMPLOYEES: {employees}\n\n"
        ctx += f"BUSINESS DESCRIPTION:\n{summary}\n\n"
        
        # Key metrics from quote data
        ctx += "KEY METRICS:\n"
        metrics = {
            'Market Cap': info.get('marketCap'),
            'Revenue (TTM)': info.get('revenue'),
            'P/E Ratio': info.get('trailingPE'),
            'Forward P/E': info.get('forwardPE'),
            'EPS (TTM)': info.get('epsTrailingTwelveMonths'),
            'Price/Book': info.get('priceToBook'),
            'Profit Margin': info.get('profitMargins'),
            'Dividend Yield': info.get('dividendYield'),
            'Beta': info.get('beta'),
            '52-Week High': info.get('fiftyTwoWeekHigh'),
            '52-Week Low': info.get('fiftyTwoWeekLow'),
            'Avg Volume': info.get('averageDailyVolume3Month'),
            '50-Day MA': info.get('fiftyDayAverage'),
            '200-Day MA': info.get('twoHundredDayAverage'),
            'Shares Outstanding': info.get('sharesOutstanding'),
            'Book Value': info.get('bookValue'),
        }
        for k, v in metrics.items():
            if v is not None:
                if isinstance(v, float) and 0 < abs(v) < 1:
                    ctx += f"  {k}: {v:.2%}\n"
                elif isinstance(v, (int, float)) and abs(v) > 1e6:
                    ctx += f"  {k}: ${v/1e9:.2f}B\n" if abs(v) >= 1e9 else f"  {k}: ${v/1e6:.1f}M\n"
                else:
                    ctx += f"  {k}: {v}\n"
        
        # Analyst targets from quote data
        targets = {
            'Target Mean': info.get('targetMeanPrice'),
            'Target High': info.get('targetHighPrice'),
            'Target Low': info.get('targetLowPrice'),
            'Recommendation': info.get('recommendationKey'),
            'Number of Analysts': info.get('numberOfAnalystOpinions'),
        }
        has_targets = any(v is not None for v in targets.values())
        if has_targets:
            ctx += "\nANALYST CONSENSUS:\n"
            for k, v in targets.items():
                if v is not None:
                    ctx += f"  {k}: {v}\n"
        
        # Supplement with historical price performance from chart API
        try:
            chart = yahoo_chart(ticker, range_str='1y')
            result = chart.get('chart', {}).get('result', [])
            if result:
                closes_raw = result[0].get('indicators', {}).get('quote', [{}])[0].get('close', [])
                valid = [c for c in closes_raw if c is not None]
                if len(valid) > 20:
                    ctx += f"\nPRICE PERFORMANCE:\n"
                    ctx += f"  Current: ${valid[-1]:.2f}\n"
                    ctx += f"  1-Year Ago: ${valid[0]:.2f}\n"
                    ctx += f"  1-Year Return: {((valid[-1] - valid[0]) / valid[0] * 100):.1f}%\n"
                    ctx += f"  1-Year High: ${max(valid):.2f}\n"
                    ctx += f"  1-Year Low: ${min(valid):.2f}\n"
        except Exception:
            pass
        
        return name, sector, ctx
    except Exception as e:
        return None, None, None

def prompt_live(cn, sector, ctx):
    """Build LLM prompt using live financial data context."""
    p = 'You are a Senior Equity Research Analyst for a top-tier investment bank.\n'
    p += f'Analyze {cn} using the live financial data provided. Return ONLY valid JSON matching this schema exactly.\n'
    p += '{\n'
    p += f'  "company_name":"{cn}","fiscal_year":"TTM","sector":"{sector}",\n'
    p += '  "business_model":"Detailed 3 sentence breakdown of how this company generates revenue",\n'
    p += '  "key_segments":["segment 1","segment 2","segment 3"],\n'
    p += '  "financial_highlights":["Specific highlight with numbers","Another quantified insight","Third data point"],\n'
    p += '  "top_risks":[{"risk":"Name","description":"1-2 sentence detail","severity":"High","likelihood":4,"impact":5}],\n'
    p += '  "strategic_priorities":[{"priority":"Name","detail":"Detail","time_horizon":"Short-term"}],\n'
    p += '  "competitive_scores":{"innovation":8,"market_position":9,"financial_health":8,"risk_profile":5},\n'
    p += '  "management_tone":"Optimistic","management_tone_score":7,\n'
    p += '  "bull_case":"2 sentences with specific catalysts","bear_case":"2 sentences with specific risks",\n'
    p += '  "analyst_verdict":"1 sentence decisive verdict with price target reasoning",\n'
    p += '  "wacc": 0.08, "growth_rate": 0.03, "fcf_base": 5000\n'
    p += '}\n'
    p += 'IMPORTANT: fcf_base must be the ACTUAL Free Cash Flow in MILLIONS USD from the data. Do NOT default to 5000.\n'
    p += 'IMPORTANT: wacc and growth_rate must reflect THIS specific company, not generic defaults.\n'
    p += 'IMPORTANT: All scores (1-10) and financial_highlights must use REAL numbers from the data below.\n'
    p += f'=== LIVE FINANCIAL DATA ===\n{ctx[:5000]}\n'
    return p

def run_llm_live(ticker, prog):
    """End-to-end pipeline: yfinance → LLM → AnalystBrief. No PDF needed."""
    prog(0.1, f'Fetching live financial data for {ticker}...')
    result = fetch_company_context(ticker)
    if result is None or result[0] is None:
        st.error(f"Could not fetch data for ticker '{ticker}'. Please check the ticker symbol.")
        return None, None
    
    name, sector, ctx = result
    prog(0.3, f'Retrieved {len(ctx)} chars of financial context for {name}...')
    prog(0.5, 'Synthesizing with Llama-3.1-8B...')
    
    cl = InferenceClient(provider='novita', api_key=TOKEN)
    last_err = ""
    for attempt in range(3):
        try:
            r = cl.chat.completions.create(
                model='meta-llama/Llama-3.1-8B-Instruct',
                messages=[{'role': 'user', 'content': prompt_live(name, sector, ctx)}],
                max_tokens=2500, temperature=0.1
            )
            raw = r.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'^```\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                raise ValueError("No valid JSON block detected from LLM.")
            b = AnalystBrief(**json.loads(m.group()))
            prog(0.9, 'Validated Schema ✓')
            return b, sector
        except Exception as e:
            last_err = str(e)
            prog(0.5, f'Retry {attempt+1}/3 ({last_err[:40]}...)')
            time.sleep(2)
    
    st.error(f"LLM Generation Failed. Last Error: {last_err}")
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

def build_sensitivity_table(fcf, wacc_base, g_base, years=5):
    wacc_steps = [round(wacc_base + delta, 4) for delta in [-0.02, -0.01, 0.0, 0.01, 0.02]]
    g_steps = [round(g_base + delta, 4) for delta in [-0.01, -0.005, 0.0, 0.005, 0.01]]
    rows = []
    for w in wacc_steps:
        row = {}
        for g in g_steps:
            if w > g and w > 0:
                val, _ = simulate_dcf(fcf, w, g, years)
                row[f"g={g:.1%}"] = val
            else:
                row[f"g={g:.1%}"] = None
        rows.append(row)
    df = pd.DataFrame(rows, index=[f"WACC={w:.1%}" for w in wacc_steps])
    return df

def fetch_live_data(ticker):
    """Fetch live market data for the dashboard metrics panel."""
    try:
        info = yahoo_quote(ticker)
        if not info:
            return None
            
        stats = yahoo_modules(ticker, "defaultKeyStatistics")
        ks = stats.get('defaultKeyStatistics', {})
        beta = ks.get('beta', {}).get('raw', 'N/A')
        ev_ebitda = ks.get('enterpriseToEbitda', {}).get('raw', 'N/A')
            
        return {
            'price': info.get('regularMarketPrice', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'ev_ebitda': ev_ebitda,
            'week52_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'week52_low': info.get('fiftyTwoWeekLow', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'beta': beta,
        }
    except Exception:
        return None

# ─────────────────────────────────────────────
# GRU DEEP LEARNING SIGNAL ENGINE
# ─────────────────────────────────────────────
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

@st.cache_data(ttl=900, show_spinner=False)
def run_gru_prediction(ticker, lookback=60, forecast_days=10, epochs=50):
    """Train a GRU on 1 year of daily closes and forecast forward."""
    try:
        import datetime
        
        # Use centralized chart API
        data = yahoo_chart(ticker, range_str='1y', interval='1d')
        result = data.get('chart', {}).get('result', [])
        if not result:
            raise ValueError(f"No chart data for {ticker}")
        
        timestamps = result[0].get('timestamp', [])
        closes_raw = result[0].get('indicators', {}).get('quote', [{}])[0].get('close', [])
        
        valid_data = [(ts, c) for ts, c in zip(timestamps, closes_raw) if c is not None]
        if len(valid_data) < lookback + 20:
            raise ValueError("Insufficient trading days")
        
        timestamps, closes_raw = zip(*valid_data)
        dates = [datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).date() for ts in timestamps]
        df = pd.DataFrame({'Close': closes_raw}, index=pd.DatetimeIndex(dates))
        
        closes = df['Close'].values.astype(float)
        
        # Normalize
        min_p, max_p = closes.min(), closes.max()
        price_range = max_p - min_p
        if price_range == 0:
            return None
        scaled = (closes - min_p) / price_range
        
        # Build sliding-window sequences
        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i - lookback:i])
            y.append(scaled[i])
        
        X = torch.FloatTensor(np.array(X)).unsqueeze(-1)  # (N, lookback, 1)
        y = torch.FloatTensor(np.array(y)).unsqueeze(-1)  # (N, 1)
        
        # Train / val split (80/20)
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        
        # Build and train model
        model = GRUModel(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            output = model(X_train)
            loss = criterion(output, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Forecast forward
        model.eval()
        with torch.no_grad():
            last_seq = torch.FloatTensor(scaled[-lookback:]).unsqueeze(0).unsqueeze(-1)
            predictions_scaled = []
            current_seq = last_seq.clone()
            
            for _ in range(forecast_days):
                pred = model(current_seq)
                predictions_scaled.append(pred.item())
                # Slide window: drop first, append prediction
                new_val = pred.unsqueeze(0)  # (1, 1, 1)
                current_seq = torch.cat([current_seq[:, 1:, :], new_val], dim=1)
        
        # Denormalize
        predicted_prices = [p * price_range + min_p for p in predictions_scaled]
        last_actual = float(closes[-1])
        pred_final = predicted_prices[-1]
        
        # Determine signal
        pct_change = ((pred_final - last_actual) / last_actual) * 100
        if pct_change > 1.5:
            signal = 'BULLISH'
        elif pct_change < -1.5:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        # Confidence (inverse of normalized loss variance, capped 60-95%)
        val_pred = model(X[split:])
        val_loss = criterion(val_pred, y[split:]).item()
        confidence = max(60, min(95, int((1 - val_loss * 10) * 100)))
        
        # Build date index for forecast
        last_date = df.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        return {
            'hist_dates': df.index[-60:].tolist(),
            'hist_prices': closes[-60:].tolist(),
            'forecast_dates': forecast_dates.tolist(),
            'forecast_prices': predicted_prices,
            'signal': signal,
            'confidence': confidence,
            'pct_change': round(pct_change, 2),
            'last_price': last_actual,
            'pred_price': round(pred_final, 2),
        }
    except Exception as e:
        import traceback
        error_msg = f"GRU Failed: {type(e).__name__} - {str(e)}\n{traceback.format_exc()}"
        st.session_state["gru_last_error"] = error_msg
        return None

def c_gru_forecast_chart(gru_data):
    """Build a Plotly chart showing historical prices + GRU forecast."""
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=gru_data['hist_dates'], y=gru_data['hist_prices'],
        mode='lines', name='Actual Price',
        line=dict(color='#3b82f6', width=2)
    ))
    
    # Forecast line
    signal_color = '#10b981' if gru_data['signal'] == 'BULLISH' else '#ef4444' if gru_data['signal'] == 'BEARISH' else '#fbbf24'
    
    # Connect forecast to last actual price
    connect_dates = [gru_data['hist_dates'][-1]] + gru_data['forecast_dates']
    connect_prices = [gru_data['hist_prices'][-1]] + gru_data['forecast_prices']
    
    fig.add_trace(go.Scatter(
        x=connect_dates, y=connect_prices,
        mode='lines+markers', name=f'GRU Forecast ({gru_data["signal"]})',
        line=dict(color=signal_color, width=2, dash='dot'),
        marker=dict(size=4, color=signal_color)
    ))
    
    # Confidence band (±2%)
    upper = [p * 1.02 for p in gru_data['forecast_prices']]
    lower = [p * 0.98 for p in gru_data['forecast_prices']]
    
    fig.add_trace(go.Scatter(
        x=gru_data['forecast_dates'] + gru_data['forecast_dates'][::-1],
        y=upper + lower[::-1],
        fill='toself', fillcolor=f'rgba({",".join(str(int(signal_color.lstrip("#")[i:i+2], 16)) for i in (0, 2, 4))}, 0.1)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False, name='Confidence Band'
    ))
    
    # Vertical divider line
    fig.add_vline(x=gru_data['hist_dates'][-1], line_dash="dash", line_color="#374151", line_width=1)
    fig.add_annotation(x=gru_data['hist_dates'][-1], y=max(gru_data['hist_prices']),
                       text="TODAY", showarrow=False, font=dict(size=10, color='#6b7280'), yshift=15)
    
    fig.update_layout(
        **BASE_CHART,
        title=dict(text="GRU NEURAL NETWORK PRICE FORECAST", font=dict(color='#e5e7eb', size=14)),
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(gridcolor='#1f2937', title=dict(text='Price ($)', font=dict(size=11))),
        xaxis=dict(gridcolor='#1f2937'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10), orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x'
    )
    return fig

def _sanitize_text(text):
    """Replace Unicode chars that fpdf/latin-1 can't encode."""
    replacements = {
        '\u2014': '--', '\u2013': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2022': '*', '\u2026': '...',
        '\u00a0': ' ', '\u2019': "'", '\u00e9': 'e', '\u00e8': 'e',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Fallback: strip any remaining non-latin-1 chars
    return text.encode('latin-1', errors='replace').decode('latin-1')

def generate_report_pdf(brief, ticker, pv=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 12, _sanitize_text(f'NEXUS EQUITY TERMINAL - {ticker}'), ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 8, _sanitize_text(f'Sector: {brief.sector} | Fiscal Year: {brief.fiscal_year}'), ln=True)
    pdf.ln(4)
    
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 8, 'ANALYST VERDICT', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 6, _sanitize_text(brief.analyst_verdict))
    pdf.ln(3)
    
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 8, 'BUSINESS MODEL', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 6, _sanitize_text(brief.business_model))
    pdf.ln(3)
    
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 8, 'COMPETITIVE SCORES', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f'Innovation: {brief.competitive_scores.innovation}/10  |  Market Position: {brief.competitive_scores.market_position}/10  |  Financial Health: {brief.competitive_scores.financial_health}/10  |  Risk: {brief.competitive_scores.risk_profile}/10', ln=True)
    pdf.ln(3)
    
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 8, 'BULL CASE', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 6, _sanitize_text(brief.bull_case))
    pdf.ln(2)
    
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 8, 'BEAR CASE', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 6, _sanitize_text(brief.bear_case))
    pdf.ln(3)
    
    if pv:
        pdf.set_font('Helvetica', 'B', 13)
        pdf.cell(0, 8, 'DCF VALUATION', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, f'WACC: {brief.wacc:.1%}  |  Growth: {brief.growth_rate:.1%}  |  Base FCF: ${brief.fcf_base:,.0f}M', ln=True)
        pdf.cell(0, 6, f'Estimated Intrinsic Value: ${pv:,.0f}M', ln=True)
    
    pdf.ln(6)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.multi_cell(0, 5, 'Disclaimer: This report was generated by an AI system using RAG + Llama-3.1. Outputs are probabilistic and must be independently verified. Not investment advice.')
    
    return pdf.output(dest='S').encode('latin-1')

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
tabs = st.tabs(["🔍 Search & Analyze", "📊 AI Analysis", "🌍 Market Comparison", "💰 Valuation Model", "🏆 Industry Ranking"])

# --- TAB 1: SEARCH & ANALYZE ---
with tabs[0]:
    st.html('''
<style>
@keyframes orbFloat1 { 0%,100% { transform: translate(0,0) scale(1); } 33% { transform: translate(80px,-60px) scale(1.1); } 66% { transform: translate(-40px,40px) scale(0.9); } }
@keyframes orbFloat2 { 0%,100% { transform: translate(0,0) scale(1); } 33% { transform: translate(-100px,50px) scale(1.2); } 66% { transform: translate(60px,-80px) scale(0.85); } }
@keyframes orbFloat3 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(50px,70px) scale(1.15); } }
@keyframes gridPulse { 0%,100% { opacity: 0.03; } 50% { opacity: 0.08; } }
@keyframes particleDrift { 0% { transform: translateY(0) translateX(0); opacity: 0; } 20% { opacity: 1; } 80% { opacity: 1; } 100% { transform: translateY(-300px) translateX(50px); opacity: 0; } }
@keyframes heroPulse { 0%,100% { opacity: 0.6; } 50% { opacity: 1; } }
</style>
<div style="position: relative; border-radius: 32px; overflow: hidden; margin-bottom: 40px; min-height: 520px; background: #030303;">
<!-- Animated Gradient Orbs -->
<div style="position:absolute; width:400px; height:400px; border-radius:50%; background: radial-gradient(circle, rgba(16,185,129,0.15) 0%, transparent 70%); top:10%; left:15%; animation: orbFloat1 12s ease-in-out infinite; filter: blur(60px);"></div>
<div style="position:absolute; width:350px; height:350px; border-radius:50%; background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%); top:30%; right:10%; animation: orbFloat2 15s ease-in-out infinite; filter: blur(80px);"></div>
<div style="position:absolute; width:300px; height:300px; border-radius:50%; background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 70%); bottom:10%; left:40%; animation: orbFloat3 18s ease-in-out infinite; filter: blur(70px);"></div>
<!-- Animated Grid -->
<div style="position:absolute; inset:0; background-image: linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px); background-size: 60px 60px; animation: gridPulse 8s ease-in-out infinite;"></div>
<!-- Floating Particles -->
<div style="position:absolute; width:3px; height:3px; border-radius:50%; background:#10b981; top:70%; left:20%; animation: particleDrift 6s linear infinite; box-shadow: 0 0 6px #10b981;"></div>
<div style="position:absolute; width:2px; height:2px; border-radius:50%; background:#3b82f6; top:80%; left:50%; animation: particleDrift 8s linear 2s infinite; box-shadow: 0 0 6px #3b82f6;"></div>
<div style="position:absolute; width:3px; height:3px; border-radius:50%; background:#8b5cf6; top:75%; left:75%; animation: particleDrift 7s linear 4s infinite; box-shadow: 0 0 6px #8b5cf6;"></div>
<div style="position:absolute; width:2px; height:2px; border-radius:50%; background:#10b981; top:85%; left:35%; animation: particleDrift 9s linear 1s infinite; box-shadow: 0 0 4px #10b981;"></div>
<div style="position:absolute; width:2px; height:2px; border-radius:50%; background:#fbbf24; top:90%; left:60%; animation: particleDrift 7.5s linear 3s infinite; box-shadow: 0 0 4px #fbbf24;"></div>
<!-- Dark fade overlay -->
<div style="position:absolute; inset:0; background: linear-gradient(180deg, transparent 0%, rgba(3,3,3,0.4) 70%, rgba(3,3,3,0.95) 100%);"></div>
<!-- Content -->
<div style="position: relative; z-index: 2; padding: 80px 20px 60px 20px; text-align: center;">
<div style="font-size: 11px; color: #10b981; font-weight: 700; letter-spacing: 4px; text-transform: uppercase; margin-bottom: 24px; animation: heroPulse 3s ease-in-out infinite;">● LIVE SYSTEM</div>
<div class="hero-title">Nexus AI Terminal</div>
<div class="hero-subtitle" style="margin: 0 auto 48px auto;">
Professional-grade equity research powered by live market data and Llama-3.1-8B.<br>
Enter any public ticker to generate an instant, comprehensive institutional analysis.
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 24px; max-width: 1000px; margin: 0 auto;">
<div class="feature-card">
<div class="f-icon">🔍</div>
<div class="f-title">SEARCH</div>
<div class="f-desc">Enter any major ticker (AAPL, NVDA, GOOGL)</div>
</div>
<div class="feature-card">
<div class="f-icon">⚡</div>
<div class="f-title">ANALYZE</div>
<div class="f-desc">Live financial data is instantly processed via LLM</div>
</div>
<div class="feature-card">
<div class="f-icon">📈</div>
<div class="f-title">FORECAST</div>
<div class="f-desc">Generate DCF valuations and GRU price signals</div>
</div>
</div>
</div>
</div>
    ''')

    # DEMO MODE — One-Click Preloaded Data
    demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_profiles.json')
    if os.path.exists(demo_path):
        st.html('''
        <div style="max-width: 600px; margin: 0 auto 40px auto; background: rgba(59,130,246,0.05); border: 1px solid rgba(59,130,246,0.2); border-radius: 100px; padding: 16px 24px; display: flex; align-items: center; justify-content: space-between;">
            <div style="font-size: 14px; color: #e5e7eb; font-weight: 500;">
                <span style="color:#3b82f6; margin-right:8px;">🎯</span> 
                <b>Want robust data instantly?</b> Load the pre-analyzed demo dataset.
            </div>
        ''')
        
        col_z, col_b, col_z2 = st.columns([1, 2, 1])
        with col_b:
            if st.button("Load Demo (50 Companies · All Sectors)", use_container_width=True):
                with open(demo_path, 'r') as f:
                    demo_data = json.load(f)
                for ticker, data in demo_data.items():
                    brief = AnalystBrief(**{k: v for k, v in data.items() if k != 'sensitivity'})
                    st.session_state.briefs[ticker] = brief
                    st.session_state.macro_db[ticker] = data
                st.success("✅ Demo loaded! View Tabs 2–5.")
                time.sleep(1)
                st.rerun()
        st.html('</div>')

    # ── LIVE SEARCH BAR ──
    st.html('''
    <div style="max-width: 800px; margin: 0 auto;">
        <div class="t-panel" style="padding: 32px; border-radius: 32px; background: rgba(16, 185, 129, 0.03); border-color: rgba(16, 185, 129, 0.1);">
            <div style="font-size: 13px; font-weight: 700; color: #10b981; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px; text-align: center;">
                Initiate Live Analysis
            </div>
    ''')
    
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_ticker = st.text_input(
            "TICKER", 
            placeholder="Search by name or ticker (e.g. Apple, NVDA, Tesla, JPM)...",
            label_visibility="collapsed",
            key="search_ticker_input"
        )
    with search_col2:
        analyze_btn = st.button("Analyze", use_container_width=True, type="primary", key="analyze_live_btn")
    
    st.html('''
        </div>
        <div style="text-align: center; font-size: 12px; color: #71717a; margin-top: 16px;">
            Search by company name or ticker symbol · Supports all major global exchanges
        </div>
    </div>
    ''')
    
    # Process live search
    if analyze_btn and search_ticker:
        # Resolve company name to ticker
        with st.spinner("🔍 Resolving ticker..."):
            resolved_ticker, resolved_name = resolve_ticker(search_ticker)
        
        if not resolved_ticker:
            st.error("❌ Could not resolve that company. Please try a different name or ticker.")
        else:
            ticker_clean = resolved_ticker.upper().strip()
            if resolved_name and resolved_name.upper() != ticker_clean:
                st.html(f'''
<div style="max-width:600px; margin:0 auto 20px auto; background: rgba(16,185,129,0.05); border:1px solid rgba(16,185,129,0.15); border-radius:100px; padding:12px 24px; text-align:center;">
<span style="font-size:13px; color:#a1a1aa;">Resolved:</span>
<span style="font-size:14px; color:#10b981; font-weight:700; margin-left:8px;">{resolved_name}</span>
<span style="font-size:13px; color:#71717a; margin-left:4px;">({ticker_clean})</span>
</div>
                ''')
        
            with st.container():
                st.html(f'''
                <div class="t-panel">
                    <div class="t-panel-header">PIPELINE EXECUTION LOG — {ticker_clean}</div>
                ''')
                
                pb = st.progress(0)
                st_txt = st.empty()
                
                def upd_live(v, m):
                    pb.progress(min(v, 1.0))
                    st_txt.markdown(f"<span style='color:#10b981; font-family:monospace;'>[{v*100:02.0f}%] <b>[{ticker_clean}]</b> {m}</span>", unsafe_allow_html=True)
                
                result = run_llm_live(ticker_clean, upd_live)
                
                if result and result[0] is not None:
                    b, sector = result
                    st.session_state.briefs[ticker_clean] = b
                    b_dict = b.model_dump()
                    st.session_state.macro_db[ticker_clean] = b_dict
                    st.session_state.macro_db[ticker_clean]['sensitivity'] = {'rates': -0.2, 'inflation': -0.4, 'supply_chain': -0.1}
                    persist_to_db(ticker_clean, st.session_state.macro_db[ticker_clean])
                    
                    upd_live(1.0, "PROFILE GENERATED & PERSISTED ✓")
                    st.success(f"✅ **{b.company_name}** analyzed successfully! Navigate to **Tab 2 (AI Analysis)** to view the full report.")
                    time.sleep(2)
                    pb.empty()
                    st_txt.empty()
                
                st.html('</div>')
    
    # Show currently loaded companies
    if st.session_state.briefs:
        st.html('''
        <div class="t-panel" style="border-left: 3px solid #fbbf24;">
            <div class="t-panel-header" style="color: #fbbf24;">LOADED COMPANIES</div>
        ''')
        
        loaded_tickers = list(st.session_state.briefs.keys())
        cols = st.columns(min(len(loaded_tickers), 6))
        for i, tk in enumerate(loaded_tickers):
            with cols[i % 6]:
                b = st.session_state.briefs[tk]
                st.html(f'''
                <div style="background:#0a0a0a; border:1px solid #1f2937; border-radius:6px; padding:10px; text-align:center; margin-bottom:8px;">
                    <div style="font-size:16px; font-weight:bold; color:#e5e7eb; font-family:'JetBrains Mono',monospace;">{tk}</div>
                    <div style="font-size:11px; color:#6b7280;">{b.sector}</div>
                </div>
                ''')
        
        st.html('</div>')
    


    if st.button("🗑️ Clear Database & Reset Workflow"):
        with sqlite3.connect(DB_PATH) as conn:
            conn.cursor().execute("DELETE FROM companies")
            conn.commit()
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

# --- TAB 2: TERMINAL ---
with tabs[1]:
    if not st.session_state.briefs:
        st.html('''
        <div style="text-align:center; padding: 100px 40px; margin-top: 40px; border-radius: 32px; background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.1);">
            <div style="font-size: 48px; margin-bottom: 24px; opacity: 0.8;">📊</div>
            <div style="font-size: 20px; color: #ffffff; font-weight: 700; margin-bottom: 16px; letter-spacing: -0.5px;">Awaiting AI Analysis</div>
            <div style="font-size: 15px; color: #a1a1aa; line-height: 1.6; max-width: 500px; margin: 0 auto;">
                Return to the <b>Search</b> tab and enter a ticker symbol. The engine will automatically pull live data and generate a comprehensive institutional-grade brief here.
            </div>
        </div>
        ''')
    else:
        st.html('<div style="display:flex; justify-content:flex-end; margin-bottom: 24px;">')
        _t2_tickers = list(st.session_state.briefs.keys())
        _t2_labels = [f"{st.session_state.briefs[t].company_name} ({t})" for t in _t2_tickers]
        _t2_sel = st.selectbox("ACTIVE ASSET", _t2_labels, key="t2_target", label_visibility="collapsed")
        target = _t2_tickers[_t2_labels.index(_t2_sel)] if _t2_sel in _t2_labels else _t2_tickers[0]
        st.html('</div>')
        b = st.session_state.briefs[target]
        
        # LIVE MARKET DATA (yfinance)
        live = fetch_live_data(target)
        if live:
            def fmt_num(v, prefix='', suffix='', decimals=2):
                if v == 'N/A' or v is None: return 'N/A'
                try:
                    v = float(v)
                    if abs(v) >= 1e12: return f'{prefix}{v/1e12:.{decimals}f}T{suffix}'
                    if abs(v) >= 1e9: return f'{prefix}{v/1e9:.{decimals}f}B{suffix}'
                    if abs(v) >= 1e6: return f'{prefix}{v/1e6:.{decimals}f}M{suffix}'
                    return f'{prefix}{v:,.{decimals}f}{suffix}'
                except (ValueError, TypeError):
                    return str(v)
            
            st.html(f'''
            <div style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px;">
                <div style="flex:1; min-width:140px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:16px; padding:20px; text-align:center; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
                    <div style="font-size:11px; color:#71717a; font-weight:600; letter-spacing:1px; margin-bottom:8px;">LIVE PRICE</div>
                    <div style="font-size:20px; color:#10b981; font-family:'Inter',sans-serif; font-weight:800; letter-spacing:-0.5px;">${fmt_num(live['price'], decimals=2)}</div>
                </div>
                <div style="flex:1; min-width:140px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:16px; padding:20px; text-align:center; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
                    <div style="font-size:11px; color:#71717a; font-weight:600; letter-spacing:1px; margin-bottom:8px;">MARKET CAP</div>
                    <div style="font-size:20px; color:#ffffff; font-family:'Inter',sans-serif; font-weight:800; letter-spacing:-0.5px;">{fmt_num(live['market_cap'], prefix='$')}</div>
                </div>
                <div style="flex:1; min-width:140px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:16px; padding:20px; text-align:center; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
                    <div style="font-size:11px; color:#71717a; font-weight:600; letter-spacing:1px; margin-bottom:8px;">P/E RATIO</div>
                    <div style="font-size:20px; color:#ffffff; font-family:'Inter',sans-serif; font-weight:800; letter-spacing:-0.5px;">{fmt_num(live['pe_ratio'], suffix='x', decimals=1)}</div>
                </div>
                <div style="flex:1; min-width:140px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:16px; padding:20px; text-align:center; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
                    <div style="font-size:11px; color:#71717a; font-weight:600; letter-spacing:1px; margin-bottom:8px;">EV/EBITDA</div>
                    <div style="font-size:20px; color:#ffffff; font-family:'Inter',sans-serif; font-weight:800; letter-spacing:-0.5px;">{fmt_num(live['ev_ebitda'], suffix='x', decimals=1)}</div>
                </div>
                <div style="flex:1; min-width:140px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:16px; padding:20px; text-align:center; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
                    <div style="font-size:11px; color:#71717a; font-weight:600; letter-spacing:1px; margin-bottom:12px;">52W RANGE</div>
                    <div style="font-size:14px; color:#a1a1aa; font-family:'Inter',sans-serif; font-weight:500;">${fmt_num(live['week52_low'], decimals=0)} — ${fmt_num(live['week52_high'], decimals=0)}</div>
                </div>
                <div style="flex:1; min-width:140px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:16px; padding:20px; text-align:center; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
                    <div style="font-size:11px; color:#71717a; font-weight:600; letter-spacing:1px; margin-bottom:8px;">BETA</div>
                    <div style="font-size:20px; color:#fbbf24; font-family:'Inter',sans-serif; font-weight:800; letter-spacing:-0.5px;">{fmt_num(live['beta'], decimals=2) if live['beta'] != 'N/A' else 'N/A'}</div>
                </div>
            </div>
            ''')
        
        # ── GRU DEEP LEARNING SIGNAL ──
        with st.spinner('🧠 Training GRU neural network...'):
            gru_data = run_gru_prediction(target)
        
        if gru_data:
            sig = gru_data['signal']
            sig_color = '#10b981' if sig == 'BULLISH' else '#ef4444' if sig == 'BEARISH' else '#fbbf24'
            sig_icon = '▲' if sig == 'BULLISH' else '▼' if sig == 'BEARISH' else '■'
            arrow = '+' if gru_data['pct_change'] > 0 else ''
            
            st.html(f'''
            <div class="t-panel" style="margin-bottom: 24px; position:relative; overflow:hidden;">
                <!-- Glowing accent line -->
                <div style="position:absolute; top:0; left:0; right:0; height:4px; background: linear-gradient(90deg, transparent, {sig_color}, transparent);"></div>
                
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:24px;">
                    <div>
                        <div style="font-size:11px; color:#a1a1aa; font-weight:700; letter-spacing:2px; margin-bottom:8px;">DEEP LEARNING FORECAST</div>
                        <div style="font-size:24px; color:#ffffff; font-weight:800; font-family:'Inter',sans-serif; letter-spacing:-0.5px;">GRU Neural Network</div>
                    </div>
                    <div style="display:flex; gap:16px; align-items:center;">
                        <div style="text-align:right;">
                            <div style="font-size:10px; color:#71717a; font-weight:600; letter-spacing:1px; margin-bottom:4px;">CONFIDENCE</div>
                            <div style="font-size:20px; color:{sig_color}; font-family:'Inter',sans-serif; font-weight:800;">{gru_data['confidence']}%</div>
                        </div>
                        <div style="background:{sig_color}15; border:1px solid {sig_color}40; border-radius:12px; padding:10px 20px; display:flex; align-items:center; gap:8px; box-shadow: 0 0 20px {sig_color}20;">
                            <span style="font-size:18px; color:{sig_color};">{sig_icon}</span>
                            <span style="font-size:15px; color:{sig_color}; font-weight:800; letter-spacing:1.5px;">{sig}</span>
                        </div>
                    </div>
                </div>
                
                <div style="display:flex; gap:20px; margin-bottom:12px;">
                    <div style="flex:1; background:rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.05); border-radius:16px; padding:20px;">
                        <div style="font-size:11px; color:#71717a; font-weight:600; letter-spacing:1px; margin-bottom:8px;">CURRENT PRICE</div>
                        <div style="font-size:20px; color:#ffffff; font-family:'Inter',sans-serif; font-weight:800; letter-spacing:-1px;">${gru_data['last_price']:,.2f}</div>
                    </div>
                    <div style="flex:1; background:rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.05); border-radius:16px; padding:20px;">
                        <div style="font-size:11px; color:#71717a; font-weight:600; letter-spacing:1px; margin-bottom:8px;">10-DAY FORECAST</div>
                        <div style="font-size:20px; color:{sig_color}; font-family:'Inter',sans-serif; font-weight:800; letter-spacing:-1px;">${gru_data['pred_price']:,.2f}</div>
                    </div>
                    <div style="flex:1; background:rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.05); border-radius:16px; padding:20px;">
                        <div style="font-size:11px; color:#71717a; font-weight:600; letter-spacing:1px; margin-bottom:8px;">PREDICTED MOVE</div>
                        <div style="font-size:20px; color:{sig_color}; font-family:'Inter',sans-serif; font-weight:800; letter-spacing:-1px;">{arrow}{gru_data['pct_change']}%</div>
                    </div>
                </div>
            </div>
            ''')
            
            st.plotly_chart(c_gru_forecast_chart(gru_data), use_container_width=True)
            
            with st.expander("ℹ️ How the GRU Signal Works"):
                st.html('''
                <div style="font-size: 12px; color: #9ca3af; line-height: 1.8;">
                    <b style="color: #3b82f6;">Model Architecture:</b> Gated Recurrent Unit (GRU) — a type of recurrent neural network designed for sequential data. Uses 32 hidden units trained on 60-day sliding windows of historical daily closing prices.<br><br>
                    <b style="color: #3b82f6;">Training:</b> Fetches 1 year of daily price data via Yahoo Finance. The model is trained for 50 epochs with an 80/20 train/validation split using MSE loss and Adam optimizer.<br><br>
                    <b style="color: #3b82f6;">Signal Logic:</b><br>
                    • <span style="color:#10b981;">BULLISH</span>: Predicted 10-day price is >1.5% above current price<br>
                    • <span style="color:#ef4444;">BEARISH</span>: Predicted 10-day price is >1.5% below current price<br>
                    • <span style="color:#fbbf24;">NEUTRAL</span>: Predicted change is within ±1.5%<br><br>
                    <b style="color: #3b82f6;">Confidence Score:</b> Derived from inverse validation loss, capped between 60-95%. Higher scores indicate better model fit on held-out data.<br><br>
                    <span style="color: #6b7280;">⚠️ This is a demonstrator model for academic purposes. It does not constitute financial advice. Real trading signals require ensemble methods, external features, and rigorous backtesting.</span>
                </div>
                ''')
        else:
            st.error(st.session_state.get("gru_last_error", "GRU Model failed: Not enough historical data or Yahoo Finance network block."))
        
        # Dense Metric Grid
        st.html('<div class="metric-grid">')
        html = f'''
        <div class="m-card bull" style="background:rgba(16,185,129,0.05);">
            <div class="m-title">MGMT CONVICTION</div>
            <div class="m-val">{b.management_tone_score}<span style="font-size:16px;color:#71717a">/10</span></div>
            <div class="m-sub">▲ {b.management_tone} Sentiment</div>
        </div>
        <div class="m-card">
            <div class="m-title">FINANCIAL HEALTH</div>
            <div class="m-val">{b.competitive_scores.financial_health}<span style="font-size:16px;color:#71717a">/10</span></div>
            <div class="m-sub">Sector Benchmarked</div>
        </div>
        <div class="m-card bear" style="background:rgba(239,68,68,0.05);">
            <div class="m-title">RISK PROFILE</div>
            <div class="m-val">{b.competitive_scores.risk_profile}<span style="font-size:16px;color:#71717a">/10</span></div>
            <div class="m-sub danger">▼ Volatility Active</div>
        </div>
        <div class="m-card">
            <div class="m-title">INNOVATION SCORE</div>
            <div class="m-val">{b.competitive_scores.innovation}<span style="font-size:16px;color:#71717a">/10</span></div>
            <div class="m-sub">R&D Output Index</div>
        </div>
        '''
        st.html(html + '</div>')
        
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.html(f'''
            <div class="t-panel">
                <div class="t-panel-header">ANALYST VERDICT // EXEC SUMMARY</div>
                <div class="verdict-box"><div class="verdict-text">{b.analyst_verdict}</div></div>
            ''')
            
            st.html('<div class="t-panel-header" style="margin-top:32px;">CORE BUSINESS MODEL</div>')
            st.html(f'<div style="color:#d4d4d8; font-size:15px; line-height:1.7; margin-bottom: 32px; font-weight: 300;">{b.business_model}</div>')
            
            st.html('<div class="t-panel-header">FINANCIAL HIGHLIGHTS</div>')
            hls = ''.join([f'<li class="insight-item">{h}</li>' for h in b.financial_highlights])
            st.html(f'<ul class="insight-list">{hls}</ul>')
            st.html("</div>")
        
        with c_right:
            st.html(f'''
            <div class="t-panel" style="border-top: 4px solid #10b981; background: linear-gradient(180deg, rgba(16,185,129,0.05) 0%, rgba(15,15,15,0.6) 100%);">
                <div class="t-panel-header" style="color: #10b981; border-bottom:none;">BULL CASE SCENARIO</div>
                <div style="font-size:14px; color:#d4d4d8; line-height:1.6; font-weight:300;">{b.bull_case}</div>
            </div>
            
            <div class="t-panel" style="border-top: 4px solid #ef4444; background: linear-gradient(180deg, rgba(239,68,68,0.05) 0%, rgba(15,15,15,0.6) 100%);">
                <div class="t-panel-header" style="color: #ef4444; border-bottom:none;">BEAR CASE SCENARIO</div>
                <div style="font-size:14px; color:#d4d4d8; line-height:1.6; font-weight:300;">{b.bear_case}</div>
            </div>
            ''')
        

        
        # PDF Export
        try:
            dcf_pv, _ = simulate_dcf(float(b.fcf_base), float(b.wacc), float(b.growth_rate), 5)
            pdf_bytes = generate_report_pdf(b, target, pv=dcf_pv)
            st.download_button(
                label="📥 Export Analysis Report (PDF)",
                data=pdf_bytes,
                file_name=f"nexus_report_{target}.pdf",
                mime="application/pdf",
                key=f"pdf_{target}"
            )
        except Exception as e:
            st.warning(f"PDF export temporarily unavailable: {str(e)[:60]}")

# --- TAB 3: MACRO UNIVERSE ---
with tabs[2]:
    if not st.session_state.macro_db:
        st.html('''
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
        ''')
    else:
        st.html('<div class="t-panel-header" style="margin-bottom: 16px;">MARKET COMPARISON MODULE</div>')
        
        t3_tabs = st.tabs(["🎯 Individual Profile", "⚖️ Head-to-Head Comparison", "🔗 Correlation Matrix"])
        
        with t3_tabs[0]:
            st.info("Select a company to see its strengths and weaknesses visualized as a radar chart.")
            _t3_tickers = list(st.session_state.macro_db.keys())
            _t3_labels = [f"{st.session_state.briefs[t].company_name} ({t})" if t in st.session_state.briefs else t for t in _t3_tickers]
            _t3_sel = st.selectbox("Select Target Company", _t3_labels, key="t3_target_single")
            target_t3 = _t3_tickers[_t3_labels.index(_t3_sel)] if _t3_sel in _t3_labels else _t3_tickers[0]
            
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
                inn_v = scores_flat.get('innovation', 5)
                fin_v = scores_flat.get('financial_health', 5)
                risk_v = scores_flat.get('risk_profile', 5)
                mkt_v = scores_flat.get('market_position', 5)
                
                def interpret(val): return '🟢 Strong' if val >= 7 else ('🟡 Average' if val >= 4 else '🔴 Weak')
                def risk_interpret(val): return '🟢 Low Risk' if val <= 3 else ('🟡 Moderate' if val <= 6 else '🔴 High Risk')
                
                st.html(f'''
                <div class="t-panel" style="margin-top: 10px; background: #0a0a0a;">
                    <div class="t-panel-header">WHAT THIS MEANS</div>
                    <div style="font-size: 13px; color: #d1d5db; line-height: 2;">
                        <b>Innovation ({inn_v}/10):</b> {interpret(inn_v)} — R&D and product pipeline strength<br>
                        <b>Market Position ({mkt_v}/10):</b> {interpret(mkt_v)} — competitive market dominance<br>
                        <b>Financial Health ({fin_v}/10):</b> {interpret(fin_v)} — balance sheet and cash reserves<br>
                        <b>Risk Profile ({risk_v}/10):</b> {risk_interpret(risk_v)} — exposure to external threats<br>
                        <b>Mgmt Tone ({mgmt}/10):</b> {interpret(mgmt)} — how confident is the leadership team<br>
                    </div>
                </div>
                ''')
                
        with t3_tabs[1]:
            st.info("Select two or more companies to compare them head-to-head across key competitive dimensions.")
            if len(st.session_state.macro_db) >= 2:
                # Default to first two companies in DB
                default_targets = list(st.session_state.macro_db.keys())[:2]
                compare_targets = st.multiselect("Select Assets to Compare", list(st.session_state.macro_db.keys()), default=default_targets, key="t3_compare_multi")
                
                if len(compare_targets) >= 2:
                    filtered_db = {k: st.session_state.macro_db[k] for k in compare_targets}
                    st.plotly_chart(c_comparison_bars(filtered_db), use_container_width=True)
                else:
                    st.warning("⚠️ Please select at least 2 companies from the dropdown above to generate the comparison chart.")
            else:
                st.warning("⚠️ You need to upload and ingest at least 2 companies in Tab 1 to unlock Head-to-Head Comparison.")
                
        with t3_tabs[2]:
            st.info("CLASSICAL ANALYTICS: This is a statistical correlation heatmap (not AI-generated). It uses the Pearson coefficient to measure linear relationships between metrics across all analyzed companies.")
            if len(st.session_state.macro_db) >= 2:
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
                    corr_matrix = corr_df.corr().fillna(0) # Fix NaN when variance is 0
                    
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
                    st.html('''
                    <div style="font-size: 12px; color: #9ca3af; line-height: 1.6; padding: 12px; background: #0a0a0a; border: 1px solid #1f2937; border-radius: 4px; margin-top: 10px;">
                        <b style="color: #fbbf24;">How to read this:</b> Each cell shows the Pearson correlation coefficient (-1 to +1) between two metrics.
                        <b style="color: #10b981;">Green = positively correlated</b> (when one goes up, the other tends to go up).
                        <b style="color: #ef4444;">Red = negatively correlated</b> (when one goes up, the other tends to go down).
                        This is a classical statistical method computed across your entire custom cohort, not an AI generation.
                    </div>
                    ''')
            else:
                st.warning("⚠️ You need to upload and ingest at least 2 companies in Tab 1 to generate the Correlation Matrix.")
                
# --- TAB 4: DCF SIMULATOR ---
with tabs[3]:
    if not st.session_state.briefs:
        st.html('''
        <div class="t-panel" style="text-align:center; padding: 60px 40px; color: #6b7280;">
            <div style="font-size: 40px; margin-bottom: 16px;">💰</div>
            <div style="font-size: 16px; color: #9ca3af; font-weight: 600; margin-bottom: 12px;">Intrinsic Value Calculator (DCF Model)</div>
            <div style="font-size: 13px; line-height: 1.6; max-width: 500px; margin: 0 auto;">
                This tab calculates how much a company is truly worth using a <b>Discounted Cash Flow</b> model — one of the most popular methods used by Wall Street analysts.<br><br>
                • The AI auto-extracts financial inputs (cash flow, growth rate, cost of capital) from the report<br>
                • You can override any number and watch the <b>intrinsic value chart update in real-time</b><br>
                • A bar chart visualizes future cash flows and the terminal value<br><br>
                <div style="margin-top: 20px; padding: 12px; border: 1px solid #fbbf24; background: rgba(251, 191, 36, 0.1); border-radius: 6px; display: inline-block;">
                    <b style="color: #fbbf24;">Action Required:</b> Please click the <b>"📄 Upload Reports"</b> tab at the top of the screen to ingest your first report.
                </div>
            </div>
        </div>
        ''')
    else:
        st.html('<div style="display:flex; justify-content:flex-end; margin-bottom: 10px;">')
        _t4_tickers = list(st.session_state.briefs.keys())
        _t4_labels = [f"{st.session_state.briefs[t].company_name} ({t})" for t in _t4_tickers]
        _t4_sel = st.selectbox("ACTIVE ASSET", _t4_labels, key="t4_target", label_visibility="collapsed")
        target_dcf = _t4_tickers[_t4_labels.index(_t4_sel)] if _t4_sel in _t4_labels else _t4_tickers[0]
        st.html('</div>')
        b = st.session_state.briefs[target_dcf]
        
        dc1, dc2 = st.columns([1, 2])
        with dc1:
            st.html('<div class="t-panel">')
            st.html('<div class="t-panel-header">DCF VARIABLES</div>')
            st.info("The Baseline Variables below were dynamically extracted by the LLM from the 10-K. Override them to model intrinsic value changes.")
            
            st.html('<div style="font-size: 10px; color: #6b7280; margin-bottom: 4px; margin-top: 12px;">DISCOUNT RATE (WACC)</div>')
            wacc = st.number_input("WACC", min_value=0.01, max_value=0.30, value=float(b.wacc), step=0.005, format="%.3f", label_visibility="collapsed", key=f"wacc_{target_dcf}", help="Weighted Average Cost of Capital. Determines the discount applied to future cash flows.")
            
            st.html('<div style="font-size: 10px; color: #6b7280; margin: 16px 0 4px 0;">TERMINAL GROWTH RATE</div>')
            g = st.number_input("Growth", min_value=-0.05, max_value=0.10, value=float(b.growth_rate), step=0.005, format="%.3f", label_visibility="collapsed", key=f"g_{target_dcf}", help="The expected perpetual growth rate of the asset after the projection horizon.")
            
            st.html('<div style="font-size: 10px; color: #6b7280; margin: 16px 0 4px 0;">BASE FCF (MILLIONS)</div>')
            fcf = st.number_input("FCF", value=float(b.fcf_base), step=100.0, label_visibility="collapsed", key=f"fcf_{target_dcf}", help="Trailing Twelve Months (TTM) Free Cash Flow baseline.")
            
            st.html('<div style="font-size: 10px; color: #6b7280; margin: 16px 0 4px 0;">PROJECTION HORIZON (YRS)</div>')
            years = st.slider("Years", 3, 10, 5, label_visibility="collapsed", key=f"yrs_{target_dcf}")
            st.html('</div>')
            
        with dc2:
            st.html('<div class="t-panel">')
            st.html('<div class="t-panel-header">INTRINSIC VALUE MODELER</div>')
            pv, vals = simulate_dcf(fcf, wacc, g, years)
            tv = pv - sum(vals)
            st.plotly_chart(c_dcf_sim(vals, tv, pv, years), use_container_width=True)
            
            # Plain English Interpretation
            st.html(f'''
            <div style="font-size: 13px; color: #d1d5db; line-height: 1.8; padding: 12px; background: #0a0a0a; border: 1px solid #1f2937; border-radius: 4px; margin-top: 8px;">
                <b style="color: #fbbf24;">What this means:</b> Based on the current inputs, the AI estimates that <b>{target_dcf}</b> is worth approximately <b style="color: #10b981;">${pv:,.0f}M</b>. 
                This is calculated by projecting {years} years of future cash flows (blue bars) and adding the present value of all cash flows beyond that horizon (purple bar = Terminal Value). 
                Try adjusting the sliders on the left to see how different assumptions change the valuation.
            </div>
            ''')
            st.html('</div>')
        
        # CLASSICAL ANALYTICS: Monte Carlo Simulation
        st.html('<div class="t-panel">')
        st.html('<div class="t-panel-header">CLASSICAL ANALYTICS: MONTE CARLO SIMULATION (1,000 TRIALS)</div>')
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
            fig_mc.update_layout(**BASE_CHART, height=350, margin=dict(l=10, r=10, t=30, b=10),
                                xaxis=dict(title='Estimated Valuation ($M)', gridcolor='#1f2937'),
                                yaxis=dict(title='Frequency', gridcolor='#1f2937'),
                                showlegend=False)
            
            mc1, mc2 = st.columns([2, 1])
            with mc1:
                st.plotly_chart(fig_mc, use_container_width=True)
            with mc2:
                st.html(f'''
                <div style="display: flex; flex-direction: column; gap: 16px; margin-top: 10px;">
                    <div style="background: #0a0a0a; border: 1px solid #7f1d1d; border-radius: 6px; padding: 16px; text-align: center;">
                        <div style="font-size: 11px; color: #ef4444; font-weight: bold; letter-spacing: 0.5px;">BEARISH SKEW (5th %ile)</div>
                        <div style="font-size: 26px; color: #ef4444; font-family: 'JetBrains Mono', monospace; font-weight: bold;">${p5:,.0f}M</div>
                    </div>
                    <div style="background: #0a0a0a; border: 1px solid #92400e; border-radius: 6px; padding: 16px; text-align: center;">
                        <div style="font-size: 11px; color: #fbbf24; font-weight: bold; letter-spacing: 0.5px;">MEDIAN EXPECTATION</div>
                        <div style="font-size: 26px; color: #fbbf24; font-family: 'JetBrains Mono', monospace; font-weight: bold;">${p50:,.0f}M</div>
                    </div>
                    <div style="background: #0a0a0a; border: 1px solid #064e3b; border-radius: 6px; padding: 16px; text-align: center;">
                        <div style="font-size: 11px; color: #10b981; font-weight: bold; letter-spacing: 0.5px;">BULLISH SKEW (95th %ile)</div>
                        <div style="font-size: 26px; color: #10b981; font-family: 'JetBrains Mono', monospace; font-weight: bold;">${p95:,.0f}M</div>
                    </div>
                </div>
                ''')
                
            st.html(f'''
            <div style="font-size: 12px; color: #9ca3af; margin-top: 16px; padding: 12px; background: #0a0a0a; border: 1px solid #1f2937; border-radius: 4px;">
                <b style="color: #fbbf24;">What this means:</b> Out of 1,000 randomized scenarios, {target_dcf}'s valuation falls between 
                <b style="color: #ef4444;">${p5:,.0f}M</b> (pessimistic) and <b style="color: #10b981;">${p95:,.0f}M</b> (optimistic) with 90% confidence.
                The median estimate is <b style="color: #fbbf24;">${p50:,.0f}M</b>. This validates the point estimate against parameter uncertainty.
            </div>
            ''')
        
        st.html('</div>')
        
        # SENSITIVITY ANALYSIS: WACC × Growth Rate Matrix
        st.html('<div class="t-panel-header" style="margin-top: 20px; margin-bottom: 12px;">SENSITIVITY ANALYSIS: WACC × GROWTH RATE MATRIX</div>')
        st.info("This classic equity research tool shows how the intrinsic value changes across different combinations of WACC (discount rate) and terminal growth rate. The highlighted cell is your current base case. Green = above base case; Red = below base case.")
        
        sens_df = build_sensitivity_table(fcf, wacc, g, years)
        base_val = pv
        
        # Format for display
        styled_data = []
        for idx, row in sens_df.iterrows():
            styled_row = {}
            for col in sens_df.columns:
                val = row[col]
                if val is not None and not pd.isna(val):
                    styled_row[col] = f"${val:,.0f}M"
                else:
                    styled_row[col] = "N/A"
            styled_data.append(styled_row)
        display_df = pd.DataFrame(styled_data, index=sens_df.index)
        
        # Build HTML table with conditional formatting
        table_html = '<table style="width:100%; border-collapse:collapse; font-family:\'JetBrains Mono\',monospace; font-size:12px;">'
        table_html += '<tr style="background:#0a0a0a;"><th style="padding:10px; border:1px solid #1f2937; color:#6b7280;">WACC \\ Growth</th>'
        for col in display_df.columns:
            table_html += f'<th style="padding:10px; border:1px solid #1f2937; color:#fbbf24; text-align:center;">{col}</th>'
        table_html += '</tr>'
        
        for i, (idx, row) in enumerate(display_df.iterrows()):
            table_html += f'<tr><td style="padding:10px; border:1px solid #1f2937; color:#3b82f6; font-weight:bold; background:#0a0a0a;">{idx}</td>'
            for j, col in enumerate(display_df.columns):
                raw_val = sens_df.iloc[i, j]
                cell_val = row[col]
                if raw_val is not None and not pd.isna(raw_val):
                    if raw_val > base_val * 1.05:
                        bg = 'rgba(16, 185, 129, 0.15)'; clr = '#10b981'
                    elif raw_val < base_val * 0.95:
                        bg = 'rgba(239, 68, 68, 0.15)'; clr = '#ef4444'
                    else:
                        bg = 'rgba(251, 191, 36, 0.15)'; clr = '#fbbf24'
                else:
                    bg = '#111'; clr = '#6b7280'
                table_html += f'<td style="padding:10px; border:1px solid #1f2937; text-align:center; background:{bg}; color:{clr};">{cell_val}</td>'
            table_html += '</tr>'
        table_html += '</table>'
        
        st.html(table_html)
        st.html('''
        <div style="font-size: 11px; color: #6b7280; margin-top: 8px; text-align: center;">
            <b style="color:#10b981;">■</b> Above base case &nbsp;&nbsp; <b style="color:#fbbf24;">■</b> Near base case (±5%) &nbsp;&nbsp; <b style="color:#ef4444;">■</b> Below base case
        </div>
        ''')

# --- TAB 5: INDUSTRY BENCHMARK ---
with tabs[4]:
    if not st.session_state.macro_db:
        st.html('''
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
        ''')
    else:
        st.html('<div class="t-panel-header" style="margin-bottom: 16px;">INDUSTRY BENCHMARK SUMMARY</div>')
        st.info("Select an industry to algorithmically benchmark all ingested companies against their sector averages. The Quantitative Sector Matrix uses color gradients to highlight outperformance (darker green) versus underperformance.")
        
        # Get unique sectors
        sectors = set()
        for v in st.session_state.macro_db.values():
            sect = v.get('sector', 'Unknown') if isinstance(v, dict) else getattr(v, 'sector', 'Unknown')
            sectors.add(sect)
            
        c1, c2 = st.columns([1, 4])
        with c1:
            st.html('<div style="font-size: 10px; color: #6b7280; margin-bottom: 8px; margin-top: 12px;">TARGET INDUSTRY</div>')
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
                st.html('<div class="metric-grid">')
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
                st.html(html + '</div>')
                
            st.html("<hr style='border-color: #1f2937; margin: 20px 0;'>")
            
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
            
            st.html('<div style="font-size: 11px; color: #9ca3af; margin-bottom: 8px;">QUANTITATIVE SECTOR MATRIX</div>')
            
            # Professional Screener Aesthetic using Native Streamlit Configurations (zero matplotlib dependency)
            st.dataframe(
                df,
                column_config={
                    "Innovation": st.column_config.ProgressColumn("Innovation", help="R&D Pipeline Score", format="%d", min_value=0, max_value=10),
                    "Financial Health": st.column_config.ProgressColumn("Financial Health", help="Balance Sheet Strength", format="%d", min_value=0, max_value=10),
                    "Risk Profile": st.column_config.ProgressColumn("Risk Profile", help="Lower is better", format="%d", min_value=0, max_value=10),
                    "Mgmt Tone": st.column_config.ProgressColumn("Mgmt Tone", help="Leadership confidence", format="%d", min_value=0, max_value=10),
                },
                use_container_width=True
            )
            
        else:
            st.warning("No companies found in this sector yet.")
