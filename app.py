import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import openai
import yfinance as yf
from scipy.stats import rankdata
from io import BytesIO
from typing import List, Dict, Any

# === Constants ===
SEC_XBRL_BASE      = "https://data.sec.gov/api/xbrl/companyfacts/"
SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT         = "PeerLensBenchmarkStudio/1.0 (your_email@example.com)"

FACT_TAGS = {
    "Revenues": "Revenues",
    "EBITDA": "EarningsBeforeInterestTaxesDepreciationAndAmortization",
    "Assets": "Assets",
    "Liabilities": "Liabilities"
    # note: MarketCapitalization isn’t a US-GAAP XBRL tag
}

# === Caching ===
@st.cache_data(ttl=24*3600)
def load_ticker_cik_map() -> Dict[str, str]:
    resp = requests.get(SEC_TICKER_CIK_URL, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    data = resp.json()
    m: Dict[str, str] = {}
    for rec in data.values():
        tic = rec.get("ticker","").upper()
        cik = rec.get("cik_str","")
        if tic and cik:
            m[tic] = cik.zfill(10)
    return m

@st.cache_data(ttl=24*3600)
def fetch_xbrl_facts(cik10: str) -> Dict[str, Any]:
    # **Must** prefix with “CIK” per SEC’s naming convention:
    url = f"{SEC_XBRL_BASE}CIK{cik10}.json"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT})
    if resp.status_code == 200:
        return resp.json().get("facts", {}).get("us-gaap", {})
    # 404 or other
    st.warning(f"SEC XBRL fetch failed for CIK {cik10}: HTTP {resp.status_code}")
    return {}

# === Extraction & Metrics ===
def extract_financials(usg: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, tag in FACT_TAGS.items():
        val = np.nan
        blk = usg.get(tag, {})
        usd = blk.get("units", {}).get("USD", [])
        if usd:
            try:
                val = float(usd[-1].get("v", np.nan))
            except:
                val = np.nan
        out[label] = val
    return out

def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # core margins/growth
    df["EBITDA Margin"]     = df["EBITDA"] / df["Revenues"]
    df["Revenue CAGR (3y)"] = (df["Revenues"] / df["Revenues"].shift(3))**(1/3) - 1
    # market cap fallback
    # (we’ll inject MarketCapitalization from yfinance in main below)
    df["EV/EBITDA"]          = df["MarketCapitalization"] / df["EBITDA"]
    df["Debt/Assets"]        = df["Liabilities"] / df["Assets"]
    df["ROA"]                = df["EBITDA"] / df["Assets"]

    for col in ["EBITDA Margin","Revenue CAGR (3y)","EV/EBITDA","Debt/Assets","ROA"]:
        df[f"{col} pct"] = rankdata(df[col].fillna(0), method="average") / len(df) * 100

    return df

# === OpenAI Narrative ===
def generate_narrative(df: pd.DataFrame) -> str:
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not key:
        return "🔑 OpenAI key not set."
    openai.api_key = key

    recs = df.to_dict("records")
    messages = [
        {"role":"system","content":"You are a financial analyst."},
        {"role":"user","content":
            f"Peer KPI data:\n{recs}\n\n"
            "Write a concise 120-word summary comparing valuation multiples, margins, and growth."
        }
    ]
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.2, max_tokens=250
    )
    return resp.choices[0].message.content.strip()

# === Streamlit App ===
def main():
    st.set_page_config(page_title="PeerLens Benchmark", layout="wide")
    st.title("📊 PeerLens Benchmark Studio")
    st.sidebar.header("Peer Inputs")

    t_input = st.sidebar.text_input("Tickers (comma-separated)", 
                                    placeholder="AAPL, MSFT, NFLX")
    csv_up   = st.sidebar.file_uploader("Or upload CSV with 'Ticker' column", type="csv")

    if st.sidebar.button("Generate Benchmark"):
        # build list
        if csv_up:
            try:
                df_in = pd.read_csv(csv_up)
                peers = df_in["Ticker"].astype(str).str.upper().tolist()
            except Exception as e:
                st.error(f"CSV read error: {e}")
                return
        else:
            peers = [t.strip().upper() for t in t_input
