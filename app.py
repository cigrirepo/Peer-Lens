import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import openai
from scipy.stats import rankdata
from io import BytesIO
from typing import List, Dict, Any

# === Constants ===
SEC_XBRL_BASE      = "https://data.sec.gov/api/xbrl/companyfacts/"
SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT         = "PeerLensBenchmarkStudio/1.0 (you@example.com)"
FACT_TAGS = {
    "Revenues": "Revenues",
    "EBITDA": "EarningsBeforeInterestTaxesDepreciationAndAmortization",
    "Assets": "Assets",
    "Liabilities": "Liabilities",
    "MarketCapitalization": "MarketCapitalization"
}

# === Caching ===
@st.cache_data(ttl=24*3600)
def load_ticker_cik_map() -> Dict[str, str]:
    resp = requests.get(SEC_TICKER_CIK_URL, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    data = resp.json()
    mapping = {}
    for rec in data.values():
        t = rec.get("ticker","").upper()
        cik = str(rec.get("cik_str","")).zfill(10)
        if t and cik:
            mapping[t] = cik
    return mapping

@st.cache_data(ttl=24*3600)
def fetch_xbrl(cik: str) -> Dict[str, Any]:
    url = f"{SEC_XBRL_BASE}CIK{cik}.json"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT})
    if resp.status_code == 200:
        return resp.json().get("facts", {}).get("us-gaap", {})
    return {}

# === Extraction & Metrics ===
def extract_financials(usg: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for label, tag in FACT_TAGS.items():
        block = usg.get(tag, {})
        val = np.nan
        for unit_vals in block.get("units", {}).values():
            if isinstance(unit_vals, list) and unit_vals:
                try:
                    val = float(unit_vals[-1].get("v", np.nan))
                except:
                    val = np.nan
                break
        out[label] = val
    return out

def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EBITDA Margin"] = df["EBITDA"] / df["Revenues"]
    df["Revenue CAGR (3y)"] = (df["Revenues"] / df["Revenues"].shift(3)) ** (1/3) - 1
    df["EV/EBITDA"] = df["MarketCapitalization"] / df["EBITDA"]
    df["Debt/Assets"] = df["Liabilities"] / df["Assets"]
    df["ROA"] = df["EBITDA"] / df["Assets"]
    for col in ["EBITDA Margin","Revenue CAGR (3y)","EV/EBITDA","Debt/Assets","ROA"]:
        df[f"{col} pct"] = rankdata(df[col].fillna(0), method="average") / len(df) * 100
    return df

# === OpenAI Narrative ===
def generate_narrative(df: pd.DataFrame) -> str:
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not key:
        return "🔑 OpenAI API key not set."
    openai.api_key = key
    payload = df.to_dict("records")
    messages = [
        {"role":"system","content":"You are a seasoned financial analyst."},
        {"role":"user","content":f"Here are peer financial KPIs:\n{payload}\nWrite a concise 120-word analysis comparing margins, growth and valuation."}
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.2, max_tokens=250
    )
    return resp.choices[0].message.content.strip()

# === Streamlit App ===
def main():
    st.set_page_config(page_title="PeerLens Benchmark Studio", layout="wide")
    st.title("📊 PeerLens Benchmark Studio")
    st.sidebar.header("Peer Inputs")
    tickers = st.sidebar.text_input("Tickers (comma-separated)", help="e.g. AAPL, MSFT")
    upload  = st.sidebar.file_uploader("Or upload CSV with 'Ticker' column", type="csv")

    if st.sidebar.button("Generate Benchmark"):
        # build list
        if upload:
            df_in = pd.read_csv(upload)
            peers = df_in["Ticker"].astype(str).str.upper().tolist()
        else:
            peers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

        if not peers:
            st.error("Please enter tickers or upload a CSV.")
            return

        st.info(f"Fetching XBRL data for {len(peers)} peers…")
        mapping = load_ticker_cik_map()
        records = []
        for t in peers:
            cik = mapping.get(t)
            if not cik:
                st.warning(f"No CIK found for {t}")
                continue
            usg = fetch_xbrl(cik)
            if not usg:
                st.warning(f"No XBRL facts for {t}")
            fin = extract_financials(usg)
            fin["Ticker"] = t
            records.append(fin)

        df = pd.DataFrame(records)
        if df.empty:
            st.error("No financial data retrieved. Check your tickers.")
            return

        df_kpis = compute_kpis(df)
        st.subheader("Peer KPI Table")
        st.dataframe(df_kpis, use_container_width=True)

        st.subheader("AI-Generated Narrative")
        st.write(generate_narrative(df_kpis))

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_kpis.to_excel(writer, index=False, sheet_name="KPIs")
        buffer.seek(0)
        st.sidebar.download_button(
            "Download as XLSX",
            data=buffer.getvalue(),
            file_name="peer_kpis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
