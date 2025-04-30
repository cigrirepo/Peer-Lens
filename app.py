import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import openai
from scipy.stats import rankdata
from io import BytesIO
from typing import List, Dict, Any

# === Constants ===
SEC_XBRL_BASE      = "https://data.sec.gov/api/xbrl/companyfacts/"
SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT         = "PeerLensBenchmarkStudio/0.1 (your_email@example.com)"

# Cache for ticker→CIK mapping
ticker_cik_map: Dict[str, str] = {}

# === Helper Functions ===
def fetch_public_filer_ciks(tickers: List[str]) -> Dict[str, str]:
    """
    Fetch and cache zero-padded 10-digit CIKs for tickers.
    """
    global ticker_cik_map
    if not ticker_cik_map:
        resp = requests.get(SEC_TICKER_CIK_URL, headers={"User-Agent": USER_AGENT})
        if resp.status_code == 200:
            data = resp.json()
            for record in data.values():
                tic = str(record.get('ticker', '')).upper()
                cik_raw = record.get('cik_str', '')
                if tic and cik_raw != '':
                    ticker_cik_map[tic] = str(cik_raw).zfill(10)
        else:
            st.error(f"Failed to load ticker-CIK map: HTTP {resp.status_code}")
    result = {}
    for t in tickers:
        cik = ticker_cik_map.get(t)
        if cik:
            result[t] = cik
        else:
            result[t] = None
            st.warning(f"CIK not found for ticker '{t}'")
    return result

def fetch_xbrl_data(cik: str) -> Dict[str, Any]:
    """
    Retrieve XBRL JSON for a CIK (must prefix with 'CIK').
    """
    url = f"{SEC_XBRL_BASE}CIK{cik}.json"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT})
    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code == 404:
        st.warning(f"No XBRL data for CIK {cik} (404)")
    else:
        st.error(f"Error fetching XBRL for CIK {cik}: HTTP {resp.status_code}")
    return {}

def extract_financials(xbrl_data: Dict[str, Any], facts: List[str]) -> Dict[str, float]:
    """
    Extract latest US-GAAP fact values.
    """
    results = {}
    facts_usg = xbrl_data.get('facts', {}).get('us-gaap', {})
    for fact in facts:
        key = fact.lower() if fact.lower() in facts_usg else fact
        block = facts_usg.get(key, {})
        units = block.get('units', {})
        value = np.nan
        for arr in units.values():
            if isinstance(arr, list) and arr:
                v = arr[-1].get('v')
                try:
                    value = float(v)
                except:
                    value = np.nan
                break
        results[fact] = value
    return results

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute KPIs and percentile ranks.
    """
    df = df.copy()
    df['ebitda_margin'] = df['Ebitda'] / df['Revenues']
    df['rev_cagr_3yr']   = (df['Revenues'] / df['Revenues'].shift(3)) ** (1/3) - 1
    df['ev_ebitda']      = df['MarketCapitalization'] / df['Ebitda']
    df['debt_to_asset']  = df['Liabilities'] / df['Assets']
    df['roa']            = df['Ebitda'] / df['Assets']
    for col in ['ebitda_margin','rev_cagr_3yr','ev_ebitda','debt_to_asset','roa']:
        df[f'{col}_pct'] = rankdata(df[col].fillna(0), method='average') / len(df) * 100
    return df

def generate_narrative(df: pd.DataFrame) -> str:
    """
    Generate a ~120-word narrative using the OpenAI Python ≥1.0.0 interface.
    """
    api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
    if not api_key:
        st.error('OpenAI API key not configured.')
        return ''
    openai.api_key = api_key

    data = df.to_json(orient='records')
    messages = [
        {"role": "system", "content": (
            "You are an expert financial analyst. "
            "Summarize the following peer KPI JSON in approximately 120 words, "
            "highlighting valuation multiples, growth rates, and margin comparisons."
        )},
        {"role": "user", "content": data}
    ]

    try:
        # New style for openai-python ≥1.0.0
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return ""

# === Streamlit App ===
def main():
    st.set_page_config(page_title="PeerLens Benchmark Studio", layout="wide")
    st.title("📊 PeerLens Benchmark Studio")
    st.markdown("Upload peer tickers or a CSV of peers to benchmark key financial metrics.")

    tickers = st.sidebar.text_input("Tickers (comma-separated)")
    file    = st.sidebar.file_uploader("CSV with 'Ticker' column", type='csv')

    if st.sidebar.button("Generate Benchmark"):
        if file:
            df_in  = pd.read_csv(file)
            peers  = df_in['Ticker'].astype(str).str.upper().tolist()
        else:
            peers = [t.strip().upper() for t in tickers.split(',') if t.strip()]

        if not peers:
            st.error("No tickers provided.")
            return

        st.info(f"Fetching data for {len(peers)} peers...")
        cik_map = fetch_public_filer_ciks(peers)

        records = []
        facts   = ['Revenues','Ebitda','Assets','Liabilities','MarketCapitalization']
        for t in peers:
            cik   = cik_map.get(t)
            xbrl  = fetch_xbrl_data(cik) if cik else {}
            fin   = extract_financials(xbrl, facts)
            fin['Ticker'] = t
            records.append(fin)

        df_fin = pd.DataFrame(records)
        if df_fin.empty:
            st.error("No financial data fetched. Check tickers or try CSV.")
            return

        df_kpis = compute_metrics(df_fin)
        st.subheader("Peer KPI Table")
        st.dataframe(df_kpis)

        st.subheader("AI-Generated Narrative")
        st.write(generate_narrative(df_kpis))

        # Excel export
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_kpis.to_excel(writer, index=False, sheet_name='KPIs')
        buf.seek(0)
        st.sidebar.download_button(
            "Download XLSX",
            data=buf.getvalue(),
            file_name='peer_kpis.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

if __name__ == '__main__':
    main()
