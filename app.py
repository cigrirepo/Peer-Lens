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

FACTS_MAPPING = {
    'Revenues': 'Revenues',
    'EBITDA': 'EarningsBeforeInterestTaxesDepreciationAndAmortization',
    'Assets': 'Assets',
    'Liabilities': 'Liabilities'
}

# === Helper Functions ===

def fetch_public_filer_ciks(tickers: List[str]) -> Dict[str, str]:
    global ticker_cik_map
    if not ticker_cik_map:
        resp = requests.get(SEC_TICKER_CIK_URL, headers={"User-Agent": USER_AGENT})
        if resp.status_code == 200:
            data = resp.json()
            for record in data.values():
                tic = str(record.get('ticker', '')).upper()
                cik_raw = record.get('cik_str', '')
                if tic and cik_raw:
                    ticker_cik_map[tic] = str(cik_raw).zfill(10)
        else:
            st.error(f"Failed to load ticker-CIK map: HTTP {resp.status_code}")
    return {t: ticker_cik_map.get(t) for t in tickers}


def fetch_xbrl_data(cik: str) -> Dict[str, Any]:
    url = f"{SEC_XBRL_BASE}CIK{cik}.json"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT})
    if resp.status_code == 200:
        return resp.json()
    st.warning(f"XBRL not found for CIK {cik} (HTTP {resp.status_code})")
    return {}


def extract_financials(xbrl_data: Dict[str, Any]) -> Dict[str, float]:
    res = {}
    facts_usg = xbrl_data.get('facts', {}).get('us-gaap', {})
    for label, tag in FACTS_MAPPING.items():
        block = facts_usg.get(tag, {})
        units = block.get('units', {})
        val = np.nan
        for arr in units.values():
            if arr:
                try:
                    val = float(arr[-1].get('v'))
                except:
                    val = np.nan
                break
        res[label] = val
    return res


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # EBITDA margin
    df['EBITDA Margin'] = df['EBITDA'] / df['Revenues']
    # Revenue CAGR 3y
    df['Revenue CAGR (3y)'] = (df['Revenues'] / df['Revenues'].shift(3)) ** (1/3) - 1
    # Percentile ranks
    for col in ['EBITDA Margin', 'Revenue CAGR (3y)']:
        df[f'{col} pct'] = rankdata(df[col].fillna(0), method='average') / len(df) * 100
    return df


def generate_narrative(df: pd.DataFrame) -> str:
    api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
    if not api_key:
        return "API key not set."
    openai.api_key = api_key
    # Build summary only on available metrics
    rows = df.to_dict('records')
    content = json.dumps(rows)
    messages = [
        {"role":"system","content":"You are a financial analyst."},
        {"role":"user","content":(
            "Summarize these peer KPIs in 5 bullets: " + content
        )}
    ]
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.2
    )
    return resp.choices[0].message.content.strip()

# === Streamlit App ===

def main():
    st.set_page_config(page_title='PeerLens', layout='wide')
    st.title('📊 PeerLens Benchmark Studio')
    tickers_input = st.sidebar.text_input('Tickers (comma-separated)')
    csv_file = st.sidebar.file_uploader("CSV with 'Ticker' column", type='csv')

    if st.sidebar.button('Generate Benchmark'):
        if csv_file:
            peers = pd.read_csv(csv_file)['Ticker'].str.upper().tolist()
        else:
            peers = [t.strip().upper() for t in tickers_input.split(',') if t]

        if not peers:
            st.error('Provide tickers or CSV.')
            return

        st.info(f'Fetching {len(peers)} peers...')
        ciks = fetch_public_filer_ciks(peers)
        records = []
        for t in peers:
            cik = ciks.get(t)
            data = fetch_xbrl_data(cik) if cik else {}
            fin = extract_financials(data)
            fin['Ticker'] = t
            records.append(fin)

        df = pd.DataFrame(records)
        if df[['Revenues','EBITDA']].isna().all(axis=1).all():
            st.error('No XBRL financial data found. Please try different tickers.')
            return

        df_kpis = compute_metrics(df)

        st.subheader('Peer KPI Table')
        st.dataframe(df_kpis)

        st.subheader('AI-Generated Narrative')
        st.write(generate_narrative(df_kpis))

        buf = BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df_kpis.to_excel(w, index=False)
        buf.seek(0)
        st.sidebar.download_button('Download XLSX', buf.getvalue(), 'peer_kpis.xlsx')

if __name__ == '__main__':
    main()

