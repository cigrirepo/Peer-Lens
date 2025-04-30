```python
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import openai
import yfinance as yf
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
        data = resp.json() if resp.status_code == 200 else {}
        for record in data.values():
            tic = record.get('ticker', '').upper()
            cik_raw = record.get('cik_str', '')
            if tic and cik_raw:
                ticker_cik_map[tic] = str(cik_raw).zfill(10)
    return {t: ticker_cik_map.get(t) for t in tickers}


def fetch_xbrl_data(cik: str) -> Dict[str, Any]:
    """
    Retrieve XBRL JSON for a CIK.
    """
    url = f"{SEC_XBRL_BASE}CIK{cik}.json"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT})
    if resp.status_code == 200:
        return resp.json()
    return {}


def extract_financials(xbrl_data: Dict[str, Any], facts: List[str]) -> Dict[str, float]:
    """
    Extract latest US-GAAP fact values or return NaNs.
    """
    res = {}
    usg = xbrl_data.get('facts', {}).get('us-gaap', {})
    for f in facts:
        blk = usg.get(f) or usg.get(f.lower(), {})
        units = blk.get('units', {})
        val = np.nan
        for arr in units.values():
            if arr:
                v = arr[-1].get('v')
                try: val = float(v)
                except: val = np.nan
                break
        res[f] = val
    return res


def fallback_financials(ticker: str) -> Dict[str, float]:
    """
    Use yfinance as fallback for fundamentals.
    """
    info = yf.Ticker(ticker).info
    return {
        'Revenues': info.get('totalRevenue', np.nan),
        'Ebitda': info.get('ebitda', np.nan),
        'Assets': info.get('totalAssets', np.nan),
        'Liabilities': info.get('totalLiab', np.nan),
        'MarketCapitalization': info.get('marketCap', np.nan)
    }


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute KPIs and percentile ranks.
    """
    df = df.copy()
    df['ebitda_margin'] = df['Ebitda'] / df['Revenues']
    df['rev_cagr_3yr']   = (df['Revenues'] / df['Revenues'].shift(3))**(1/3) - 1
    df['ev_ebitda']      = df['MarketCapitalization'] / df['Ebitda']
    df['debt_to_asset']  = df['Liabilities'] / df['Assets']
    df['roa']            = df['Ebitda'] / df['Assets']
    for col in ['ebitda_margin','rev_cagr_3yr','ev_ebitda','debt_to_asset','roa']:
        df[f'{col}_pct'] = rankdata(df[col].fillna(0), method='average') / len(df)*100
    return df


def generate_narrative(df: pd.DataFrame) -> str:
    """
    Generate a concise summary with LLM.
    """
    key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
    if not key:
        return "API key not set"
    openai.api_key = key
    data = df.to_json(orient='records')
    msgs = [
        {'role':'system','content':'You are an expert financial analyst.'},
        {'role':'user','content':(
            'Here are peer KPIs: ' + data +
            ' Write a 5-bullet summary of valuation, growth, margins.'
        )}
    ]
    resp = openai.chat.completions.create(
        model='gpt-3.5-turbo', messages=msgs, temperature=0.2
    )
    return resp.choices[0].message.content.strip()

# === Streamlit ===
def main():
    st.set_page_config(page_title='PeerLens', layout='wide')
    st.title('📊 PeerLens Benchmark Studio')
    tickers = st.sidebar.text_input('Tickers (comma-separated)')
    file    = st.sidebar.file_uploader("CSV with 'Ticker' column", type='csv')
    if st.sidebar.button('Generate Benchmark'):
        peers = []
        if file:
            peers = pd.read_csv(file)['Ticker'].str.upper().tolist()
        else:
            peers = [t.strip().upper() for t in tickers.split(',') if t]
        if not peers:
            st.error('Provide tickers or CSV.')
            return
        st.info(f'Fetching {len(peers)} peers...')
        ciks = fetch_public_filer_ciks(peers)
        recs = []
        facts = ['Revenues','Ebitda','Assets','Liabilities','MarketCapitalization']
        for t in peers:
            cik = ciks.get(t)
            data = fetch_xbrl_data(cik) if cik else {}
            fin = extract_financials(data, facts)
            if all(np.isnan(v) for v in fin.values()):
                fin = fallback_financials(t)
            fin['Ticker'] = t
            recs.append(fin)
        df = pd.DataFrame(recs)
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

if __name__=='__main__':
    main()
```

**Changes:**
- **Fallback to yfinance** when XBRL returns no data, eliminating blanks.  
- **Fixed SEC endpoint** to prefix `CIK` + zero-padded CIK (10 digits).  
- **Improved narrative prompt** to create bullet summaries.  



