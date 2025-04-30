import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import openai
import yfinance as yf
from yfinance.exceptions import YFRatelimitError
from scipy.stats import rankdata
from io import BytesIO
from typing import List, Dict, Any

# === Constants ===
SEC_XBRL_BASE      = "https://data.sec.gov/api/xbrl/companyfacts/"
SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT         = "PeerLensBenchmarkStudio/0.1 (your_email@example.com)"

# Cache for ticker→CIK mapping
ticker_cik_map: Dict[str, str] = {}

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
    return resp.json() if resp.status_code == 200 else {}

def extract_financials(xbrl_data: Dict[str, Any], facts: List[str]) -> Dict[str, float]:
    res = {}
    usg = xbrl_data.get('facts', {}).get('us-gaap', {})
    for f in facts:
        blk = usg.get(f) or usg.get(f.lower(), {})
        units = blk.get('units', {})
        val = np.nan
        for arr in units.values():
            if arr:
                try:
                    val = float(arr[-1].get('v'))
                except:
                    val = np.nan
                break
        res[f] = val
    return res

def fallback_financials(ticker: str, facts: List[str]) -> Dict[str, float]:
    """
    Use yfinance as fallback for fundamentals, handling rate limits gracefully.
    """
    try:
        info = yf.Ticker(ticker).info
    except YFRatelimitError:
        st.warning(f"yfinance rate limit hit for {ticker}; skipping fallback.")
        return {f: np.nan for f in facts}
    except Exception as e:
        st.warning(f"yfinance error for {ticker}: {e}")
        return {f: np.nan for f in facts}
    return {
        'Revenues': info.get('totalRevenue', np.nan),
        'Ebitda': info.get('ebitda', np.nan),
        'Assets': info.get('totalAssets', np.nan),
        'Liabilities': info.get('totalLiab', np.nan),
        'MarketCapitalization': info.get('marketCap', np.nan)
    }

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ebitda_margin'] = df['Ebitda'] / df['Revenues']
    df['rev_cagr_3yr']   = (df['Revenues'] / df['Revenues'].shift(3))**(1/3) - 1
    df['ev_ebitda']      = df['MarketCapitalization'] / df['Ebitda']
    df['debt_to_asset']  = df['Liabilities'] / df['Assets']
    df['roa']            = df['Ebitda'] / df['Assets']
    for col in ['ebitda_margin','rev_cagr_3yr','ev_ebitda','debt_to_asset','roa']:
        df[f'{col}_pct'] = rankdata(df[col].fillna(0), method='average') / len(df) * 100
    return df

def generate_narrative(df: pd.DataFrame) -> str:
    api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
    if not api_key:
        return "OpenAI API key not set."
    openai.api_key = api_key
    data = df.to_json(orient='records')
    messages = [
        {"role":"system","content":"You are an expert financial analyst."},
        {"role":"user","content":"Here are peer KPIs: " + data + " Provide a 5-bullet summary."}
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.2
    )
    return response.choices[0].message.content.strip()

def main():
    st.set_page_config(page_title='PeerLens Benchmark Studio', layout='wide')
    st.title('📊 PeerLens Benchmark Studio')
    st.sidebar.markdown("### Peer Inputs")
    tickers = st.sidebar.text_input('Tickers (comma-separated)')
    csv_file = st.sidebar.file_uploader("CSV with 'Ticker' column", type='csv')

    if st.sidebar.button('Generate Benchmark'):
        # Build peer list
        if csv_file:
            df_in = pd.read_csv(csv_file)
            peers = df_in['Ticker'].astype(str).str.upper().tolist()
        else:
            peers = [t.strip().upper() for t in tickers.split(',') if t.strip()]

        if not peers:
            st.error('Please provide tickers or upload a CSV.')
            return

        st.info(f'Fetching data for {len(peers)} peers...')
        ciks = fetch_public_filer_ciks(peers)
        records = []
        facts = ['Revenues','Ebitda','Assets','Liabilities','MarketCapitalization']

        for t in peers:
            cik   = ciks.get(t)
            xbrl  = fetch_xbrl_data(cik) if cik else {}
            fin   = extract_financials(xbrl, facts)
            if all(np.isnan(v) for v in fin.values()):
                fin = fallback_financials(t, facts)
            fin['Ticker'] = t
            records.append(fin)

        df = pd.DataFrame(records)
        df_kpis = compute_metrics(df)

        st.subheader('Peer KPI Table')
        st.dataframe(df_kpis)

        st.subheader('AI-Generated Narrative')
        st.write(generate_narrative(df_kpis))

        # Excel export
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_kpis.to_excel(writer, index=False, sheet_name='KPIs')
        buf.seek(0)
        st.sidebar.download_button(
            'Download XLSX',
            data=buf.getvalue(),
            file_name='peer_kpis.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

if __name__ == '__main__':
    main()
