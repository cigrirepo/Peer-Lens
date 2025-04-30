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

FACT_TAGS = {
    'Revenues': 'Revenues',
    'EBITDA': 'EarningsBeforeInterestTaxesDepreciationAndAmortization',
    'Assets': 'Assets',
    'Liabilities': 'Liabilities',
    'MarketCap': 'MarketCapitalization'
}

# === Helper Functions ===
def fetch_public_filer_ciks(tickers: List[str]) -> Dict[str, str]:
    resp = requests.get(SEC_TICKER_CIK_URL, headers={"User-Agent": USER_AGENT})
    mapping = {}
    if resp.status_code == 200:
        data = resp.json()
        for rec in data.values():
            t = rec.get('ticker','').upper()
            cik = str(rec.get('cik_str','')).zfill(10)
            mapping[t] = cik
    return {t: mapping.get(t) for t in tickers}


def fetch_xbrl_financials(ciks: Dict[str,str]) -> List[Dict[str,Any]]:
    results = []
    for t, cik in ciks.items():
        row = {'Ticker':t}
        if cik:
            url = f"{SEC_XBRL_BASE}CIK{cik}.json"
            j = requests.get(url, headers={"User-Agent":USER_AGENT}).json()
            usg = j.get('facts',{}).get('us-gaap',{})
            for label, tag in FACT_TAGS.items():
                blk = usg.get(tag, {})
                val = np.nan
                for arr in blk.get('units',{}).values():
                    if arr:
                        try: val = float(arr[-1].get('v'))
                        except: val=np.nan
                        break
                row[label] = val
        results.append(row)
    return results


def fetch_yf_financials(peers: List[str]) -> List[Dict[str,Any]]:
    ticker_str = ' '.join(peers)
    multi = yf.Tickers(ticker_str)
    results = []
    for t in peers:
        info = multi.tickers.get(t, {}).info
        results.append({
            'Ticker': t,
            'Revenues': info.get('totalRevenue', np.nan),
            'EBITDA': info.get('ebitda', np.nan),
            'Assets': info.get('totalAssets', np.nan),
            'Liabilities': info.get('totalLiab', np.nan),
            'MarketCap': info.get('marketCap', np.nan)
        })
    return results


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EBITDA Margin'] = df['EBITDA'] / df['Revenues']
    df['Revenue CAGR (3y)'] = (df['Revenues'] / df['Revenues'].shift(3))**(1/3) - 1
    for col in ['EBITDA Margin','Revenue CAGR (3y)']:
        df[f'{col} pct'] = rankdata(df[col].fillna(0), method='average')/len(df)*100
    return df


def generate_narrative(df: pd.DataFrame) -> str:
    key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
    if not key:
        return "API key not set."
    openai.api_key = key
    data = df.to_dict('records')
    content = json.dumps(data)
    msgs = [
        {'role':'system','content':'You are a financial analyst.'},
        {'role':'user','content':'Summarize these peer KPIs in 5 bullets: ' + content}
    ]
    res = openai.chat.completions.create(
        model='gpt-3.5-turbo', messages=msgs, temperature=0.2
    )
    return res.choices[0].message.content.strip()


def main():
    st.set_page_config(page_title='PeerLens', layout='wide')
    st.title('📊 PeerLens Benchmark Studio')
    tickers_input = st.sidebar.text_input('Tickers (comma-separated)')
    if st.sidebar.button('Generate Benchmark'):
        peers = [t.strip().upper() for t in tickers_input.split(',') if t]
        if not peers:
            st.error('Provide at least one ticker.')
            return
        st.info(f'Fetching XBRL data for {len(peers)} peers...')
        ciks = fetch_public_filer_ciks(peers)
        xbrl_data = fetch_xbrl_financials(ciks)
        df_xbrl = pd.DataFrame(xbrl_data)
        if df_xbrl[['Revenues','EBITDA']].isna().all(axis=1).all():
            st.warning('Falling back to yfinance for fundamentals...')
            yf_data = fetch_yf_financials(peers)
            df = pd.DataFrame(yf_data)
        else:
            df = df_xbrl.rename(columns={'MarketCap':'MarketCapitalization'})
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

