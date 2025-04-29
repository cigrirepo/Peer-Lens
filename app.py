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
SEC_XBRL_BASE = "https://data.sec.gov/api/xbrl/companyfacts/"
SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "PeerLensBenchmarkStudio/0.1 (your_email@example.com)"

# Cache for ticker→CIK mapping
ticker_cik_map: Dict[str, str] = {}

# === Helper Functions ===
def fetch_public_filer_ciks(tickers: List[str]) -> Dict[str, str]:
    """
    Given a list of tickers, fetch and cache CIKs using SEC's company_tickers.json.
    """
    global ticker_cik_map
    if not ticker_cik_map:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(SEC_TICKER_CIK_URL, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            for record in data.values():
                tic = record.get('ticker', '').upper()
                cik_str = str(record.get('cik_str', '')).lstrip('0')
                if tic and cik_str:
                    ticker_cik_map[tic] = cik_str
        else:
            st.error(f"Failed to load ticker-CIK map: HTTP {resp.status_code}")
    result: Dict[str, str] = {}
    for tic in tickers:
        cik = ticker_cik_map.get(tic)
        if cik:
            result[tic] = cik
        else:
            result[tic] = None
            st.warning(f"CIK not found for ticker '{tic}'")
    return result

def fetch_xbrl_data(cik: str) -> Dict[str, Any]:
    """
    Retrieve XBRL data for a given CIK from the SEC API.
    """
    url = f"{SEC_XBRL_BASE}{cik}.json"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    st.error(f"Error fetching XBRL for CIK {cik}: HTTP {resp.status_code}")
    return {}

def extract_financials(xbrl_data: Dict[str, Any], facts: List[str]) -> Dict[str, float]:
    """
    Extract latest reported values for a list of US-GAAP facts from XBRL JSON.
    """
    results: Dict[str, float] = {}
    facts_dict = xbrl_data.get('facts', {}).get('us-gaap', {})
    for fact in facts:
        key = fact.lower() if fact.lower() in facts_dict else fact
        item = facts_dict.get(key, {})
        units = item.get('units', {})
        value = np.nan
        for unit_vals in units.values():
            if isinstance(unit_vals, list) and unit_vals:
                last = unit_vals[-1].get('v')
                try:
                    value = float(last)
                except:
                    value = np.nan
                break
        results[fact] = value
    return results

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate KPI columns and percentile ranks on the raw financials.
    """
    df = df.copy()
    # Profitability & Growth Metrics
    df['ebitda_margin'] = df['Ebitda'] / df['Revenues']
    df['rev_cagr_3yr'] = (df['Revenues'] / df['Revenues'].shift(3)) ** (1/3) - 1
    # Valuation Metric
    df['ev_ebitda'] = df['MarketCapitalization'] / df['Ebitda']
    # Leverage & Efficiency
    df['debt_to_asset'] = df['Liabilities'] / df['Assets']
    df['roa'] = df['Ebitda'] / df['Assets']
    # Percentile ranks
    for col in ['ebitda_margin', 'rev_cagr_3yr', 'ev_ebitda', 'debt_to_asset', 'roa']:
        df[f'{col}_pct'] = rankdata(df[col].fillna(0), method='average') / len(df) * 100
    return df

def generate_narrative(df: pd.DataFrame) -> str:
    """
    Generate a 120-word narrative using OpenAI GPT based on the KPI DataFrame.
    """
    api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
    if not api_key:
        st.error('OpenAI API key not configured.')
        return ''
    openai.api_key = api_key

    data_json = df.to_json(orient='records')
    system_msg = (
        'You are a financial research analyst. '
        'Given a JSON array of company KPI records, write a concise 120-word summary '
        'highlighting valuation multiples, growth rates, and margin comparisons.'
    )
    user_msg = f"Here are the peer KPIs: {data_json}"

    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role':'system','content':system_msg}, {'role':'user','content':user_msg}],
            temperature=0.2,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI request failed: {e}")
        return ''

# === Streamlit App ===
def main():
    st.set_page_config(page_title="PeerLens Benchmark Studio", layout="wide")
    st.title("📊 PeerLens Benchmark Studio")
    st.markdown(
        "Upload peer tickers or a CSV of peers to benchmark key financial metrics and generate an AI narrative."
    )

    st.sidebar.header("Peer Inputs")
    tickers_input = st.sidebar.text_input("Comma-separated tickers, e.g. AAPL,MSFT")
    upload_csv = st.sidebar.file_uploader("Or upload CSV with a 'Ticker' column", type=['csv'])

    if st.sidebar.button("Generate Benchmark"):
        if upload_csv:
            df_peers = pd.read_csv(upload_csv)
            peers = df_peers['Ticker'].astype(str).str.upper().tolist()
        else:
            peers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

        if not peers:
            st.error("Please provide at least one ticker or upload a CSV.")
            return

        st.info(f"Fetching data for {len(peers)} peers...")
        cik_map = fetch_public_filer_ciks(peers)

        records: List[Dict[str, Any]] = []
        facts = ['Revenues', 'Ebitda', 'Assets', 'Liabilities', 'MarketCapitalization']
        for ticker in peers:
            cik = cik_map.get(ticker)
            if cik:
                xbrl = fetch_xbrl_data(cik)
                fin = extract_financials(xbrl, facts)
                fin['Ticker'] = ticker
                records.append(fin)
        df_fin = pd.DataFrame(records)

        df_kpis = compute_metrics(df_fin)
        st.subheader("Peer KPI Table")
        st.dataframe(df_kpis)

        narrative = generate_narrative(df_kpis)
        st.subheader("AI-Generated Narrative")
        st.write(narrative)

        # Excel export
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_kpis.to_excel(writer, index=False, sheet_name="KPIs")
        output.seek(0)
        st.sidebar.download_button(
            "Download XLSX",
            data=output.getvalue(),
            file_name="peer_kpis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        # TODO: implement PPTX export via python-pptx

if __name__ == "__main__":
    main()
