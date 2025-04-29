import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import rankdata
from typing import List, Dict, Any

# === Constants ===
SEC_XBRL_BASE = "https://data.sec.gov/api/xbrl/companyfacts/"
USER_AGENT = "PeerLensBenchmarkStudio/0.1 (your_email@example.com)"

# === Helper Functions ===
def fetch_public_filer_ciks(tickers: List[str]) -> Dict[str, str]:
    """
    Given a list of tickers, fetch CIKs using SEC Company Tickers endpoint.
    """
    # TODO: implement mapping via SEC or third-party service
    return {ticker: None for ticker in tickers}


def fetch_xbrl_data(cik: str) -> Dict[str, Any]:
    """
    Retrieve XBRL data for a given CIK from the SEC API.
    """
    url = f"{SEC_XBRL_BASE}{cik}.json"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error(f"Error fetching XBRL for CIK {cik}: HTTP {resp.status_code}")
        return {}


def extract_financials(xbrl_data: Dict[str, Any], facts: List[str]) -> Dict[str, float]:
    """
    Extract latest values for requested fact names (e.g., 'Revenues', 'Ebitda').
    """
    results = {}
    # TODO: parse JSON structure to get LTM, TTMs, etc.
    return results


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of peer financials, calculate KPIs and percentiles.
    """
    # Example: EBITDA margin, Revenue CAGR
    df['ebitda_margin'] = df['Ebitda'] / df['Revenues']
    df['rev_cagr_3yr'] = (df['Revenues'] / df['Revenues'].shift(3)) ** (1/3) - 1
    # Compute percentile ranks
    for col in ['ebitda_margin', 'rev_cagr_3yr', 'ev_ebitda']:
        df[f'{col}_pct'] = rankdata(df[col], method='average') / len(df) * 100
    return df


def generate_narrative(df: pd.DataFrame) -> str:
    """
    Call OpenAI API to generate a 120-word narrative based on KPI table.
    """
    # TODO: integrate OpenAI GPT prompt with df.to_json()
    narrative = "[AI Narrative Placeholder]"
    return narrative

# === Streamlit App ===

def main():
    st.set_page_config(page_title="PeerLens Benchmark Studio", layout="wide")
    st.title("📊 PeerLens Benchmark Studio")
    st.markdown("Upload a list of peer tickers or CSV to benchmark key financial metrics and generate an AI narrative.")

    st.sidebar.header("Peer Inputs")
    tickers_input = st.sidebar.text_input("Enter tickers (comma-separated)")
    upload_csv = st.sidebar.file_uploader("Or upload CSV with column 'Ticker'", type=['csv'])
    generate_btn = st.sidebar.button("Generate Benchmark")

    peers = []
    if generate_btn:
        if upload_csv is not None:
            df_peers = pd.read_csv(upload_csv)
            peers = df_peers['Ticker'].dropna().unique().tolist()
        elif tickers_input:
            peers = [t.strip().upper() for t in tickers_input.split(',')]

        if peers:
            st.info(f"Fetching data for {len(peers)} peers...")
            # 1. Map tickers to CIKs
            cik_map = fetch_public_filer_ciks(peers)
            # 2. Fetch XBRL and extract financials
            records = []
            facts = ['Revenues', 'Ebitda', 'Assets', 'Liabilities', 'MarketCapitalization']
            for ticker in peers:
                cik = cik_map.get(ticker)
                if cik:
                    xbrl = fetch_xbrl_data(cik)
                    fin = extract_financials(xbrl, facts)
                    fin['Ticker'] = ticker
                    records.append(fin)
            df_fin = pd.DataFrame(records)

            # 3. Compute metrics & percentiles
            df_kpis = compute_metrics(df_fin)
            st.dataframe(df_kpis)

            # 4. Narrative
            narrative = generate_narrative(df_kpis)
            st.markdown("### AI-Generated Narrative")
            st.write(narrative)

            # 5. Export buttons
            st.sidebar.download_button("Download XLSX", data=df_kpis.to_excel(index=False), file_name="peer_kpis.xlsx")
            # TODO: Download PPTX implementation
        else:
            st.error("Please enter tickers or upload a CSV.")

if __name__ == "__main__":
    main()
