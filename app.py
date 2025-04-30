import requests

SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_XBRL_BASE      = "https://data.sec.gov/api/xbrl/companyfacts/"

def fetch_cik(ticker: str) -> str:
    resp = requests.get(SEC_TICKER_CIK_URL, headers={"User-Agent":"demo"})
    resp.raise_for_status()
    data = resp.json()
    for rec in data.values():
        if rec.get("ticker","").upper() == ticker:
            return rec["cik_str"].zfill(10)
    raise ValueError(f"Ticker {ticker} not found")

def fetch_latest_revenue(cik10: str) -> float:
    url = f"{SEC_XBRL_BASE}CIK{cik10}.json"
    resp = requests.get(url, headers={"User-Agent":"demo"})
    resp.raise_for_status()
    facts = resp.json()["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    return facts[-1]["v"]

if __name__=="__main__":
    cik = fetch_cik("AAPL")
    print("AAPL CIK10:", cik)                # expect 0000320193
    rev = fetch_latest_revenue(cik)
    print("AAPL latest Revenues:", rev)
