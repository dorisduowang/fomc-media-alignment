"""
Market-dictionary tone of the FOMC minutes and its validation against the
high-frequency monetary policy surprise.

The tone of the Committee's own language is measured with the Loughran and
McDonald (2011) finance master dictionary rather than with a hand-assembled word
list. The master dictionary was constructed from the vocabulary of corporate
filings and has been validated against market outcomes across the subsequent
textual-finance literature, so it supplies category word lists -- negative,
positive, and uncertainty -- whose membership was fixed externally and before
this study. For every FOMC minute the script counts the words that fall in each
category, expresses the counts per thousand tokens, and forms a net tone as the
difference between positive and negative intensity. It then relates that
text-based tone to the Bauer and Swanson (2023) high-frequency policy surprise,
which is the market's own contemporaneous reading of the same meeting, so that
the lexical measure is anchored to a market signal rather than to a second
dictionary.

The minutes are retrieved once from the Federal Reserve's public archive and
cached locally; subsequent runs read the cache.

Outputs:
    results/tables/sentiment_comparison.csv   per-meeting LM category intensities
    results/tables/sentiment_agreement.csv    tone-surprise validation summary
    data_reference/fomc_minutes_raw.csv       cached minutes text (88 meetings)
"""

from pathlib import Path
import re
import html as _html
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
DATA = ROOT / "data_reference"
TAB.mkdir(parents=True, exist_ok=True)

LM_MASTER = Path(
    "/opt/anaconda3/lib/python3.13/site-packages/pysentiment2/static/LM.csv"
)
MINUTES_CACHE = DATA / "fomc_minutes_raw.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; FOMC minutes tone study)"}

# The 88 regularly scheduled meetings, 2015-2025, keyed by the date the Federal
# Reserve uses in the minutes filename (the closing day of the meeting).
MEETING_DATES = [
    "20150128", "20150318", "20150429", "20150617", "20150729", "20150917",
    "20151028", "20151216", "20160127", "20160316", "20160427", "20160615",
    "20160727", "20160921", "20161102", "20161214", "20170201", "20170315",
    "20170503", "20170614", "20170726", "20170920", "20171101", "20171213",
    "20180131", "20180321", "20180502", "20180613", "20180801", "20180926",
    "20181108", "20181219", "20190130", "20190320", "20190501", "20190619",
    "20190731", "20190918", "20191030", "20191211", "20200129", "20200315",
    "20200429", "20200610", "20200729", "20200916", "20201105", "20201216",
    "20210127", "20210317", "20210428", "20210616", "20210728", "20210922",
    "20211103", "20211215", "20220126", "20220316", "20220504", "20220615",
    "20220727", "20220921", "20221102", "20221214", "20230201", "20230322",
    "20230503", "20230614", "20230726", "20230920", "20231101", "20231213",
    "20240131", "20240320", "20240501", "20240612", "20240731", "20240918",
    "20241107", "20241218", "20250129", "20250319", "20250507", "20250618",
    "20250730", "20250917", "20251029", "20251210",
]


def fetch_minute(date):
    """Return the paragraph text of one minute from the Federal Reserve site."""
    url = f"https://www.federalreserve.gov/monetarypolicy/fomcminutes{date}.htm"
    req = urllib.request.Request(url, headers=HEADERS)
    html = urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "ignore")
    start = html.find('id="article"')
    seg = html[start:] if start >= 0 else html
    for end in ("Last Update", 'id="footer', "Back to Top"):
        cut = seg.find(end)
        if cut > 0:
            seg = seg[:cut]
    paras = re.findall(r"<p[^>]*>(.*?)</p>", seg, flags=re.S)
    text = _html.unescape(re.sub(r"<[^>]+>", " ", " ".join(paras)))
    return re.sub(r"\s+", " ", text).strip()


def load_minutes():
    """Load the cached minutes, fetching any that are missing."""
    if MINUTES_CACHE.exists():
        cached = pd.read_csv(MINUTES_CACHE, dtype={"date": str})
    else:
        cached = pd.DataFrame(columns=["date", "text"])
    have = set(cached["date"])
    rows = []
    for d in MEETING_DATES:
        if d in have:
            continue
        rows.append({"date": d, "text": fetch_minute(d)})
        time.sleep(0.5)
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached = cached.sort_values("date").reset_index(drop=True)
        cached.to_csv(MINUTES_CACHE, index=False)
    return cached


def lm_category_sets():
    """Return the Loughran-McDonald category word sets (lower-cased)."""
    lm = pd.read_csv(LM_MASTER)
    lm["w"] = lm["Word"].str.lower()
    sets = {}
    for cat in ("Negative", "Positive", "Uncertainty", "Constraining"):
        sets[cat] = set(lm.loc[lm[cat] != 0, "w"])
    return sets


_TOKEN = re.compile(r"[a-z]+")


def score_minute(text, sets):
    toks = _TOKEN.findall(text.lower())
    n = len(toks)
    if n == 0:
        return {k: 0.0 for k in ("neg", "pos", "unc", "con", "net", "n_tokens")}
    counts = {c: sum(1 for t in toks if t in s) for c, s in sets.items()}
    per1k = lambda x: 1000.0 * x / n
    return {
        "neg": per1k(counts["Negative"]),
        "pos": per1k(counts["Positive"]),
        "unc": per1k(counts["Uncertainty"]),
        "con": per1k(counts["Constraining"]),
        "net": per1k(counts["Positive"] - counts["Negative"]),
        "n_tokens": n,
    }


def main():
    minutes = load_minutes()
    minutes["meeting_date"] = pd.to_datetime(minutes["date"], format="%Y%m%d")
    sets = lm_category_sets()

    scored = minutes.join(
        minutes["text"].apply(lambda t: pd.Series(score_minute(t, sets)))
    )
    scored = scored.rename(
        columns={
            "neg": "lm_neg_per_1k",
            "pos": "lm_pos_per_1k",
            "unc": "lm_unc_per_1k",
            "con": "lm_con_per_1k",
            "net": "lm_net_per_1k",
        }
    )

    mps = pd.read_csv(DATA / "market" / "mps_fomc.csv")
    mps["Date"] = pd.to_datetime(mps["Date"])
    mps = mps.dropna(subset=["MPS"]).sort_values("Date")

    def nearest_surprise(meeting):
        diff = (mps["Date"] - meeting).abs()
        j = diff.idxmin()
        if diff.loc[j] <= pd.Timedelta(days=1):
            return mps.loc[j, "MPS"]
        return np.nan

    scored["mps"] = scored["meeting_date"].map(nearest_surprise)
    scored["abs_mps"] = scored["mps"].abs()

    # Realized S&P 500 volatility over the five trading days after the meeting
    # and the last VIX close before it, so the text tone can be read against the
    # market's own gauges of the same window.
    spx = pd.read_csv(DATA / "market" / "SPX_daily_2014_2025.csv")
    spx["date"] = pd.to_datetime(spx["date"])
    spx = spx.dropna(subset=["spx"]).sort_values("date").reset_index(drop=True)
    spx["logret"] = np.log(spx["spx"]).diff()
    rets = spx.dropna(subset=["logret"]).reset_index(drop=True)

    def realized_vol_post(meeting, w=5):
        sub = rets[rets["date"] > meeting]["logret"].iloc[:w]
        return float(sub.std(ddof=1) * np.sqrt(252)) if len(sub) >= 2 else np.nan

    vix = pd.read_csv(DATA / "market" / "VIX_2014_2025.csv")
    vix.columns = ["date", "vix"]
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.dropna(subset=["vix"]).sort_values("date")

    def pre_vix(meeting):
        prior = vix[vix["date"] < meeting]["vix"]
        return prior.iloc[-1] if len(prior) else np.nan

    scored["vol_post"] = scored["meeting_date"].map(realized_vol_post)
    scored["vix_pre"] = scored["meeting_date"].map(pre_vix)

    out = scored[
        [
            "meeting_date",
            "n_tokens",
            "lm_neg_per_1k",
            "lm_pos_per_1k",
            "lm_unc_per_1k",
            "lm_con_per_1k",
            "lm_net_per_1k",
            "mps",
            "abs_mps",
            "vol_post",
            "vix_pre",
        ]
    ].sort_values("meeting_date")
    out.to_csv(TAB / "sentiment_comparison.csv", index=False)

    pairs = [
        ("negative_tone_vs_pre_vix", "lm_neg_per_1k", "vix_pre"),
        ("negative_tone_vs_post_vol", "lm_neg_per_1k", "vol_post"),
        ("net_tone_vs_post_vol", "lm_net_per_1k", "vol_post"),
        ("negative_vs_abs_surprise", "lm_neg_per_1k", "abs_mps"),
        ("net_tone_vs_signed_surprise", "lm_net_per_1k", "mps"),
    ]
    recs = []
    for name, a, b in pairs:
        d = out.dropna(subset=[a, b])
        r_p, p_p = pearsonr(d[a], d[b])
        r_s, p_s = spearmanr(d[a], d[b])
        recs.append(
            {
                "relation": name,
                "n_meetings": len(d),
                "pearson_r": r_p,
                "pearson_p": p_p,
                "spearman_rho": r_s,
                "spearman_p": p_s,
                "first_meeting": d["meeting_date"].min().date(),
                "last_meeting": d["meeting_date"].max().date(),
            }
        )
    agree = pd.DataFrame(recs)
    agree.to_csv(TAB / "sentiment_agreement.csv", index=False)
    pd.set_option("display.width", 160)
    print(agree.to_string(index=False))


if __name__ == "__main__":
    main()
