import os
import json
import time
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import requests
import feedparser

# -----------------------
# ENV
# -----------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

NEWS_LOOKBACK_HOURS = int(os.getenv("NEWS_LOOKBACK_HOURS", "36"))
NEWS_TTL_HOURS = int(os.getenv("NEWS_TTL_HOURS", "72"))
NEWS_STATE_PATH = os.getenv("NEWS_STATE_PATH", "state/news_state.json")
FEEDS_PATH = os.getenv("NEWS_FEEDS_PATH", "config/news_feeds.json")
USER_AGENT = os.getenv("NEWS_USER_AGENT", "telegram-news-bot/1.1 (+GitHub Actions)")

# Region-Fokus (DE/US priorisiert)
REGION_TOTAL = {
    "de": int(os.getenv("NEWS_DE_TOTAL", "6")),
    "us": int(os.getenv("NEWS_US_TOTAL", "6")),
    "eu": int(os.getenv("NEWS_EU_TOTAL", "3")),
    "world": int(os.getenv("NEWS_WORLD_TOTAL", "4")),
}

# Anteil wirtschaftlich pro Region (Rest = "normale" News)
REGION_ECON_SHARE = {
    "de": float(os.getenv("NEWS_ECON_SHARE_DE", "0.67")),
    "us": float(os.getenv("NEWS_ECON_SHARE_US", "0.67")),
    "eu": float(os.getenv("NEWS_ECON_SHARE_EU", "0.67")),
    "world": float(os.getenv("NEWS_ECON_SHARE_WORLD", "0.50")),
}

# Minimaler Score, um Noise zu dämpfen (insb. World)
MIN_SCORE = {
    "de": float(os.getenv("NEWS_MIN_SCORE_DE", "0.8")),
    "us": float(os.getenv("NEWS_MIN_SCORE_US", "0.8")),
    "eu": float(os.getenv("NEWS_MIN_SCORE_EU", "0.8")),
    "world": float(os.getenv("NEWS_MIN_SCORE_WORLD", "1.0")),
}

# Wenn ein Feed keine Zeit liefert, akzeptieren wir ihn dennoch, aber Recency-Bonus entfällt.
ECON_KEYWORDS = [
    # Makro / Zentralbanken
    "inflation", "cpi", "ppi", "zins", "zinsen", "interest rate", "rate hike", "rate cut",
    "ezb", "ecb", "fed", "fomc", "bip", "gdp", "recession", "rezession",
    "arbeitsmarkt", "unemployment", "jobs report", "payrolls",

    # Handel / Rohstoffe / Energie
    "zoll", "tariff", "trade", "exports", "imports",
    "oil", "crude", "brent", "wti", "öl", "gas", "lng", "energy", "energie", "power",

    # Finanzen / Unternehmen
    "bank", "banken", "debt", "schulden", "bond", "anleihe", "yield", "treasury",
    "earnings", "results", "quarter", "quartal", "guidance",
    "merger", "acquisition", "übernahme", "ipo", "bankruptcy", "insolvenz",

    # Märkte / Währungen
    "market", "markets", "märkte", "stocks", "shares", "aktien", "index", "dax", "s&p",
    "dollar", "usd", "eur", "euro"
]

# -----------------------
# Data model
# -----------------------
@dataclass
class Item:
    region: str
    source: str
    title: str
    url: str
    published_ts: Optional[int]
    summary: str
    weight: float

# -----------------------
# Helpers
# -----------------------
def now_ts() -> int:
    return int(time.time())

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def parse_entry_time(entry) -> Optional[int]:
    t = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not t:
        return None
    return int(time.mktime(t))

def is_recent(ts: Optional[int]) -> bool:
    if ts is None:
        return True
    return ts >= now_ts() - NEWS_LOOKBACK_HOURS * 3600

def econ_score(text: str) -> float:
    t = (text or "").lower()
    return sum(1.0 for kw in ECON_KEYWORDS if kw in t)

def is_econ(item: Item) -> bool:
    return econ_score(f"{item.title} {item.summary}") >= 1.0

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s

def normalize_url(url: str) -> str:
    """
    Normalisiert URLs, damit Tracking-Parameter nicht zu Duplikaten führen.
    Entfernt typische Tracking-Query-Parameter.
    """
    try:
        parts = urlsplit(url)
        query = parse_qsl(parts.query, keep_blank_values=True)

        drop_prefixes = ("utm_",)
        drop_keys = {"ref", "referrer", "cmpid", "cmp", "ocid", "smid", "mc_cid", "mc_eid"}

        new_query = []
        for k, v in query:
            lk = k.lower()
            if any(lk.startswith(p) for p in drop_prefixes):
                continue
            if lk in drop_keys:
                continue
            new_query.append((k, v))

        query_str = urlencode(new_query, doseq=True)
        normalized = urlunsplit((parts.scheme, parts.netloc, parts.path, query_str, ""))
        return normalized.rstrip("/")
    except Exception:
        return url

def quality_score(item: Item) -> float:
    # Basisscore: Wirtschaftsrelevanz
    s = econ_score(f"{item.title} {item.summary}")

    # Recency Bonus
    if item.published_ts:
        age_h = max(0.0, (now_ts() - item.published_ts) / 3600.0)
        # starke Gewichtung bis 12h, danach abfallend
        s += max(0.0, 2.0 - (age_h / 12.0))

    # Quellengewicht
    s *= max(0.5, min(2.0, item.weight))
    return s

def get_berlin_time_str() -> str:
    # Python 3.11+ mit zoneinfo
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.now(ZoneInfo("Europe/Berlin"))
        return dt.strftime("%d.%m.%Y %H:%M")
    except Exception:
        # Fallback, falls ZoneInfo nicht verfügbar (sollte in Actions aber gehen)
        dt = datetime.now(timezone.utc) + timedelta(hours=1)
        return dt.strftime("%d.%m.%Y %H:%M")

def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

def build_digest(groups: Dict[str, List[Item]]) -> str:
    header = f"News Briefing – {get_berlin_time_str()} (DE Zeit)\n"

    titles = {"de": "Deutschland", "us": "USA", "eu": "Europa", "world": "Welt"}
    order = ["de", "us", "eu", "world"]  # DE/US zuerst

    out: List[str] = [header]
    for region in order:
        out.append(f"\n{titles[region]}")
        region_items = groups.get(region, [])
        if not region_items:
            out.append("- (keine Headlines gefunden)")
            continue

        for i, it in enumerate(region_items, 1):
            title = clean_text(it.title)
            out.append(f"{i}. {title} ({it.source})\n{it.url}")

    text = "\n".join(out)

    # Telegram ~4096 chars; wir lassen Luft.
    if len(text) > 3900:
        text = text[:3900].rstrip() + "\n\n…(gekürzt)"
    return text

# -----------------------
# Main
# -----------------------
def main() -> None:
    feeds = load_json(FEEDS_PATH, {})
    state = load_json(NEWS_STATE_PATH, {"last_digest_ts": 0, "seen": {}})

    # Timing: nur alle 2 Stunden senden (robust gegen Actions-Runs)
    last = int(state.get("last_digest_ts", 0))
    if last and now_ts() - last < 2 * 3600 - 30:
        print("Skip: already sent within 2 hours.")
        return

    seen: Dict[str, int] = state.get("seen", {})
    items: List[Item] = []

    headers = {"User-Agent": USER_AGENT}

    # 1) Collect
    for region, flist in feeds.items():
        if region not in REGION_TOTAL:
            continue
        for f in flist:
            name = f.get("name", "Source")
            url = f.get("url")
            weight = float(f.get("weight", 1.0))
            if not url:
                continue

            try:
                resp = requests.get(url, headers=headers, timeout=25)
                resp.raise_for_status()
                parsed = feedparser.parse(resp.text)
            except Exception as e:
                print(f"Feed error [{region}] {name}: {e}")
                continue

            for entry in parsed.entries[:50]:
                title = clean_text(getattr(entry, "title", "") or "")
                link = clean_text(getattr(entry, "link", "") or "")
                summary = clean_text(getattr(entry, "summary", "") or "")

                if not title or not link:
                    continue

                pub_ts = parse_entry_time(entry)
                if not is_recent(pub_ts):
                    continue

                nurl = normalize_url(link)
                h = sha1(nurl)
                if h in seen:
                    continue

                items.append(Item(
                    region=region,
                    source=name,
                    title=title,
                    url=nurl,
                    published_ts=pub_ts,
                    summary=summary,
                    weight=weight
                ))

    if not items:
        print("No new items collected (all seen or feeds empty).")
        return

    # 2) Score
    scored: List[Tuple[float, Item]] = [(quality_score(it), it) for it in items]
    scored.sort(key=lambda x: x[0], reverse=True)

    # 3) Buckets: econ vs normal
    econ_bucket = {k: [] for k in REGION_TOTAL}
    norm_bucket = {k: [] for k in REGION_TOTAL}

    for score, it in scored:
        if score < MIN_SCORE.get(it.region, 0.0):
            continue
        (econ_bucket if is_econ(it) else norm_bucket)[it.region].append(it)

    # 4) Select per region: fill econ target, then normals, then fallback
    groups: Dict[str, List[Item]] = {k: [] for k in REGION_TOTAL}

    for region, total in REGION_TOTAL.items():
        econ_target = int(round(total * REGION_ECON_SHARE[region]))

        sel: List[Item] = []
        sel.extend(econ_bucket[region][:econ_target])

        remaining = total - len(sel)
        if remaining > 0:
            sel.extend(norm_bucket[region][:remaining])

        # Fallback: wenn nicht genug "normal", nimm weitere econ (oder umgekehrt)
        if len(sel) < total:
            need = total - len(sel)
            # zuerst restliche econ, dann restliche normal (je nachdem, was übrig ist)
            more_econ = econ_bucket[region][len(sel):len(sel) + need]
            sel.extend(more_econ)
        if len(sel) < total:
            need = total - len(sel)
            more_norm = norm_bucket[region][len(sel):len(sel) + need]
            sel.extend(more_norm)

        groups[region] = sel[:total]

    if not any(groups.values()):
        print("No items selected after scoring/filtering.")
        return

    # 5) Send
    text = build_digest(groups)
    send_telegram(text)

    # 6) Update state: mark seen + TTL cleanup
    ts = now_ts()
    for region_items in groups.values():
        for it in region_items:
            seen[sha1(it.url)] = ts

    ttl = NEWS_TTL_HOURS * 3600
    seen = {k: v for k, v in seen.items() if (ts - int(v)) <= ttl}

    state["last_digest_ts"] = ts
    state["seen"] = seen
    save_json(NEWS_STATE_PATH, state)

    print("Digest sent + state updated.")

if __name__ == "__main__":
    main()
