import os
import json
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import requests
import feedparser

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

NEWS_LOOKBACK_HOURS = int(os.getenv("NEWS_LOOKBACK_HOURS", "36"))
NEWS_TTL_HOURS = int(os.getenv("NEWS_TTL_HOURS", "72"))
NEWS_STATE_PATH = os.getenv("NEWS_STATE_PATH", "state/news_state.json")
FEEDS_PATH = os.getenv("NEWS_FEEDS_PATH", "config/news_feeds.json")
USER_AGENT = os.getenv("NEWS_USER_AGENT", "telegram-news-bot/1.0 (+GitHub Actions)")

REGION_TOTAL = {
    "de": int(os.getenv("NEWS_DE_TOTAL", "6")),
    "us": int(os.getenv("NEWS_US_TOTAL", "6")),
    "eu": int(os.getenv("NEWS_EU_TOTAL", "3")),
    "world": int(os.getenv("NEWS_WORLD_TOTAL", "4")),
}
REGION_ECON_SHARE = {
    "de": float(os.getenv("NEWS_ECON_SHARE_DE", "0.67")),
    "us": float(os.getenv("NEWS_ECON_SHARE_US", "0.67")),
    "eu": float(os.getenv("NEWS_ECON_SHARE_EU", "0.67")),
    "world": float(os.getenv("NEWS_ECON_SHARE_WORLD", "0.5")),
}

ECON_KEYWORDS = [
    "inflation","zins","zinsen","ezb","fed","gdp","bip","recession","rezession",
    "arbeitsmarkt","unemployment","zoll","tariff","oil","öl","gas","energy","energie",
    "bank","banken","debt","schulden","budget","haushalt","bond","anleihe",
    "earnings","quartal","guidance","merger","übernahme","ipo","insolvenz",
    "market","märkte","stocks","aktien","dollar","eur","euro"
]

@dataclass
class Item:
    region: str
    source: str
    title: str
    url: str
    published_ts: Optional[int]
    summary: str
    weight: float

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

def quality_score(item: Item) -> float:
    s = econ_score(f"{item.title} {item.summary}")
    if item.published_ts:
        age_h = max(0.0, (now_ts() - item.published_ts) / 3600.0)
        s += max(0.0, 2.0 - (age_h / 12.0))
    s *= max(0.5, min(2.0, item.weight))
    return s

def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

def build_digest(groups: Dict[str, List[Item]]) -> str:
    berlin = datetime.now(timezone.utc) + timedelta(hours=1)
    header = f"News Briefing – {berlin.strftime('%d.%m.%Y %H:%M')} (DE Zeit)\n"

    titles = {"de": "Deutschland", "us": "USA", "eu": "Europa", "world": "Welt"}
    order = ["de", "us", "eu", "world"]

    out = [header]
    for region in order:
        out.append(f"\n{titles[region]}")
        if not groups.get(region):
            out.append("- (keine Headlines gefunden)")
            continue
        for i, it in enumerate(groups[region], 1):
            out.append(f"{i}. {it.title} ({it.source})\n{it.url}")
    return "\n".join(out)[:4000]

def main() -> None:
    feeds = load_json(FEEDS_PATH, {})
    state = load_json(NEWS_STATE_PATH, {"last_digest_ts": 0, "seen": {}})

    last = int(state.get("last_digest_ts", 0))
    if last and now_ts() - last < 2 * 3600 - 30:
        print("Skip: already sent within 2 hours.")
        return

    seen: Dict[str, int] = state.get("seen", {})
    items: List[Item] = []

    headers = {"User-Agent": USER_AGENT}

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

            for entry in parsed.entries[:40]:
                title = (getattr(entry, "title", "") or "").strip()
                link = (getattr(entry, "link", "") or "").strip()
                summary = (getattr(entry, "summary", "") or "").strip()
                if not title or not link:
                    continue

                pub_ts = parse_entry_time(entry)
                if not is_recent(pub_ts):
                    continue

                h = sha1(link)
                if h in seen:
                    continue

                items.append(Item(region=region, source=name, title=title, url=link,
                                  published_ts=pub_ts, summary=summary, weight=weight))

    scored: List[Tuple[float, Item]] = [(quality_score(it), it) for it in items]
    scored.sort(key=lambda x: x[0], reverse=True)

    econ_bucket = {k: [] for k in REGION_TOTAL}
    norm_bucket = {k: [] for k in REGION_TOTAL}

    for score, it in scored:
        (econ_bucket if is_econ(it) else norm_bucket)[it.region].append(it)

    groups: Dict[str, List[Item]] = {k: [] for k in REGION_TOTAL}
    for region, total in REGION_TOTAL.items():
        econ_target = int(round(total * REGION_ECON_SHARE[region]))
        sel = econ_bucket[region][:econ_target]
        sel += norm_bucket[region][:max(0, total - len(sel))]
        if len(sel) < total:
            need = total - len(sel)
            sel += econ_bucket[region][len(sel):len(sel) + need]
        groups[region] = sel[:total]

    if not any(groups.values()):
        print("No items selected.")
        return

    text = build_digest(groups)
    send_telegram(text)

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
