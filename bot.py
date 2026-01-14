import os
import json
import time
import hashlib
import re
import html
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import requests
import feedparser

# -----------------------
# ENV / CONFIG
# -----------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

NEWS_LOOKBACK_HOURS = int(os.getenv("NEWS_LOOKBACK_HOURS", "36"))
NEWS_TTL_HOURS = int(os.getenv("NEWS_TTL_HOURS", "120"))  # länger, damit Themen nicht wieder auftauchen
NEWS_STATE_PATH = os.getenv("NEWS_STATE_PATH", "state/news_state.json")
FEEDS_PATH = os.getenv("NEWS_FEEDS_PATH", "config/news_feeds.json")

USER_AGENT = os.getenv("NEWS_USER_AGENT", "telegram-news-bot/3.1 (curated; dedupe; timewindow)")

# Nur zwischen 06:00 und 21:00 Uhr (DE Zeit) senden
ACTIVE_START_HOUR = int(os.getenv("NEWS_ACTIVE_START_HOUR", "6"))
ACTIVE_END_HOUR = int(os.getenv("NEWS_ACTIVE_END_HOUR", "21"))

# Alle 2h-Slots (06,08,10,...,20). 21:00 soll nicht als Slot zählen.
SEND_EVERY_N_HOURS = int(os.getenv("NEWS_SEND_EVERY_N_HOURS", "2"))

# Kuratierte Themen pro Region
MIN_TOPICS_PER_REGION = int(os.getenv("NEWS_MIN_TOPICS_PER_REGION", "2"))
MAX_TOPICS_PER_REGION = int(os.getenv("NEWS_MAX_TOPICS_PER_REGION", "4"))

# Telegram Limit
TELEGRAM_CHAR_LIMIT = int(os.getenv("NEWS_TELEGRAM_CHAR_LIMIT", "3900"))

# Wie viel Kontext pro Thema
BRIEFING_MAX_CHARS = int(os.getenv("NEWS_BRIEFING_MAX_CHARS", "320"))

# “Major Update”-Logik:
# Thema wird erneut gesendet, wenn:
# - neuer Artikel deutlich neuer ist ODER
# - Score deutlich höher ist ODER
# - Alert-Wörter enthalten sind
MAJOR_UPDATE_MIN_HOURS = int(os.getenv("NEWS_MAJOR_UPDATE_MIN_HOURS", "6"))   # frühestens nach 6h wieder
MAJOR_UPDATE_SCORE_DELTA = float(os.getenv("NEWS_MAJOR_UPDATE_SCORE_DELTA", "2.0"))

ALERT_KEYWORDS = [
    "eilmeldung", "breaking", "urgent", "hackerangriff", "angriff", "explosion",
    "notstand", "krise", "crash", "stürzt", "massiv", "sanktionen", "zölle",
    "fed", "ezb", "zins", "insolvenz", "bankrott", "bankrun"
]

# Reihenfolge / Labels
REGION_ORDER = ["de", "us", "eu", "world"]
REGION_TITLES = {"de": "Deutschland", "us": "Amerika (USA)", "eu": "Europa", "world": "Welt"}

# Relevanz/Score
MIN_SCORE = {
    "de": float(os.getenv("NEWS_MIN_SCORE_DE", "0.9")),
    "us": float(os.getenv("NEWS_MIN_SCORE_US", "0.9")),
    "eu": float(os.getenv("NEWS_MIN_SCORE_EU", "0.9")),
    "world": float(os.getenv("NEWS_MIN_SCORE_WORLD", "1.0")),
}

# Wirtschaftsgewichtung
ECON_KEYWORDS = [
    "inflation", "cpi", "ppi", "fed", "fomc", "ecb", "ezb", "zinsen", "zins", "interest rate",
    "gdp", "bip", "rezession", "recession", "arbeitsmarkt", "unemployment", "jobs",
    "zoll", "zölle", "tariff", "trade", "export", "import",
    "öl", "oil", "gas", "lng", "energie", "energy",
    "aktien", "stocks", "market", "märkte", "börse", "dax", "s&p", "dow",
    "bank", "banken", "anleihe", "bond", "yield", "treasury",
    "quartal", "earnings", "results", "guidance", "übernahme", "merger", "acquisition",
    "insolvenz", "bankrott", "schulden", "debt"
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

def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = html.unescape(s)                 # &#x27; etc.
    s = re.sub(r"<[^>]+>", " ", s)       # HTML tags aus RSS-Teasern entfernen
    s = re.sub(r"\s+", " ", s).strip()
    return s

def strip_control_chars(s: str) -> str:
    if not s:
        return s
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)

def parse_entry_time(entry) -> Optional[int]:
    t = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not t:
        return None
    return int(time.mktime(t))

def is_recent(ts: Optional[int]) -> bool:
    if ts is None:
        return True
    return ts >= now_ts() - NEWS_LOOKBACK_HOURS * 3600

def normalize_url(url: str) -> str:
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

def short_domain(url: str) -> str:
    try:
        return urlsplit(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""

def econ_score(text: str) -> float:
    t = (text or "").lower()
    return sum(1.0 for kw in ECON_KEYWORDS if kw in t)

def has_alert(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ALERT_KEYWORDS)

def quality_score(item: Item) -> float:
    """
    Score: Wirtschaftssignal + Recency + Quellengewicht.
    """
    s = econ_score(f"{item.title} {item.summary}")

    if item.published_ts:
        age_h = max(0.0, (now_ts() - item.published_ts) / 3600.0)
        s += max(0.0, 2.0 - (age_h / 12.0))

    s *= max(0.6, min(2.0, item.weight))

    # Alert-Boost (krasse News)
    if has_alert(f"{item.title} {item.summary}"):
        s += 2.5

    return s

def berlin_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Berlin"))
    except Exception:
        return datetime.now()

def berlin_time_str() -> str:
    return berlin_now().strftime("%d.%m.%Y %H:%M")

def in_active_window_and_slot() -> Tuple[bool, str]:
    """
    True nur in Zeitfenster und exakt auf 2h-Slots:
    06,08,10,12,14,16,18,20.
    """
    dt = berlin_now()
    hour = dt.hour

    if hour < ACTIVE_START_HOUR or hour >= ACTIVE_END_HOUR:
        return False, f"Outside active window ({ACTIVE_START_HOUR}-{ACTIVE_END_HOUR})."

    # Slot-Bedingung relativ zu Startstunde
    if ((hour - ACTIVE_START_HOUR) % SEND_EVERY_N_HOURS) != 0:
        return False, "Not a send slot."

    slot_id = dt.strftime("%Y-%m-%d") + f"_{hour:02d}"
    return True, slot_id

def summarize_for_briefing(summary: str, max_chars: int) -> str:
    s = strip_control_chars(clean_text(summary))
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars].rsplit(" ", 1)[0]
    return (cut if cut else s[:max_chars]).rstrip() + "…"

def topic_signature(title: str) -> str:
    """
    Thema (Topic) identifizieren: Normalisierung + Tokenisierung.
    Dadurch werden ähnliche Headlines als “gleiches Thema” behandelt.
    """
    t = strip_control_chars(clean_text(title)).lower()
    t = re.sub(r"[^a-z0-9äöüß\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    toks = [w for w in t.split() if len(w) >= 4]
    return " ".join(toks[:14])

# -----------------------
# Telegram sending (HTML + fallback)
# -----------------------
def _telegram_send(payload: Dict) -> Tuple[bool, str]:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json=payload, timeout=30)
        if r.ok:
            return True, ""
        return False, r.text
    except Exception as e:
        return False, str(e)

def send_telegram_html_with_fallback(text_html: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

    text_html = (text_html or "").strip()
    if len(text_html) < 10:
        print("Skip send: message too short/empty.")
        return

    payload_html = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text_html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    ok, err = _telegram_send(payload_html)
    if ok:
        return

    print(f"Telegram HTML failed; fallback to plain. Error: {err}")
    text_plain = re.sub(r"<[^>]+>", "", text_html).strip()
    if len(text_plain) < 10:
        print("Skip fallback: plain too short.")
        return

    payload_plain = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text_plain[:4000],
        "disable_web_page_preview": True,
    }
    ok2, err2 = _telegram_send(payload_plain)
    if not ok2:
        print(f"Telegram plain failed: {err2}")

# -----------------------
# Build message (curated topics)
# -----------------------
def build_message(groups: Dict[str, List[Tuple[float, Item]]]) -> str:
    header = f"<b>News Briefing</b> – {html.escape(berlin_time_str())} (DE Zeit)\n"
    parts: List[str] = [header]

    for region in REGION_ORDER:
        items = groups.get(region, [])
        if not items:
            continue

        parts.append(f"\n<b>{html.escape(REGION_TITLES[region])}</b>")

        for score, it in items:
            title = strip_control_chars(clean_text(it.title))
            teaser = summarize_for_briefing(it.summary, BRIEFING_MAX_CHARS)
            src = strip_control_chars(clean_text(it.source))
            dom = short_domain(it.url)

            t = html.escape(title)
            u = html.escape(it.url, quote=True)
            s = html.escape(src)
            d = html.escape(dom)

            # Klickbarer Titel + Briefing
            line1 = f"• <a href=\"{u}\">{t}</a> <i>({s}{' – ' + d if d else ''})</i>"
            parts.append(line1)

            if teaser:
                parts.append(f"  <i>{html.escape(teaser)}</i>")

    text = "\n".join(parts)
    if len(text) > TELEGRAM_CHAR_LIMIT:
        text = text[:TELEGRAM_CHAR_LIMIT].rstrip() + "\n<i>…(gekürzt)</i>"
    return text

# -----------------------
# Main
# -----------------------
def main() -> None:
    # Zeitfenster + Slot-Check
    allowed, slot_id = in_active_window_and_slot()
    if not allowed:
        print(f"Skip: {slot_id}")
        return

    feeds = load_json(FEEDS_PATH, {})
    state = load_json(NEWS_STATE_PATH, {
        "last_digest_ts": 0,
        "last_slot_id": "",
        "seen_urls": {},
        "topics": {}
    })

    # Slot-Dedupe: innerhalb eines Slots niemals doppelt senden
    if state.get("last_slot_id") == slot_id:
        print("Skip: already sent in this slot.")
        return

    seen_urls: Dict[str, int] = state.get("seen_urls", {})
    topics: Dict[str, Dict] = state.get("topics", {})  # signature -> metadata

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # 1) Collect items
    collected: List[Item] = []
    for region, flist in feeds.items():
        if region not in REGION_ORDER:
            continue
        for f in flist:
            name = f.get("name", "Source")
            url = f.get("url")
            weight = float(f.get("weight", 1.0))
            if not url:
                continue

            try:
                resp = session.get(url, timeout=25)
                resp.raise_for_status()
                parsed = feedparser.parse(resp.text)
            except Exception as e:
                print(f"Feed error [{region}] {name}: {e}")
                continue

            for entry in parsed.entries[:80]:
                title = strip_control_chars(clean_text(getattr(entry, "title", "") or ""))
                link = strip_control_chars(clean_text(getattr(entry, "link", "") or ""))
                summary = strip_control_chars(clean_text(getattr(entry, "summary", "") or ""))

                if not title or not link:
                    continue

                pub_ts = parse_entry_time(entry)
                if not is_recent(pub_ts):
                    continue

                nurl = normalize_url(link)
                url_hash = sha1(nurl)
                if url_hash in seen_urls:
                    continue

                collected.append(Item(
                    region=region,
                    source=name,
                    title=title,
                    url=nurl,
                    published_ts=pub_ts,
                    summary=summary,
                    weight=weight
                ))

    if not collected:
        print("No new items collected.")
        return

    # 2) Score & topic-dedupe: pro Topic nur das beste Item behalten
    best_by_topic: Dict[str, Tuple[float, Item]] = {}
    for it in collected:
        score = quality_score(it)
        if score < MIN_SCORE.get(it.region, 0.0):
            continue

        sig = topic_signature(it.title)
        if not sig:
            continue

        prev = best_by_topic.get(sig)
        if (prev is None) or (score > prev[0]):
            best_by_topic[sig] = (score, it)

    if not best_by_topic:
        print("No items after scoring/filtering.")
        return

    # 3) Decide if a topic is allowed to be sent (avoid repeats unless major update)
    def is_major_update(sig: str, score: float, it: Item) -> bool:
        meta = topics.get(sig)
        if not meta:
            return True  # nie gesendet -> ok

        last_sent_ts = int(meta.get("last_sent_ts", 0))
        last_score = float(meta.get("last_score", 0.0))
        last_pub = int(meta.get("last_pub_ts", 0))

        # Alert-Wörter -> immer durchlassen
        if has_alert(f"{it.title} {it.summary}"):
            return True

        # deutlich neuer als letzter Stand?
        pub_ts = it.published_ts or 0
        min_gap_ok = (now_ts() - last_sent_ts) >= (MAJOR_UPDATE_MIN_HOURS * 3600)

        newer_ok = (pub_ts > (last_pub + 3600))  # mind. 1h neuer publiziert
        score_jump_ok = (score - last_score) >= MAJOR_UPDATE_SCORE_DELTA

        return min_gap_ok and (newer_ok or score_jump_ok)

    # 4) Group by region and select top 2–4 topics per region
    per_region: Dict[str, List[Tuple[float, Item, str]]] = {r: [] for r in REGION_ORDER}
    for sig, (score, it) in best_by_topic.items():
        if is_major_update(sig, score, it):
            per_region[it.region].append((score, it, sig))

    groups_final: Dict[str, List[Tuple[float, Item]]] = {}
    topics_sent: List[Tuple[str, float, Item]] = []

    for region in REGION_ORDER:
        candidates = per_region.get(region, [])
        candidates.sort(key=lambda x: x[0], reverse=True)

        chosen: List[Tuple[float, Item]] = []
        chosen_sigs: List[str] = []

        for score, it, sig in candidates:
            if len(chosen) >= MAX_TOPICS_PER_REGION:
                break
            chosen.append((score, it))
            chosen_sigs.append(sig)

        if len(chosen) < MIN_TOPICS_PER_REGION and not ALLOW_FEWER_THAN_MIN:
            continue

        if chosen:
            groups_final[region] = chosen
            for (score, it), sig in zip(chosen, chosen_sigs):
                topics_sent.append((sig, score, it))

    if not groups_final:
        print("No topics selected (all repeats, no major updates).")
        return

    # 5) Build + Send
    msg = build_message(groups_final)
    send_telegram_html_with_fallback(msg)

    # 6) Update state
    ts = now_ts()

    # URLs als gesehen markieren
    for _, it in [x for region_items in groups_final.values() for x in region_items]:
        seen_urls[sha1(it.url)] = ts

    # Topic-Metadaten aktualisieren
    for sig, score, it in topics_sent:
        topics[sig] = {
            "last_sent_ts": ts,
            "last_score": score,
            "last_pub_ts": int(it.published_ts or 0),
            "last_url_hash": sha1(it.url),
            "last_title": clean_text(it.title)[:200]
        }

    # TTL Cleanup
    ttl = NEWS_TTL_HOURS * 3600
    seen_urls = {k: v for k, v in seen_urls.items() if (ts - int(v)) <= ttl}

    # optional: topics sehr lange behalten (damit Wiederholungen vermieden werden)
    topics_ttl = max(ttl, 14 * 24 * 3600)  # 14 Tage
    topics = {k: v for k, v in topics.items() if (ts - int(v.get("last_sent_ts", 0))) <= topics_ttl}

    state["last_digest_ts"] = ts
    state["last_slot_id"] = slot_id
    state["seen_urls"] = seen_urls
    state["topics"] = topics
    save_json(NEWS_STATE_PATH, state)

    print("Digest sent + state updated.")

if __name__ == "__main__":
    main()
