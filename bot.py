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
NEWS_TTL_HOURS = int(os.getenv("NEWS_TTL_HOURS", "72"))
NEWS_STATE_PATH = os.getenv("NEWS_STATE_PATH", "state/news_state.json")
FEEDS_PATH = os.getenv("NEWS_FEEDS_PATH", "config/news_feeds.json")

USER_AGENT = os.getenv("NEWS_USER_AGENT", "telegram-news-bot/2.0 (LibreTranslate; GitHub Actions)")

# Frequenz: alle 2 Stunden
DIGEST_INTERVAL_SECONDS = int(os.getenv("NEWS_DIGEST_INTERVAL_SECONDS", str(2 * 3600)))

# Ausgabe-Setup (DE/US vorne)
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
    "world": float(os.getenv("NEWS_ECON_SHARE_WORLD", "0.50")),
}
MIN_SCORE = {
    "de": float(os.getenv("NEWS_MIN_SCORE_DE", "0.8")),
    "us": float(os.getenv("NEWS_MIN_SCORE_US", "0.8")),
    "eu": float(os.getenv("NEWS_MIN_SCORE_EU", "0.8")),
    "world": float(os.getenv("NEWS_MIN_SCORE_WORLD", "1.0")),
}

# LibreTranslate Endpunkte (kostenlos; ohne API-Key). Wir nutzen Failover.
LIBRE_ENDPOINTS = [
    os.getenv("LIBRETRANSLATE_ENDPOINT_1", "https://libretranslate.de/translate").strip(),
    os.getenv("LIBRETRANSLATE_ENDPOINT_2", "https://translate.argosopentech.com/translate").strip(),
]
LIBRE_TIMEOUT = int(os.getenv("LIBRETRANSLATE_TIMEOUT", "15"))
LIBRE_SLEEP_BETWEEN_CALLS_MS = int(os.getenv("LIBRETRANSLATE_SLEEP_MS", "120"))

# Übersetzungsstrategie
TRANSLATE_SUMMARY = os.getenv("NEWS_TRANSLATE_SUMMARY", "0").strip() == "1"  # default aus (spart Calls)
TRANSLATE_ONLY_IF_NOT_GERMAN = os.getenv("NEWS_TRANSLATE_ONLY_IF_NOT_GERMAN", "1").strip() == "1"

# Hard limit: Telegram ~4096 chars, wir bleiben konservativ
TELEGRAM_CHAR_LIMIT = int(os.getenv("NEWS_TELEGRAM_CHAR_LIMIT", "3900"))

# Schlüsselwörter zur Wirtschaftsgewichtung
ECON_KEYWORDS = [
    # Zentralbanken / Makro
    "inflation", "cpi", "ppi", "fed", "fomc", "ecb", "ezb", "interest rate", "rate hike", "rate cut",
    "gdp", "bip", "recession", "rezession", "unemployment", "arbeitsmarkt", "jobs", "payrolls",
    # Handel / Energie
    "tariff", "zoll", "trade", "exports", "imports", "oil", "crude", "brent", "wti", "gas", "lng", "energy", "energie",
    # Finanzen / Unternehmen
    "earnings", "results", "guidance", "merger", "acquisition", "übernahme", "ipo", "insolvenz", "bankruptcy",
    "bond", "anleihe", "yield", "treasury", "debt", "schulden",
    # Märkte / Währungen
    "market", "markets", "märkte", "stocks", "aktien", "shares", "index", "dax", "s&p", "dow",
    "usd", "dollar", "eur", "euro"
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
# Utilities
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
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s

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
    """
    Entfernt typische Tracking-Parameter (utm_ etc.), damit Dedupe sauber ist.
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

def short_domain(url: str) -> str:
    try:
        netloc = urlsplit(url).netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return ""

def econ_score(text: str) -> float:
    t = (text or "").lower()
    return sum(1.0 for kw in ECON_KEYWORDS if kw in t)

def is_econ(item: Item) -> bool:
    return econ_score(f"{item.title} {item.summary}") >= 1.0

def quality_score(item: Item) -> float:
    """
    Score: Wirtschafts-Relevanz + Recency + Quellengewicht
    """
    s = econ_score(f"{item.title} {item.summary}")

    if item.published_ts:
        age_h = max(0.0, (now_ts() - item.published_ts) / 3600.0)
        s += max(0.0, 2.0 - (age_h / 12.0))  # frische News höher

    s *= max(0.5, min(2.0, item.weight))
    return s

def berlin_time_str() -> str:
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.now(ZoneInfo("Europe/Berlin"))
        return dt.strftime("%d.%m.%Y %H:%M")
    except Exception:
        return datetime.now().strftime("%d.%m.%Y %H:%M")

def looks_german(text: str) -> bool:
    """
    Heuristik: wenn Umlaute/ß oder typische deutsche Stopwörter vorkommen, übersetzen wir nicht.
    Ziel: Übersetzungs-Calls sparen.
    """
    t = (text or "").lower()
    if any(ch in t for ch in ["ä", "ö", "ü", "ß"]):
        return True
    # Sehr grobe deutsche Funktionswörter:
    de_markers = [" der ", " die ", " das ", " und ", " nicht ", " ein ", " eine ", " wird ", " wurden ", " über "]
    return any(m in f" {t} " for m in de_markers)

# -----------------------
# LibreTranslate translation (free, with failover + caching)
# -----------------------
def translate_en_to_de(text: str, session: requests.Session, cache: Dict[str, str]) -> str:
    """
    Übersetzt EN->DE. Fallback: Originaltext.
    Cache über Hash, um doppelte Requests innerhalb eines Runs zu vermeiden.
    """
    text = clean_text(text)
    if not text:
        return text

    if TRANSLATE_ONLY_IF_NOT_GERMAN and looks_german(text):
        return text

    key = sha1(text)
    if key in cache:
        return cache[key]

    payload = {"q": text, "source": "en", "target": "de", "format": "text"}

    for ep in [e for e in LIBRE_ENDPOINTS if e]:
        try:
            r = session.post(
                ep,
                json=payload,
                timeout=LIBRE_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            if r.ok:
                data = r.json()
                translated = clean_text(data.get("translatedText", "")) or text
                cache[key] = translated
                # kleine Pause, um öffentliche Instanzen nicht zu stressen
                if LIBRE_SLEEP_BETWEEN_CALLS_MS > 0:
                    time.sleep(LIBRE_SLEEP_BETWEEN_CALLS_MS / 1000.0)
                return translated
        except Exception:
            continue

    cache[key] = text
    return text

# -----------------------
# Telegram output (HTML with short links)
# -----------------------
def send_telegram_html(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

def build_digest_html(groups: Dict[str, List[Item]]) -> str:
    header = f"<b>News Briefing</b> – {html.escape(berlin_time_str())} (DE Zeit)\n"
    titles = {"de": "Deutschland", "us": "USA", "eu": "Europa", "world": "Welt"}
    order = ["de", "us", "eu", "world"]

    parts: List[str] = [header]

    for region in order:
        parts.append(f"\n<b>{titles[region]}</b>")
        region_items = groups.get(region, [])
        if not region_items:
            parts.append("• (keine Headlines gefunden)")
            continue

        for it in region_items:
            t = html.escape(clean_text(it.title))
            u = html.escape(it.url, quote=True)
            src = html.escape(it.source)
            dom = html.escape(short_domain(it.url))

            # Klickbarer Titel -> URL bleibt "klein" (unsichtbar)
            suffix = f"{src}" + (f" – {dom}" if dom else "")
            parts.append(f"• <a href=\"{u}\">{t}</a> <i>({suffix})</i>")

    text = "\n".join(parts)
    if len(text) > TELEGRAM_CHAR_LIMIT:
        text = text[:TELEGRAM_CHAR_LIMIT].rstrip() + "\n<i>…(gekürzt)</i>"
    return text

# -----------------------
# Main
# -----------------------
def main() -> None:
    feeds = load_json(FEEDS_PATH, {})
    state = load_json(NEWS_STATE_PATH, {"last_digest_ts": 0, "seen": {}})

    # Timing: nur alle 2h senden (robust gegen beliebige Workflow-Läufe)
    last = int(state.get("last_digest_ts", 0))
    if last and (now_ts() - last) < (DIGEST_INTERVAL_SECONDS - 30):
        print("Skip: already sent within interval.")
        return

    seen: Dict[str, int] = state.get("seen", {})
    items: List[Item] = []

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

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
                resp = session.get(url, timeout=25)
                resp.raise_for_status()
                parsed = feedparser.parse(resp.text)
            except Exception as e:
                print(f"Feed error [{region}] {name}: {e}")
                continue

            for entry in parsed.entries[:60]:
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
        print("No new items collected (feeds empty or all seen).")
        return

    # 2) Translate to German (titles always; summaries optional)
    # Ziel: du willst alles deutsch – wir übersetzen englische Headlines zuverlässig.
    translate_cache: Dict[str, str] = {}
    for it in items:
        it.title = translate_en_to_de(it.title, session=session, cache=translate_cache)
        if TRANSLATE_SUMMARY and it.summary:
            it.summary = translate_en_to_de(it.summary, session=session, cache=translate_cache)

    # 3) Score & sort
    scored: List[Tuple[float, Item]] = [(quality_score(it), it) for it in items]
    scored.sort(key=lambda x: x[0], reverse=True)

    # 4) Bucket by econ vs normal (after translation, econ detection still works via keywords;
    # for German-only econ terms ist es ok. Optional könnte man zweisprachige Keywords pflegen.)
    econ_bucket: Dict[str, List[Item]] = {k: [] for k in REGION_TOTAL}
    norm_bucket: Dict[str, List[Item]] = {k: [] for k in REGION_TOTAL}

    for score, it in scored:
        if score < MIN_SCORE.get(it.region, 0.0):
            continue
        (econ_bucket if is_econ(it) else norm_bucket)[it.region].append(it)

    # 5) Select per region: econ target then normal then fallback
    groups: Dict[str, List[Item]] = {k: [] for k in REGION_TOTAL}
    for region, total in REGION_TOTAL.items():
        econ_target = int(round(total * REGION_ECON_SHARE[region]))

        sel: List[Item] = []
        sel.extend(econ_bucket[region][:econ_target])

        remaining = total - len(sel)
        if remaining > 0:
            sel.extend(norm_bucket[region][:remaining])

        # fallback: falls eine Kategorie zu dünn ist
        if len(sel) < total:
            need = total - len(sel)
            sel.extend(econ_bucket[region][len(sel):len(sel) + need])
        if len(sel) < total:
            need = total - len(sel)
            sel.extend(norm_bucket[region][len(sel):len(sel) + need])

        groups[region] = sel[:total]

    if not any(groups.values()):
        print("No items selected after scoring/filtering.")
        return

    # 6) Send Telegram (HTML)
    msg = build_digest_html(groups)
    send_telegram_html(msg)

    # 7) Update state: seen + TTL cleanup
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
