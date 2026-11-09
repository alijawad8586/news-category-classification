import asyncio
import argparse
import logging
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin, urldefrag, parse_qsl, urlencode
from urllib.robotparser import RobotFileParser

import httpx
import psycopg2
import psycopg2.extras
from psycopg2 import errors
import feedparser
import dateparser
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ============================== CONFIG ==============================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "postgres", "password": os.environ.get("DB_PASSWORD"),
    "dbname": "tech_news_db",
}

if not DB_CONFIG["password"]:
    raise RuntimeError("Set the DB_PASSWORD environment variable before running.")

HTTP_WORKERS      = 16     # parallel HTTP tasks
BROWSER_PAGES     = 5      # max simultaneous tabs (4-5 recommended)
REQUEST_TIMEOUT   = 12
MAX_ATTEMPTS      = 3
MIN_INTERVAL      = 0.25   # seconds between requests PER DOMAIN (politeness)
SITEMAP_MAX_PER_SITE = 2000
MIN_CONTENT_CHARS = 250
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

TARGET_SOURCES = [
    "https://techcrunch.com/", "https://www.theverge.com/", "https://arstechnica.com/",
    "https://www.wired.com/", "https://www.zdnet.com/", "https://www.cnet.com/",
    "https://www.engadget.com/", "https://gizmodo.com/", "https://mashable.com/",
    "https://thenextweb.com/", "https://venturebeat.com/", "https://www.theregister.com/",
    "https://slashdot.org/", "https://hackaday.com/", "https://www.techradar.com/",
    "https://www.pcmag.com/", "https://www.digitaltrends.com/", "https://www.geekwire.com/",
    "https://siliconangle.com/", "https://www.producthunt.com/",
    "https://dev.to/", "https://hackernoon.com/", "https://www.technologyreview.com/",
    "https://spectrum.ieee.org/", "https://www.cio.com/", "https://www.computerworld.com/",
    "https://www.infoworld.com/", "https://www.networkworld.com/", "https://www.csoonline.com/",
    "https://www.darkreading.com/", "https://www.bleepingcomputer.com/",
    "https://thehackernews.com/", "https://krebsonsecurity.com/", "https://www.securityweek.com/",
    "https://www.cyberscoop.com/", "https://www.tomshardware.com/", "https://www.servethehome.com/",
    "https://www.pcgamer.com/", "https://www.rockpapershotgun.com/",
    "https://www.eurogamer.net/", "https://kotaku.com/", "https://www.polygon.com/",
    "https://www.androidauthority.com/", "https://9to5mac.com/", "https://9to5google.com/",
    "https://www.macrumors.com/", "https://www.imore.com/", "https://appleinsider.com/",
    "https://www.windowscentral.com/", "https://www.neowin.net/", "https://www.onmsft.com/",
    "https://www.thurrott.com/", "https://www.startupnation.com/",
    "https://www.fastcompany.com/", "https://www.businessinsider.com/",
    "https://www.forbes.com/", "https://techinasia.com/",
    "https://www.siliconrepublic.com/", "https://www.computing.co.uk/", "https://www.v3.co.uk/",
    "https://www.itpro.com/", "https://www.crn.com/", "https://www.channelfutures.com/",
    "https://www.datamation.com/", "https://www.infoq.com/", "https://thenewstack.io/",
    "https://devops.com/", "https://www.cloudcomputing-news.net/",
    "https://www.datacenterdynamics.com/", "https://www.nextplatform.com/",
    "https://www.coindesk.com/", "https://cointelegraph.com/", "https://decrypt.co/",
    "https://www.theblock.co/", "https://www.wired.co.uk/",
    "https://www.techspot.com/", "https://www.guru3d.com/", "https://www.bit-tech.net/",
    "https://www.kitguru.net/", "https://www.techpowerup.com/", "https://www.hardwaretimes.com/",
    "https://www.notebookcheck.net/", "https://www.laptopmag.com/", "https://www.techrepublic.com/",
    "https://www.esecurityplanet.com/", "https://www.govtech.com/",
    "https://www.healthcareitnews.com/", "https://www.mobihealthnews.com/",
    "https://www.medtechdive.com/", "https://www.edsurge.com/",
    "https://www.techhive.com/", "https://www.androidpolice.com/",
    "https://www.extremetech.com/", "https://www.anandtech.com/",
    "https://www.trustedreviews.com/", "https://www.stuff.tv/",
    "https://www.techadvisor.com/", "https://www.welivesecurity.com/",
    "https://www.bleepingcomputer.com/", "https://www.phoronix.com/",
    "https://www.linux.com/", "https://itsfoss.com/", "https://www.omgubuntu.co.uk/",
]

# ============================== DATABASE ==============================
class Blocked(Exception):
    """403 / CAPTCHA / access denied - do NOT retry."""

def setup_database():
    conn = psycopg2.connect(dbname="postgres", user=DB_CONFIG["user"],
                            password=DB_CONFIG["password"],
                            host=DB_CONFIG["host"], port=DB_CONFIG["port"])
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(f'CREATE DATABASE "{DB_CONFIG["dbname"]}";')
                logging.info(f"Database '{DB_CONFIG['dbname']}' created.")
            except errors.DuplicateDatabase:
                pass
    finally:
        conn.close()

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            # Sirf 4 columns, jaisa aapne kaha
            cur.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    url TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    publish_date TEXT
                );
            """)
            # Queue = resume + dedup + status (alag table)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS scrape_queue (
                    url TEXT PRIMARY KEY,
                    source TEXT,
                    status TEXT DEFAULT 'pending',
                    attempts INT DEFAULT 0,
                    last_error TEXT
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON scrape_queue(status, attempts);")
        conn.commit()
    finally:
        conn.close()


class Database:
    """Single connection, serialized via asyncio.Lock (DB calls are milliseconds)."""
    def __init__(self):
        self.conn = None
        self.lock = asyncio.Lock()

    async def connect(self):
        self.conn = await asyncio.to_thread(psycopg2.connect, **DB_CONFIG)

    async def seed(self, urls, source):
        if not urls:
            return 0
        rows = [(u, source) for u in urls]
        async with self.lock:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO scrape_queue (url, source, status) VALUES %s ON CONFLICT DO NOTHING",
                    rows, template="(%s,%s,'pending')", page_size=1000)
            self.conn.commit()
        return len(rows)

    async def claim_next(self):
        async with self.lock:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT url FROM scrape_queue
                    WHERE status='pending' AND attempts < %s
                    ORDER BY url LIMIT 1 FOR UPDATE SKIP LOCKED
                """, (MAX_ATTEMPTS,))
                row = cur.fetchone()
                if not row:
                    self.conn.commit()
                    return None
                cur.execute("UPDATE scrape_queue SET status='processing' WHERE url=%s", (row[0],))
            self.conn.commit()
            return row[0]

    async def finish(self, url, status, title=None, content=None, date=None, error=None):
        async with self.lock:
            try:
                with self.conn.cursor() as cur:
                    if status == "done":
                        cur.execute("""INSERT INTO articles (url,title,content,publish_date)
                                       VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                                    (url, title, content, date))
                    cur.execute("""UPDATE scrape_queue
                                   SET status=%s, attempts=attempts+1, last_error=%s
                                   WHERE url=%s""", (status, error, url))
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    async def stats(self):
        async with self.lock:
            with self.conn.cursor() as cur:
                cur.execute("SELECT status, COUNT(*) FROM scrape_queue GROUP BY status")
                return dict(cur.fetchall())

    async def close(self):
        if self.conn:
            await asyncio.to_thread(self.conn.close)


def export_csv(path):
    """Server-side streaming COPY -> Excel friendly, no memory blowup."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur, open(path, "w", newline="", encoding="utf-8-sig") as f:
            cur.copy_expert(
                "COPY (SELECT url,title,publish_date,content FROM articles) TO STDOUT WITH CSV HEADER", f)
        logging.info(f"CSV exported: {path}")
    finally:
        conn.close()

# ====================== RATE LIMIT + ROBOTS ======================
class Politeness:
    def __init__(self):
        self.next_slot = {}
        self.lock = asyncio.Lock()
        self.robots = {}

    async def wait(self, url):
        host = urlparse(url).netloc
        async with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot.get(host, 0)) + MIN_INTERVAL
            self.next_slot[host] = slot
        delay = slot - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    async def allowed(self, client, url):
        host = urlparse(url).netloc
        if host not in self.robots:
            try:
                r = await client.get(urljoin(f"https://{host}/", "/robots.txt"))
                parser = RobotFileParser()
                parser.parse(r.text.splitlines() if r.status_code == 200 else [])
                self.robots[host] = parser if r.status_code == 200 else None
            except Exception:
                self.robots[host] = None
        parser = self.robots[host]
        return True if parser is None else parser.can_fetch("Mozilla", url)

# ====================== DISCOVERY (SITEMAP FIRST) ======================
EXCLUDE = re.compile(
    r"/(tag|tags|category|categories|author|page|comments|search|login|signup|"
    r"subscribe|account|cart|checkout|privacy|terms|about|contact|careers|advert)"
    r"[/=?#]|\.(jpg|png|pdf|css|js|xml|json)$", re.I)

def clean_url(url):
    url = urldefrag(url)[0]
    p = urlparse(url)
    q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_")
         and k.lower() not in ("fbclid", "ref")]
    return f"{p.scheme}://{p.netloc.lower()}{p.path.rstrip('/')}" + (f"?{urlencode(q)}" if q else "")

def looks_like_article(url, base):
    u = clean_url(url)
    p = urlparse(u)
    if p.scheme not in ("http", "https") or not p.netloc.endswith(urlparse(base).netloc.split(":")[0]):
        return False
    if EXCLUDE.search(p.path):
        return False
    if len(p.path.strip("/").split("/")) < 2:
        return False
    return bool(re.search(r"/(20\d{2}/|\d{4}-\d{2}-|/p/|article|news|review|story|stories|tech)", p.path, re.I)
                or p.path.count("-") >= 2)

async def fetch_text(client, polite, url):
    if not await polite.allowed(client, url):
        raise Blocked("robots.txt disallow")
    await polite.wait(url)
    r = await client.get(url)
    if r.status_code in (401, 403, 429):
        raise Blocked(f"HTTP {r.status_code}")
    r.raise_for_status()
    return r.text

async def parse_sitemap(client, polite, url, found, cap, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if depth > 2 or url in seen or len(found) >= cap:
        return
    seen.add(url)
    try:
        text = await fetch_text(client, polite, url)
        root = ET.fromstring(text)
    except Exception:
        return
    locs = [el.text.strip() for el in root.iter() if el.tag.split("}")[-1] == "loc" and el.text]
    if root.tag.split("}")[-1] == "sitemapindex":
        for sub in locs[:40]:
            await parse_sitemap(client, polite, sub, found, cap, depth + 1, seen)
    else:
        for link in locs:
            if len(found) >= cap:
                break
            if looks_like_article(link, url):
                found.add(clean_url(link))

async def discover(client, polite, db, cap):
    total = 0
    for base in TARGET_SOURCES:
        found = set()
        host = urlparse(base).netloc
        for name in ("sitemap.xml", "sitemap_index.xml", "post-sitemap.xml", "news-sitemap.xml"):
            try:
                await parse_sitemap(client, polite, urljoin(base, name), found, cap)
            except Exception:
                pass
            if len(found) >= 50:
                break
        if len(found) < 20:  # RSS fallback
            for name in ("feed/", "rss.xml", "feed.xml"):
                try:
                    xml = await fetch_text(client, polite, urljoin(base, name))
                    feed = feedparser.parse(xml)
                    for e in feed.entries:
                        if getattr(e, "link", None):
                            found.add(clean_url(e.link))
                    if feed.entries:
                        break
                except Exception:
                    pass
        added = await db.seed(sorted(found), host)
        total += len(found)
        logging.info(f"{host}: {len(found)} candidate URLs queued")
    logging.info(f"DISCOVERY COMPLETE: {total} candidate URLs in queue")

# ====================== EXTRACTION ======================
def parse_article(html):
    soup = BeautifulSoup(html, "html.parser")
    og = soup.select_one('meta[property="og:title"]')
    title = og["content"].strip() if og and og.get("content") else ""
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else "")

    date = ""
    t = soup.find("time")
    if t and t.get("datetime"):
        date = t["datetime"]
    else:
        m = re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* \d{1,2},? \d{4}\b",
                      soup.get_text(" ", strip=True)[:1500], re.I)
        if m:
            date = m.group(0)
    if date:
        d = dateparser.parse(date)
        if d:
            date = d.strftime("%Y-%m-%d")

    content = ""
    nodes = soup.select('[itemprop="articleBody"], article, main, .post-content, .entry-content, .article-body')
    if nodes:
        node = max(nodes, key=lambda e: len(e.get_text(" ", strip=True)))
        for bad in node.select("script,style,nav,footer,header,aside,form,button,iframe,div.ad"):
            bad.decompose()
        ps = [p.get_text(" ", strip=True) for p in node.find_all("p")]
        content = "\n\n".join(p for p in ps if len(p) > 40)
    if len(content) < MIN_CONTENT_CHARS and soup.body:
        body = soup.body
        for bad in body.select("script,style,nav,footer,header,aside,form"):
            bad.decompose()
        content = body.get_text(" ", strip=True)
    return title.strip(), date, content.strip()

def has_captcha(html):
    t = html[:20000].lower()
    return any(x in t for x in ("cf-challenge", "checking your browser", "just a moment...",
                                "enable javascript and cookies", "recaptcha", "hcaptcha", "access denied"))

# ====================== FAST HTTP + REUSED BROWSER ======================
class BrowserPool:
    def __init__(self, context):
        self.context = context
        self.free = asyncio.Queue()
        self.sem = asyncio.Semaphore(BROWSER_PAGES)

    async def get_html(self, url):
        async with self.sem:
            page = self.free.get_nowait() if not self.free.empty() else None
            keep = False
            try:
                if page is None:
                    page = await self.context.new_page()
                    await page.set_default_timeout(REQUEST_TIMEOUT * 1000)
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                if resp and resp.status in (401, 403, 429):
                    raise Blocked(f"HTTP {resp.status}")
                await page.wait_for_timeout(400)
                html = await page.content()
                if has_captcha(html):
                    raise Blocked("CAPTCHA")
                keep = True
                return html
            except PWTimeout:
                raise RuntimeError("browser timeout")
            finally:
                if keep:
                    await self.free.put(page)     # REUSE tab, no relaunch
                elif page is not None:
                    try:
                        await page.close()
                    except Exception:
                        pass

async def http_get(client, polite, url):
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            text = await fetch_text(client, polite, url)
            if has_captcha(text):
                raise Blocked("CAPTCHA")
            return text
        except Blocked:
            raise
        except Exception as e:
            last = e
            if attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(2 ** attempt + random.uniform(0, 0.5))
    raise RuntimeError(str(last))

# ====================== WORKER PIPELINE ======================
async def worker(name, db, client, polite, pool, blocked_hosts, state, limit, lock):
    while True:
        async with lock:
            if state["processed"] >= limit:
                return
        url = await db.claim_next()
        if not url:
            return
        host = urlparse(url).netloc
        status, error, title, content, date = "failed", None, None, None, None
        try:
            if host in blocked_hosts:
                raise Blocked(blocked_hosts[host])

            html = await http_get(client, polite, url)
            title, date, content = parse_article(html)

            # Browser fallback ONLY when needed
            if len(content) < MIN_CONTENT_CHARS or not title:
                try:
                    html = await pool.get_html(url)
                    title, date, content = parse_article(html)
                except Blocked as b:
                    blocked_hosts[host] = str(b)
                    raise

            if len(content) < MIN_CONTENT_CHARS or not title:
                error = "no usable article content"
            else:
                status = "done"
        except Blocked as b:
            status, error = "captcha", str(b)
            blocked_hosts.setdefault(host, str(b))
            logging.warning(f"MANUAL CHECK NEEDED ({host}): {url}")
        except Exception as e:
            error = str(e)[:300]
        finally:
            await db.finish(url, status, title, content, date, error)
            async with lock:
                state["processed"] += 1
                state["done" if status == "done" else "bad"] += 1
                n = state["processed"]
                done = state["done"]
            if n % 25 == 0:
                elapsed = time.monotonic() - state["start"]
                logging.info(f"[{done} saved | {state['bad']} failed | {done/max(elapsed,1):.2f} articles/sec]")

# ====================== MAIN ======================
async def main(args):
    setup_database()
    db = Database()
    await db.connect()

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=HTTP_WORKERS, max_keepalive_connections=HTTP_WORKERS),
    ) as client:
        polite = Politeness()
        stats = await db.stats()
        if args.rediscover or sum(stats.values()) == 0:
            logging.info("Phase 1: Discovering via sitemaps + RSS...")
            await discover(client, polite, db, args.sitemap_max)
        else:
            logging.info(f"Resuming existing queue: {stats}")

        state = {"processed": 0, "done": 0, "bad": 0, "start": time.monotonic()}
        lock, blocked = asyncio.Lock(), {}

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=not args.headed,
                                              args=["--disable-blink-features=AutomationControlled",
                                                    "--disable-dev-shm-usage"])
            context = await browser.new_context(user_agent=USER_AGENT, viewport={"width": 1366, "height": 900})
            pool = BrowserPool(context)
            try:
                workers = [asyncio.create_task(
                    worker(f"W{i}", db, client, polite, pool, blocked, state, lock, args.limit)
                ) for i in range(HTTP_WORKERS)]
                await asyncio.gather(*workers, return_exceptions=True)
            finally:
                while not pool.free.empty():
                    pg = pool.free.get_nowait()
                    await pg.close()
                await context.close()
                await browser.close()

    logging.info(f"RUN COMPLETE: {state['done']} saved, {state['bad']} failed/captcha")
    logging.info(f"Queue status: {await db.stats()}")
    if not args.no_csv:
        await asyncio.to_thread(export_csv, args.csv)
    await db.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100000, help="max articles this run")
    ap.add_argument("--sitemap-max", type=int, default=SITEMAP_MAX_PER_SITE)
    ap.add_argument("--rediscover", action="store_true")
    ap.add_argument("--headed", action="store_true", help="visible browser for CAPTCHA checks")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--csv", default="scraped_news.csv")
    try:
        asyncio.run(main(ap.parse_args()))
    except KeyboardInterrupt:
        logging.info("Stopped safely. Queue + saved articles remain in PostgreSQL. Run again to resume.")
