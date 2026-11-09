import asyncio
import argparse
import logging
import random
import re
import time
import csv
import os
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin, urldefrag, parse_qsl, urlencode
from urllib.robotparser import RobotFileParser

import httpx
import dateparser       
from bs4 import BeautifulSoup

# ============================== CONFIG ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

TARGET_ARTICLES = 50000    # 🎯 Goal: 50,000 Articles
HTTP_WORKERS = 30          # ⚡ Concurrent Speed: 30 Parallel Tasks
REQUEST_TIMEOUT = 15
MAX_ATTEMPTS = 2
MIN_INTERVAL = 0.3         # Politeness: 0.3s delay per domain
MIN_CONTENT_CHARS = 200    # Minimum valid article length
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

OUTPUT_CSV = "tech_articles_101k.csv"

# 100 High-Quality Tech News Websites
TARGET_SOURCES = [
    "https://www.smashingmagazine.com/", "https://dzone.com/", "https://www.freecodecamp.org/",
    "https://css-tricks.com/", "https://www.sitepoint.com/", "https://towardsdatascience.com/",
    "https://www.artificialintelligence-news.com/", "https://machinelearningmastery.com/", "https://www.marktechpost.com/",
    "https://www.dataconomy.com/", "https://www.analyticsinsight.net/", "https://www.kdnuggets.com/",
    "https://www.space.com/", "https://www.scientificamerican.com/", "https://phys.org/",
    "https://www.sciencedaily.com/", "https://www.livescience.com/", "https://www.newscientist.com/",
    "https://www.sciencealert.com/", "https://www.universetoday.com/", "https://www.astronomy.com/",
    "https://www.securityaffairs.com/", "https://threatpost.com/", "https://www.schneier.com/",
    "https://www.zerodayinitiative.com/", "https://www.recordedfuture.com/", "https://www.mandiant.com/",
    "https://www.securityboulevard.com/", "https://www.helpnetsecurity.com/", "https://www.cybersecuritydive.com/",
    "https://www.cso.com.au/", "https://www.itnews.com.au/", "https://www.securitymagazine.com/",
    "https://techstartups.com/", "https://www.eu-startups.com/", "https://www.startupdaily.net/",
    "https://betanews.com/", "https://www.eweek.com/", "https://www.informationweek.com/",
    "https://www.networkcomputing.com/", "https://www.itpro.co.uk/", "https://www.silicon.co.uk/",
    "https://www.channelweb.co.uk/", "https://www.computerweekly.com/", "https://www.techtarget.com/",
    "https://www.cryptoslate.com/", "https://bitcoinmagazine.com/", "https://www.cryptopolitan.com/",
    "https://cryptonews.com/", "https://www.cryptobriefing.com/", "https://www.cryptopotato.com/",
    "https://www.ign.com/", "https://www.gamespot.com/", "https://www.pcgamesn.com/",
    "https://www.destructoid.com/", "https://www.gamesradar.com/", "https://www.siliconera.com/",
    "https://www.tomsguide.com/", "https://www.gsmarena.com/", "https://www.phonearena.com/",
    "https://www.androidcentral.com/", "https://www.macworld.com/", "https://www.cultofmac.com/",
    "https://www.macobserver.com/", "https://www.windowslatest.com/", "https://www.xda-developers.com/",
    "https://www.androidheadlines.com/", "https://www.droid-life.com/", "https://www.hardwarezone.com.sg/",
    "https://www.overclockers.com/", "https://www.legitreviews.com/", "https://www.hardforum.com/",
    "https://www.custompc.co.uk/", "https://www.pcgameshardware.com/", "https://www.chromestory.com/",
    "https://blog.cloudflare.com/", "https://netflixtechblog.com/", "https://eng.uber.com/",
    "https://engineering.fb.com/", "https://doordash.engineering/", "https://slack.engineering/",
    "https://dropbox.tech/", "https://www.twilio.com/blog", "https://auth0.com/blog/",
    "https://www.digitalocean.com/community/", "https://www.linode.com/blog/", "https://vultr.com/blog/",
    "https://www.theguardian.com/technology", "https://www.independent.co.uk/tech", "https://www.telegraph.co.uk/technology/",
    "https://www.dailymail.co.uk/sciencetech/index.html", "https://www.express.co.uk/life-style/science-technology", "https://www.mirror.co.uk/tech/",
    "https://www.standard.co.uk/tech", "https://www.cityam.com/technology/", "https://www.ft.com/technology",
    "https://www.economist.com/science-and-technology", "https://www.wsj.com/tech", "https://www.bloomberg.com/technology",
    "https://www.reuters.com/technology/", "https://www.cnbc.com/technology/", "https://www.axios.com/technology",
    "https://www.bbc.com/news/technology", "https://www.npr.org/sections/technology/", "https://www.nbcnews.com/tech",
    "https://www.cbsnews.com/technology/", "https://www.usatoday.com/tech/", "https://www.latimes.com/business/technology",
    "https://www.makeuseof.com/", "https://www.howtogeek.com/", "https://www.lifewire.com/"
]

# ============================== UTILS ==============================
class Blocked(Exception):
    pass

EXCLUDE = re.compile(
    r"/(tag|tags|category|categories|author|page|comments|search|login|signup|"
    r"subscribe|account|cart|checkout|privacy|terms|about|contact|careers|advert)"
    r"[/=?#]|\.(jpg|png|pdf|css|js|xml|json)$", re.I
)

def clean_url(url):
    url = urldefrag(url)[0]
    p = urlparse(url)
    q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_") and k.lower() not in ("fbclid", "ref")]
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
    return bool(re.search(r"/(20\d{2}/|\d{4}-\d{2}-|/p/|article|news|review|story|stories|tech)", p.path, re.I) or p.path.count("-") >= 2)

def has_captcha(html):
    t = html[:20000].lower()
    return any(x in t for x in ("cf-challenge", "checking your browser", "just a moment...", "enable javascript", "recaptcha", "hcaptcha", "access denied"))

# ============================== RATE LIMIT + ROBOTS ==============================
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
                r = await client.get(urljoin(f"https://{host}/", "/robots.txt"), timeout=5)
                parser = RobotFileParser()
                parser.parse(r.text.splitlines() if r.status_code == 200 else [])
                self.robots[host] = parser if r.status_code == 200 else None
            except Exception:
                self.robots[host] = None
        parser = self.robots[host]
        return True if parser is None else parser.can_fetch("Mozilla", url)

# ============================== DISCOVERY (SITEMAP ONLY) ==============================
async def fetch_text(client, polite, url):
    if not await polite.allowed(client, url):
        raise Blocked("robots.txt disallow")
    await polite.wait(url)
    r = await client.get(url, timeout=REQUEST_TIMEOUT)
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
        for sub in locs[:50]: 
            await parse_sitemap(client, polite, sub, found, cap, depth + 1, seen)
    else:
        for link in locs:
            if len(found) >= cap:
                break
            if looks_like_article(link, url):
                found.add(clean_url(link))

async def discover_all_urls(client, polite, max_per_site):
    all_urls = set()
    logging.info("Phase 1: Discovering URLs via Sitemaps ONLY...")
    
    for base in TARGET_SOURCES:
        host = urlparse(base).netloc
        found = set()
        
        for name in ("sitemap.xml", "sitemap_index.xml", "post-sitemap.xml", "news-sitemap.xml", "sitemap_index/sitemap-1.xml"):
            try:
                await parse_sitemap(client, polite, urljoin(base, name), found, max_per_site)
            except Exception:
                pass
            if len(found) >= 100: 
                break
                
        all_urls.update(found)
        if len(found) > 0:
            logging.info(f"  {host}: {len(found)} article URLs found")
        
    logging.info(f"DISCOVERY COMPLETE: {len(all_urls)} total unique article URLs queued.")
    return list(all_urls)

# ============================== EXTRACTION ==============================
def parse_article(html, final_url):
    soup = BeautifulSoup(html, "html.parser")
    
    og = soup.select_one('meta[property="og:title"]')
    title = og["content"].strip() if og and og.get("content") else ""
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else "Unknown Title")

    date = "Unknown"
    t = soup.find("time")
    if t and t.get("datetime"):
        date = t["datetime"]
    else:
        m = re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* \d{1,2},? \d{4}\b", soup.get_text(" ", strip=True)[:1500], re.I)
        if m:
            date = m.group(0)
    
    if date != "Unknown":
        d = dateparser.parse(date)
        if d:
            date = d.strftime("%Y-%m-%d")

    content = ""
    nodes = soup.select('[itemprop="articleBody"], article, main, .post-content, .entry-content, .article-body')
    if nodes:
        node = max(nodes, key=lambda e: len(e.get_text(" ", strip=True)))
        for bad in node.select("script, style, nav, footer, header, aside, form, button, iframe, .ad, [class*='ad']"):
            bad.decompose()
        ps = [p.get_text(" ", strip=True) for p in node.find_all("p")]
        content = "\n\n".join(p for p in ps if len(p) > 40)
        
    if len(content) < MIN_CONTENT_CHARS and soup.body:
        body = soup.body
        for bad in body.select("script, style, nav, footer, header, aside, form"):
            bad.decompose()
        content = body.get_text(" ", strip=True)

    return title.strip(), date, content.strip()

# ============================== WORKERS ==============================
async def http_get(client, polite, url):
    for attempt in range(MAX_ATTEMPTS):
        try:
            text = await fetch_text(client, polite, url)
            if has_captcha(text):
                raise Blocked("CAPTCHA detected")
            return text
        except Blocked:
            raise
        except Exception as e:
            if attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(1 + random.uniform(0, 0.5))
    raise RuntimeError("HTTP Fetch Failed")

async def worker(worker_id, url_queue, result_queue, client, polite, blocked_hosts, stop_event):
    while not stop_event.is_set():
        try:
            url = await asyncio.wait_for(url_queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            continue
            
        if url is None: 
            url_queue.task_done()
            break
            
        host = urlparse(url).netloc
        status = "failed"
        title, date, content = "Unknown", "Unknown", ""
        
        try:
            if host in blocked_hosts:
                raise Blocked(f"Host blocked: {blocked_hosts[host]}")

            html = await http_get(client, polite, url)
            title, date, content = parse_article(html, url)

            if len(content) >= MIN_CONTENT_CHARS and title and title != "Unknown Title":
                status = "success"
                
        except Blocked as b:
            logging.debug(f"[{worker_id}] Blocked/Skipped: {host}")
        except Exception as e:
            logging.debug(f"[{worker_id}] Failed: {url} ({str(e)[:50]})")
        finally:
            if status == "success":
                # Order: url, title, content, publish_date (As requested)
                await result_queue.put((url, title, content, date))
            
            url_queue.task_done()

async def csv_writer_worker(result_queue, filename, target, stop_event):
    """Dedicated concurrent task to safely write to CSV."""
    file_exists = os.path.exists(filename)
    
    with open(filename, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(filename) == 0:
            # Exact columns as requested
            writer.writerow(["url", "title", "content", "publish_date"])
        
        saved_count = 0
        while True:
            item = await result_queue.get()
            if item is None: 
                break
            
            writer.writerow(item)
            f.flush() # Force write to disk immediately for safety
            saved_count += 1
            
            if saved_count % 1000 == 0:
                logging.info(f"💾 Progress: {saved_count} / {target} articles saved to CSV...")
                
            if saved_count >= target:
                logging.info(f"🎯 TARGET REACHED: {target} articles successfully scraped!")
                stop_event.set() # Signal all workers to stop immediately
                break
                
            result_queue.task_done()

# ============================== MAIN ==============================
async def main(args):
    if os.path.exists(OUTPUT_CSV) and not args.append:
        os.remove(OUTPUT_CSV)
        logging.info(f"✅ Created new CSV file: {OUTPUT_CSV}")

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=HTTP_WORKERS + 10, max_keepalive_connections=HTTP_WORKERS),
    ) as client:
        polite = Politeness()
        
        # 1. Discovery Phase
        urls_to_scrape = await discover_all_urls(client, polite, args.sitemap_max)
        if not urls_to_scrape:
            logging.error("No URLs found in sitemaps. Exiting.")
            return

        # 2. Setup Queues & Events
        url_queue = asyncio.Queue()
        result_queue = asyncio.Queue()
        stop_event = asyncio.Event()
        
        for url in urls_to_scrape:
            await url_queue.put(url)
            
        for _ in range(HTTP_WORKERS):
            await url_queue.put(None) # Poison pills to stop workers

        blocked_hosts = {}
        
        # 3. Start Writer Task
        writer_task = asyncio.create_task(csv_writer_worker(result_queue, OUTPUT_CSV, TARGET_ARTICLES, stop_event))
        
        # 4. Start Worker Tasks
        logging.info(f"⚡ Phase 2: Starting {HTTP_WORKERS} concurrent workers. Target: {TARGET_ARTICLES} articles...")
        start_time = time.monotonic()
        
        workers = [
            asyncio.create_task(worker(f"W{i}", url_queue, result_queue, client, polite, blocked_hosts, stop_event))
            for i in range(HTTP_WORKERS)
        ]
        
        # Wait until target is reached OR all workers finish their queue
        while not stop_event.is_set() and not all(w.done() for w in workers):
            await asyncio.sleep(1)
            
        # 5. Cleanup
        stop_event.set()
        for w in workers:
            if not w.done():
                w.cancel()
                
        await result_queue.put(None)
        await writer_task

        elapsed = time.monotonic() - start_time
        logging.info(f"✅ RUN COMPLETE in {elapsed:.2f} seconds!")
        logging.info(f"📁 Final output saved to: {os.path.abspath(OUTPUT_CSV)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tech News Sitemap Scraper to CSV (Target 50k)")
    ap.add_argument("--sitemap-max", type=int, default=2000, help="Max articles to extract per website sitemap")
    ap.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")
    
    try:
        asyncio.run(main(ap.parse_args()))
    except KeyboardInterrupt:
        logging.info("\n⚠️ Stopped safely by user. Check the CSV file for saved articles.")