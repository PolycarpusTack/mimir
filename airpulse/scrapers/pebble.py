#!/usr/bin/env python3
"""Pebble Beach Systems blog scraper."""
import json, sys
from datetime import datetime

try:
    from scrapling.fetchers import Fetcher
except ImportError:
    print("ERROR: scrapling not installed", file=sys.stderr)
    sys.exit(2)

def get_arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

def main():
    source_id = get_arg("--source-id", "unknown")
    limit = int(get_arg("--limit", "50"))
    try:
        page = Fetcher.get("https://www.pebble.tv/news/", stealthy_headers=True)
        articles = page.css("article, .news-item, .post", auto_save=True)
        count = 0
        for article in articles[:limit]:
            url = article.css("a::attr(href)").get()
            title = article.css("h2::text, h3::text").get()
            if url and title:
                item = {"url": url if url.startswith("http") else f"https://www.pebble.tv{url}",
                        "title": title.strip(), "summary": None,
                        "published_at": datetime.utcnow().isoformat() + "Z", "source_id": source_id}
                print(json.dumps(item))
                count += 1
        sys.exit(0 if count > 0 else 1)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
