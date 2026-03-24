#!/usr/bin/env python3
"""Grass Valley blog scraper using Scrapling adaptive element tracking."""
import json
import sys
from datetime import datetime

try:
    from scrapling.fetchers import Fetcher
except ImportError:
    print("ERROR: scrapling not installed", file=sys.stderr)
    sys.exit(2)


def get_source_id():
    if "--source-id" in sys.argv:
        return sys.argv[sys.argv.index("--source-id") + 1]
    return "unknown"


def get_limit():
    if "--limit" in sys.argv:
        return int(sys.argv[sys.argv.index("--limit") + 1])
    return 50


def main():
    source_id = get_source_id()
    limit = get_limit()

    try:
        page = Fetcher.get("https://blog.grassvalley.com/", stealthy_headers=True)
        articles = page.css(".blog-post-item", auto_save=True)

        count = 0
        for article in articles[:limit]:
            url = article.css("a::attr(href)").get()
            title = article.css("h2::text").get()
            summary = article.css(".excerpt::text").get()
            date_str = article.css("time::attr(datetime)").get()

            if url and title:
                item = {
                    "url": url if url.startswith("http") else f"https://blog.grassvalley.com{url}",
                    "title": title.strip(),
                    "summary": summary.strip() if summary else None,
                    "published_at": date_str or datetime.utcnow().isoformat() + "Z",
                    "source_id": source_id,
                }
                print(json.dumps(item))
                count += 1

        if count == 0:
            print("WARNING: No articles found", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
