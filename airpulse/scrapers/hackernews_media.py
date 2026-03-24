#!/usr/bin/env python3
"""Hacker News media filter scraper — extracts HN posts matching broadcast/OTT/streaming keywords."""
import json
import sys
import urllib.request

KEYWORDS = ["broadcast", "streaming", "ott", "playout", "dvb", "atsc", "fast channel",
            "media tech", "video encoding", "transcoding", "cdn", "drm"]


def get_arg(flag, default=None):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


def main():
    source_id = get_arg("--source-id", "unknown")
    limit = int(get_arg("--limit", "50"))

    try:
        # Use HN Algolia API (no auth needed)
        query = " OR ".join(KEYWORDS[:5])
        url = f"https://hn.algolia.com/api/v1/search_by_date?query={query}&tags=story&hitsPerPage={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": "AirPulse/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        count = 0
        for hit in data.get("hits", []):
            title = hit.get("title", "")
            story_url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"

            # Filter by keywords in title
            title_lower = title.lower()
            if any(kw in title_lower for kw in KEYWORDS):
                item = {
                    "url": story_url,
                    "title": title,
                    "summary": hit.get("story_text") or None,
                    "published_at": hit.get("created_at", ""),
                    "source_id": source_id,
                }
                print(json.dumps(item))
                count += 1

        sys.exit(0 if count > 0 else 1)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
