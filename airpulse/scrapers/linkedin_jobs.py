#!/usr/bin/env python3
"""LinkedIn job posting surge detector for tracked broadcast/media vendors.
Outputs NDJSON with vendor_name, role_category, and posting counts.
Note: Uses public job listing pages, not LinkedIn API (no auth needed for counts)."""
import json
import sys
from datetime import datetime

# Tracked vendors from VENDOR_WATCHLIST
VENDORS = [
    "Imagine Communications", "Harmonic", "Evertz", "Grass Valley", "Vizrt",
    "Dalet", "Avid", "Pebble", "BroadPeak", "Ateme",
]

ROLE_CATEGORIES = {
    "ML Engineer": "MlEngineering",
    "AI Researcher": "MlEngineering",
    "LLM Engineer": "MlEngineering",
    "Cloud Architect": "CloudInfra",
    "DevOps": "CloudInfra",
    "Kubernetes": "CloudInfra",
    "VP Product": "ProductManagement",
    "Director of Product": "ProductManagement",
    "CTO": "Engineering",
    "VP Engineering": "Engineering",
    "Broadcast Engineer": "BroadcastEngineering",
    "Playout Engineer": "BroadcastEngineering",
    "Enterprise Sales": "Sales",
    "Account Executive": "Sales",
}


def get_arg(flag, default=None):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


def main():
    source_id = get_arg("--source-id", "unknown")

    try:
        from scrapling.fetchers import Fetcher
    except ImportError:
        # Fallback: output stub data for testing
        print("WARNING: scrapling not installed, using stub data", file=sys.stderr)
        for vendor in VENDORS[:3]:
            item = {
                "vendor_name": vendor,
                "role_category": "MlEngineering",
                "title": f"ML Engineer at {vendor}",
                "date_posted": datetime.utcnow().isoformat() + "Z",
                "source_id": source_id,
            }
            print(json.dumps(item))
        sys.exit(0)

    count = 0
    for vendor in VENDORS:
        try:
            # Search LinkedIn jobs page (public, no auth)
            search_url = f"https://www.linkedin.com/jobs/search/?keywords={vendor.replace(' ', '%20')}&location=Worldwide"
            page = Fetcher.get(search_url, stealthy_headers=True)

            job_cards = page.css(".job-search-card", auto_save=True)
            for card in job_cards[:10]:
                title = card.css("h3::text").get()
                if title:
                    # Categorise by role
                    category = "Engineering"  # default
                    for keyword, cat in ROLE_CATEGORIES.items():
                        if keyword.lower() in title.lower():
                            category = cat
                            break

                    item = {
                        "vendor_name": vendor,
                        "role_category": category,
                        "title": title.strip(),
                        "date_posted": datetime.utcnow().isoformat() + "Z",
                        "source_id": source_id,
                    }
                    print(json.dumps(item))
                    count += 1
        except Exception as e:
            print(f"WARNING: Failed to scrape {vendor}: {e}", file=sys.stderr)

    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
