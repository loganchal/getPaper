#!/usr/bin/env python3
"""Probe Paffenholz's public smooth-Fano dimension-8 data stream."""
from __future__ import annotations

import gzip
import html.parser
import urllib.parse
import urllib.request

PAGE = "https://polymake.org/polytopes/paffenholz/www/fano.html"
TARGET = "smooth_fano_8d.gz"


class Links(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def main() -> None:
    with urllib.request.urlopen(PAGE, timeout=60) as r:
        page = r.read().decode("utf-8", errors="replace")
    parser = Links()
    parser.feed(page)
    matches = [urllib.parse.urljoin(PAGE, h) for h in parser.hrefs if TARGET in h]
    if not matches:
        raise RuntimeError(f"No {TARGET!r} link found; links={parser.hrefs[-20:]}")
    url = matches[0]
    print("DATA_URL", url, flush=True)

    polytopes: list[list[list[int]]] = []
    current: list[list[int]] = []
    with urllib.request.urlopen(url, timeout=120) as raw, gzip.GzipFile(fileobj=raw) as gz:
        for raw_line in gz:
            line = raw_line.decode("ascii").strip()
            if line:
                current.append([int(x) for x in line.split()])
            elif current:
                polytopes.append(current)
                current = []
                if len(polytopes) == 5:
                    break
    if current and len(polytopes) < 5:
        polytopes.append(current)

    print("POLYTOPES_READ", len(polytopes))
    for i, rows in enumerate(polytopes):
        widths = sorted({len(r) for r in rows})
        constants = sorted({r[0] for r in rows})
        normals = [r[1:] for r in rows]
        d = len(normals[0])
        minus_units = sum(v == [-int(j == k) for j in range(d)] for k in range(d) for v in normals)
        # The nested count above counts each actual -e_k once.
        print(f"POLY {i}: facets={len(rows)} widths={widths} constants={constants} dim={d} minus_units={minus_units}")
        for row in rows[:min(12, len(rows))]:
            print(" ", " ".join(map(str, row)))


if __name__ == "__main__":
    main()
