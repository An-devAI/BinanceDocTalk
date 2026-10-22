"""Allowlisted collection. No browser evasion, login, or automatic rights assertion."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
import re
import time
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.robotparser import RobotFileParser

from .io import digest, read_jsonl, utc_now, write_jsonl
from .schemas import Document, Paragraph

USER_AGENT = "BinanceDocTalk/0.1 (independent educational corpus collector)"
ALLOWED_HOSTS = {"www.binance.com", "academy.binance.com"}


def allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    return (parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS
            and parsed.port in {None, 443} and not parsed.username
            and ("/academy/" in parsed.path or "/support/faq/" in parsed.path))


class RestrictedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_url(newurl):
            raise ValueError("Source redirected outside the allowed public document paths")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class ArticleParser(HTMLParser):
    """Extract visible paragraphs after h1; preserve headings for source review."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.heading = "Introduction"
        self.paragraphs: list[dict] = []
        self.active: str | None = None
        self.parts: list[str] = []
        self.skip_depth = 0
        self.started = False
        self.stopped = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "footer", "aside"}:
            self.skip_depth += 1
        if not self.skip_depth and tag in {"h1", "h2", "h3", "h4", "p", "li"}:
            if self.active is None:
                self.active, self.parts = tag, []

    def handle_data(self, data):
        if self.active and not self.skip_depth:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "footer", "aside"} and self.skip_depth:
            self.skip_depth -= 1
        if tag != self.active:
            return
        text = re.sub(r"\s+", " ", "".join(self.parts)).strip()
        self.active = None
        if not text:
            return
        if tag == "h1":
            self.title, self.started = text, True
            return
        if not self.started or self.stopped:
            return
        if tag.startswith("h"):
            self.heading = text
            if text.casefold() in {"further reading", "related articles", "disclaimer"}:
                self.stopped = True
        elif len(text) >= 20:
            self.paragraphs.append({"paragraph_id": f"p{len(self.paragraphs)+1:03}",
                                    "heading": self.heading, "text": text})


def extract_html(html: str, source: dict, fetched_at: str | None = None) -> dict:
    parser = ArticleParser()
    parser.feed(html)
    includes = [x.casefold() for x in source.get("include_headings", [])]
    excludes = [x.casefold() for x in source.get("exclude_headings", [])]
    paragraphs = [p for p in parser.paragraphs
                  if (not includes or p["heading"].casefold() in includes)
                  and p["heading"].casefold() not in excludes]
    if not parser.title or not paragraphs:
        raise ValueError(f"{source['doc_id']}: no usable article; review HTML/heading selectors")
    return {
        "doc_id": source["doc_id"], "title": parser.title, "url": source["url"],
        "original_url": source["url"], "language": "en", "kind": "official",
        "fetched_at": fetched_at or utc_now(), "source_updated_at": None,
        "extractor_version": "visible-paragraphs-v1", "paragraphs": paragraphs,
        "content_hash": digest(paragraphs), "review_status": "pending_human_review",
        "include_headings": source.get("include_headings", []),
        "exclude_headings": source.get("exclude_headings", []),
    }


def collect(manifest: Path, output: Path, acknowledge_terms: bool,
            doc_ids: list[str] | None = None) -> dict:
    if not acknowledge_terms:
        raise ValueError("Read applicable terms and use --acknowledge-source-terms if permitted")
    if output.exists():
        raise ValueError("Collection output already exists; choose a new corpus version")
    sources = read_jsonl(manifest)
    if doc_ids:
        requested = set(doc_ids)
        if requested - {s["doc_id"] for s in sources}:
            raise ValueError("Unknown source ID")
        sources = [s for s in sources if s["doc_id"] in requested]
    opener = build_opener(RestrictedRedirect())
    documents, failures = [], []
    policies: dict[str, RobotFileParser] = {}
    for source in sources:
        try:
            if not allowed_url(source["url"]):
                raise ValueError("URL is outside the source allowlist")
            host = urlparse(source["url"]).hostname
            if host not in policies:
                policy = RobotFileParser()
                # A missing/unreadable policy is not treated as permission.
                request = Request(f"https://{host}/robots.txt", headers={"User-Agent": USER_AGENT})
                with build_opener().open(request, timeout=30) as response:
                    if urlparse(response.url).hostname != host:
                        raise ValueError("Unexpected robots policy redirect")
                    policy.parse(response.read(1_000_000).decode("utf-8").splitlines())
                policies[host] = policy
            if not policies[host].can_fetch(USER_AGENT, source["url"]):
                raise ValueError("robots.txt disallows this collector")
            request = Request(source["url"], headers={"User-Agent": USER_AGENT})
            with opener.open(request, timeout=30) as response:
                if response.status != 200:
                    raise ValueError(f"Source returned HTTP {response.status}; no article collected")
                if "text/html" not in response.headers.get("Content-Type", ""):
                    raise ValueError("Expected an HTML document")
                raw = response.read(4_000_001)
                if len(raw) > 4_000_000:
                    raise ValueError("Document exceeds collection size limit")
                document = extract_html(raw.decode("utf-8"), source)
                document["url"] = response.url
            expected = source.get("expected_content_hash")
            if expected and expected != document["content_hash"]:
                raise ValueError("Source content hash changed; create a new corpus version")
            documents.append(document)
        except Exception as exc:
            failures.append({"doc_id": source["doc_id"], "error_type": type(exc).__name__,
                             "message": str(exc)})
        time.sleep(2)
    if documents:
        write_jsonl(output, documents)
    return {"collected": len(documents), "failures": failures,
            "review_status": "pending_human_review"}


def load_documents(path: Path, allow_unreviewed: bool = False) -> list[Document]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Empty corpus")
    if len({r.get("doc_id") for r in rows}) != len(rows):
        raise ValueError("Duplicate document IDs")
    documents = []
    for row in rows:
        document = Document.from_dict(row)
        if digest([asdict(p) for p in document.paragraphs]) != document.content_hash:
            raise ValueError(f"Content hash mismatch: {document.doc_id}")
        if document.kind == "official":
            if not allowed_url(document.url):
                raise ValueError("Official document URL outside source allowlist")
            if not document.fetched_at:
                raise ValueError("Official source needs an actual collection timestamp")
            try:
                collected_at = datetime.fromisoformat(document.fetched_at)
            except (ValueError, TypeError):
                raise ValueError("Collection timestamp must be ISO 8601") from None
            if collected_at.utcoffset() is None:
                raise ValueError("Collection timestamp must include a timezone")
            if not allow_unreviewed and row.get("review_status") != "human_reviewed":
                raise ValueError("Official source needs human review before indexing")
            if not allow_unreviewed and (not row.get("reviewer") or not row.get("reviewed_at")):
                raise ValueError("Official human review needs a reviewer and timestamp")
        documents.append(document)
    if len({d.kind for d in documents}) != 1:
        raise ValueError("Do not mix synthetic demo notes with official documents")
    return documents
