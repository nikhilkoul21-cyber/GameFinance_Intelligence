"""Parse a raw SEC filing (HTML) into clean text, split by 'Item' section.

SEC filings are long HTML documents (often inline-XBRL) full of tags, tables
and boilerplate. For RAG we want plain readable text, and we want to know
WHICH section each piece came from — "Item 1A. Risk Factors", "Item 7. MD&A",
etc. — because section is a powerful retrieval and citation signal.

Section detection here is deliberately pragmatic (regex on the visible text),
not a perfect legal parser. It is good enough to label the big, important
sections and can be refined later — an honest v1 trade-off.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass(frozen=True)
class Section:
    """A labelled span of clean text from a filing."""
    item: str      # e.g. "Item 1A. Risk Factors", or "PREAMBLE" if before first item
    text: str


# Matches headings like "Item 1A.", "ITEM 7.", "Item 1B" at a line start.
# We capture the item number/letter so we can label the section. The title
# capture is unbounded (`.*`) — a fixed cap used to make long real headings
# (e.g. "Item 7. Management's Discussion and Analysis...", 86+ chars) fail to
# match at all, silently merging that whole section into whatever came before.
# This only works because html_to_text() isolates each heading on its own
# line; without that, an unbounded capture would swallow unrelated text.
_ITEM_RE = re.compile(
    r"(?im)^\s*item\s+(\d{1,2}[A-Z]?)\.?\s*[:\-—]?\s*(.*)$"
)

# The standard SEC "forward-looking statements" legal disclaimer, almost
# always right after the cover page. Phrasing varies by company (TTWO/EA:
# "CAUTIONARY NOTE ABOUT..."; RBLX: "SPECIAL NOTE REGARDING...") but it's
# always this one ALL-CAPS heading alone on its own line. See
# _strip_forward_looking_disclaimer for why this needs to be removed, not
# just left to rank itself out.
_FWD_LOOKING_HEADING_RE = re.compile(
    r"(?im)^(?:CAUTIONARY|SPECIAL)\s+(?:NOTE|STATEMENT)\s+"
    r"(?:ABOUT|REGARDING|CONCERNING)\s+FORWARD[\s-]LOOKING\s+STATEMENTS\s*$"
)
# Where the disclaimer block ends: the next "PART n" or "Item n." heading.
_FWD_LOOKING_STOP_RE = re.compile(r"(?im)^(PART\s+[IVX]+\b|ITEM\s+\d{1,2}[A-Z]?\.)")

# Table of contents tables list every "Item N." heading in one cell each,
# right next to a page number — if left in, their "Item N." cells are
# indistinguishable from real section headings and get split out as (empty,
# duplicate) sections ahead of the real ones. Real headings live in their own
# <div>, never inside a <table>, so any table containing several of these is
# unambiguously the ToC and carries no information we need.
_TOC_CELL_RE = re.compile(r"^\s*item\s+\d", re.I)
_TOC_MIN_HITS = 5

# Common 10-K/10-Q item titles, for nicer labels when the heading text is thin.
_KNOWN_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for Registrant's Common Equity",
    "7": "Management's Discussion and Analysis (MD&A)",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
}


def _drop_hidden_and_toc(soup: BeautifulSoup) -> None:
    """Remove content that is invisible or isn't prose: XBRL tagging data and
    the table of contents (see _TOC_CELL_RE above for why the ToC must go)."""
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Inline-XBRL filings wrap every tagged fact — including a full duplicate
    # of EVERY numeric/text fact in the document — in display:none elements
    # (most commonly one big <div style="display:none"><ix:header>...). Left
    # in, this becomes thousands of characters of "ttwo-20250331\n0000946581\n
    # false\n..." ahead of the real text.
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    for table in soup.find_all("table"):
        cells = table.find_all(["td", "th"])
        hits = sum(1 for c in cells if _TOC_CELL_RE.match(c.get_text(strip=True)))
        if hits >= _TOC_MIN_HITS:
            table.decompose()


def html_to_text(html: bytes | str) -> str:
    """Strip HTML to readable plain text.

    Removes script/style/hidden-XBRL/ToC, then inserts our own line breaks at
    block-tag boundaries (div/p/tr/li/br/h1-h6) and " | " between table cells,
    rather than letting BeautifulSoup's get_text(separator=...) insert a break
    between EVERY pair of tags. That default behaviour is what used to put a
    heading's title on its own line regardless of what it's inside — a word
    sitting in its own inline <span> or <a> (e.g. a cross-reference like
    "Refer to <a>Item 1 - Business</a> for details") came out as 3 separate
    lines, with "Item 1 - Business" alone on one, indistinguishable from a
    real heading. It also split every table cell onto its own line, so a
    row like "Total net revenue | $5,633.6 | $5,349.6" lost the association
    between each number and its column/row entirely. Inserting separators
    only at real block/cell boundaries keeps each sentence and each table row
    intact as one line, which is what the section splitter and the chunker
    both assume.
    """
    soup = BeautifulSoup(html, "lxml")
    _drop_hidden_and_toc(soup)

    for cell in soup.find_all(["td", "th"]):
        cell.append(" | ")
    for tag in soup.find_all(["div", "p", "tr", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6"]):
        tag.append("\n")

    text = soup.get_text(separator="")
    # Normalise unicode spaces and collapse runs of blank lines.
    text = text.replace("\xa0", " ")

    lines = []
    for ln in text.splitlines():
        ln = ln.strip().strip("|").strip()
        ln = re.sub(r"\s*\|\s*", " | ", ln)   # tidy cell separators
        ln = re.sub(r"(\s*\|\s*){2,}", " | ", ln)  # collapse empty-cell runs
        if ln:
            lines.append(ln)
    return "\n".join(lines)


def split_into_sections(text: str) -> list[Section]:
    """Split clean filing text into sections at each 'Item N.' heading."""
    matches = list(_ITEM_RE.finditer(text))
    if not matches:
        # No item headings found (common for short 8-Ks) — treat whole doc as one.
        return [Section(item="FULL_DOCUMENT", text=text)]

    sections: list[Section] = []

    # Anything before the first item heading (cover page etc.).
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append(Section(item="PREAMBLE", text=pre))

    for idx, m in enumerate(matches):
        num = m.group(1).upper()
        heading_tail = (m.group(2) or "").strip(" .:-—")
        title = heading_tail or _KNOWN_TITLES.get(num, "")
        label = f"Item {num}. {title}".strip().rstrip(".")

        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append(Section(item=label, text=body))

    return sections


def parse_filing(html: bytes | str) -> list[Section]:
    """Full parse: raw HTML -> list of labelled Sections."""
    return split_into_sections(html_to_text(html))
