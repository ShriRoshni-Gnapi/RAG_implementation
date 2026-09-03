"""
Ingest ONE PT textbook: extract -> clean -> detect structure -> chunk -> metadata.

Writes:
    data/extracted/<book_id>_pages.json
    data/chunks/<book_id>_chunks.json
    reports/<book_id>_sample_chunks.txt
    reports/<book_id>_ingest_report.json

Run:  py -3.14 ingest_book.py paz_acute_care
"""

import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone

import pymupdf

from config import (
    BOOKS, BOOKS_DIR, EXTRACTED_DIR, CHUNKS_DIR, REPORTS_DIR,
    CHUNK_TARGET_TOKENS, CHUNK_MAX_TOKENS, CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS, CHUNK_MAX_CHARS, CHUNK_MIN_CHARS,
    HEADER_FOOTER_FREQ_THRESHOLD, HEADER_FOOTER_LINES, MIN_PAGE_CHARS,
    CORPUS_VERSION, CHUNK_CONFIG_ID, EMBEDDING_MODEL,
)

# The tokenizer of the ACTUAL embedding model. Chunk sizes are decided with
# this, so a chunk can never exceed what the model will read.
_TOK = None
_VOCAB = None


def get_tokenizer():
    global _TOK, _VOCAB
    if _TOK is None:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        except Exception:
            from sentence_transformers import SentenceTransformer
            _TOK = SentenceTransformer(EMBEDDING_MODEL).tokenizer
        _VOCAB = set(_TOK.get_vocab().keys())
    return _TOK


def n_tokens(text):
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def _known_word(w):
    get_tokenizer()
    return w in _VOCAB or ("##" + w) in _VOCAB

# --------------------------------------------------------------------------
import struct


# --------------------------------------------------------------------------
# 1. EXTRACTION (PDF, DOCX, DOC)
# --------------------------------------------------------------------------

def extract_pdf_pages(pdf_path):
    """Per-page text + page label + image count for PDF documents via PyMuPDF."""
    doc = pymupdf.open(pdf_path)
    try:
        labels = doc.get_page_labels()
        has_labels = bool(labels)
    except Exception:
        has_labels = False

    pages, failures = [], []
    for i in range(doc.page_count):
        rec = {"pdf_page": i, "page_label": None, "page_label_source": "index",
               "text": "", "n_images": 0, "status": "ok"}
        try:
            page = doc[i]
            rec["text"] = page.get_text("text") or ""
            try:
                rec["n_images"] = len(page.get_images(full=True))
            except Exception:
                rec["n_images"] = 0
            if has_labels:
                try:
                    lbl = page.get_label()
                    if lbl:
                        rec["page_label"] = lbl
                        rec["page_label_source"] = "label"
                except Exception:
                    pass
            if rec["page_label"] is None:
                rec["page_label"] = str(i + 1)
        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
            rec["page_label"] = str(i + 1)
            failures.append(i)
        pages.append(rec)

    toc = []
    try:
        toc = doc.get_toc(simple=True) or []
    except Exception:
        toc = []
    doc.close()
    return pages, toc, failures


def extract_docx_pages(docx_path):
    """Extract paragraphs, headings, and tables from modern .docx files via python-docx."""
    import docx
    doc = docx.Document(docx_path)
    pages, toc, failures = [], [], []
    current_page_text, current_char_count, page_num = [], 0, 1

    def flush_page():
        nonlocal page_num, current_page_text, current_char_count
        if current_page_text:
            text = "\n\n".join(current_page_text).strip()
            if text:
                pages.append({
                    "pdf_page": page_num - 1,
                    "page_label": str(page_num),
                    "page_label_source": "doc_section",
                    "text": text,
                    "n_images": 0,
                    "status": "ok",
                })
                page_num += 1
            current_page_text, current_char_count = [], 0

    for p in doc.paragraphs:
        txt = p.text.strip()
        if not txt:
            continue
        style_name = (p.style.name if p.style else "").lower()
        if "heading 1" in style_name or (txt.isupper() and 3 < len(txt) < 60):
            toc.append([1, txt, page_num])
            if current_char_count > 300:
                flush_page()
        elif "heading 2" in style_name or re.match(r"^(?:Station|Scenario|Case|Q:)\b", txt, re.I):
            toc.append([2, txt, page_num])
            if current_char_count > 800:
                flush_page()
        elif "heading 3" in style_name:
            toc.append([3, txt, page_num])

        current_page_text.append(txt)
        current_char_count += len(txt)
        if current_char_count >= 1800:
            flush_page()

    for table in doc.tables:
        rows_text = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                rows_text.append(" | ".join(cells))
        if rows_text:
            tbl_txt = "\n".join(rows_text)
            current_page_text.append(tbl_txt)
            current_char_count += len(tbl_txt)
            if current_char_count >= 1800:
                flush_page()

    flush_page()
    if not pages:
        pages.append({"pdf_page": 0, "page_label": "1", "page_label_source": "doc_section",
                      "text": "", "n_images": 0, "status": "ok"})
    return pages, toc, failures


def extract_doc_pages(doc_path):
    """Extract text and tables from legacy Word 97-2003 .doc binary files via OLE2 piece-table parser."""
    import olefile
    ole = olefile.OleFileIO(str(doc_path))
    try:
        word_doc = ole.openstream("WordDocument").read()
        flags = struct.unpack_from("<H", word_doc, 0x000A)[0]
        table_stream_name = "1Table" if (flags & 0x0200) else "0Table"
        fcClx = struct.unpack_from("<I", word_doc, 0x01A2)[0]
        lcbClx = struct.unpack_from("<I", word_doc, 0x01A6)[0]
        table_stream = ole.openstream(table_stream_name).read()
        clx_data = table_stream[fcClx : fcClx + lcbClx]
        pos = 0
        text_pieces = []
        while pos < len(clx_data):
            clxt = clx_data[pos]
            if clxt == 1:
                cbGrpprl = struct.unpack_from("<H", clx_data, pos + 1)[0]
                pos += 3 + cbGrpprl
            elif clxt == 2:
                lcbPlcfpcd = struct.unpack_from("<I", clx_data, pos + 1)[0]
                pcd_data = clx_data[pos + 5 : pos + 5 + lcbPlcfpcd]
                n = (len(pcd_data) - 4) // 12
                cps = [struct.unpack_from("<I", pcd_data, i * 4)[0] for i in range(n + 1)]
                pcds = pcd_data[(n + 1) * 4 :]
                for i in range(n):
                    cp_start, cp_end = cps[i], cps[i + 1]
                    fc_val = struct.unpack_from("<I", pcds, i * 8 + 2)[0]
                    is_compressed = bool(fc_val & (1 << 30))
                    fc = fc_val & ~(1 << 30) & ~(1 << 31)
                    char_len = cp_end - cp_start
                    if is_compressed:
                        raw_bytes = word_doc[fc // 2 : fc // 2 + char_len]
                        piece = raw_bytes.decode("cp1252", errors="replace")
                    else:
                        raw_bytes = word_doc[fc : fc + char_len * 2]
                        piece = raw_bytes.decode("utf-16le", errors="replace")
                    text_pieces.append(piece)
                break
            else:
                break
    finally:
        ole.close()

    raw = "".join(text_pieces)
    clean = raw.replace("\x07\x07", "\n").replace("\x07", " | ").replace("\r\n", "\n").replace("\r", "\n")

    pages, toc, failures = [], [], []
    page_num, current_page_text, current_char_count = 1, [], 0

    def flush_page():
        nonlocal page_num, current_page_text, current_char_count
        if current_page_text:
            text = "\n\n".join(current_page_text).strip()
            if text:
                pages.append({
                    "pdf_page": page_num - 1,
                    "page_label": str(page_num),
                    "page_label_source": "doc_section",
                    "text": text,
                    "n_images": 0,
                    "status": "ok",
                })
                page_num += 1
            current_page_text, current_char_count = [], 0

    sections = re.split(r"\x0c+|\n{3,}", clean)
    for sec in sections:
        lines = [l.strip() for l in sec.split("\n") if l.strip()]
        for line in lines:
            if (line.isupper() and 3 < len(line) < 60) or re.match(
                r"^(?:Table|Procedure|Contraindication|Precaution|Position|Purpose|Description)\b", line, re.I
            ):
                toc.append([1, line, page_num])
                if current_char_count > 300:
                    flush_page()
            current_page_text.append(line)
            current_char_count += len(line)
            if current_char_count >= 1800:
                flush_page()
        if current_char_count > 400:
            flush_page()

    flush_page()
    if not pages:
        pages.append({"pdf_page": 0, "page_label": "1", "page_label_source": "doc_section",
                      "text": "", "n_images": 0, "status": "ok"})
    return pages, toc, failures


def extract_pages(file_path):
    """Per-page/section text + label + TOC for PDF, DOCX, or DOC."""
    lower = str(file_path).lower()
    if lower.endswith(".docx"):
        return extract_docx_pages(file_path)
    elif lower.endswith(".doc"):
        return extract_doc_pages(file_path)
    return extract_pdf_pages(file_path)



# --------------------------------------------------------------------------
# 2. STRUCTURE (Tier A: outline-driven)
# --------------------------------------------------------------------------

def build_structure_map(toc, n_pages):
    """
    Map every pdf page -> {chapter, section, subsection, path}.

    toc entries are [level, title, 1-based page]. We keep an ancestor stack so
    each entry knows its full path, then assign each page the LAST entry that
    starts at or before it.
    """
    entries, stack = [], []
    for level, title, page1 in toc:
        title = re.sub(r"\s+", " ", (title or "")).strip()
        if not title:
            continue
        while len(stack) >= level:
            stack.pop()
        while len(stack) < level - 1:
            stack.append(None)
        stack.append(title)
        path = [t for t in stack if t]
        entries.append({
            "level": level,
            "title": title,
            "pdf_page": max(0, (page1 or 1) - 1),
            "chapter": path[0] if len(path) >= 1 else None,
            "section": path[1] if len(path) >= 2 else None,
            "subsection": path[-1] if len(path) >= 3 else None,
            "path": path,
        })
    entries.sort(key=lambda e: (e["pdf_page"], e["level"]))

    page_map, cur, idx = {}, None, 0
    for p in range(n_pages):
        while idx < len(entries) and entries[idx]["pdf_page"] <= p:
            cur = entries[idx]
            idx += 1
        page_map[p] = cur
    return page_map, entries


def classify_content_type(node, has_outline=True):
    """
    references / front_matter / prose, from the structural path.

    `has_outline` matters. When a book HAS an outline, a page that maps to no
    entry sits before the first heading - cover, title page, copyright - so
    front_matter is right. When a book has NO outline at all, every page maps
    to None, and calling all of it front_matter would exclude the entire book
    from the index. Tier-C books (e.g. SCI Functional Status) hit exactly that:
    112 chunks produced, 0 indexable. Unknown structure means unknown, not
    front matter, so those default to prose.
    """
    if node is None:
        return "front_matter" if has_outline else "prose"
    joined = " > ".join(node["path"]).lower()
    if re.search(r"\breferences?\b|\bbibliograph|\bfurther reading\b", joined):
        return "references"
    if re.search(r"\bindex\b|\bcopyright\b|\bpreface\b|\bcontributors?\b|"
                 r"\bdedication\b|\backnowledg|\btable of contents\b|\bfront cover\b|"
                 r"\btitle page\b|\bforeword\b", joined):
        return "front_matter"
    return "prose"


# --------------------------------------------------------------------------
# 3. CLEANING
# --------------------------------------------------------------------------

def _norm_pattern(line):
    return re.sub(r"\d+", "#", line.strip())[:60]


def find_boilerplate(pages):
    """Frequency-based running head/foot detection. No hard-coded strings."""
    top, bot = Counter(), Counter()
    considered = 0
    for p in pages:
        lines = [l for l in p["text"].split("\n") if l.strip()]
        if len(lines) < 4:
            continue
        considered += 1
        for l in lines[:HEADER_FOOTER_LINES]:
            top[_norm_pattern(l)] += 1
        for l in lines[-HEADER_FOOTER_LINES:]:
            bot[_norm_pattern(l)] += 1
    if considered == 0:
        return set(), set(), {}
    cut = considered * HEADER_FOOTER_FREQ_THRESHOLD
    tp = {k for k, v in top.items() if v >= cut and k}
    bp = {k for k, v in bot.items() if v >= cut and k}
    stats = {"pages_considered": considered,
             "top_patterns": top.most_common(8),
             "bottom_patterns": bot.most_common(8)}
    return tp, bp, stats


LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
             "ﬃ": "ffi", "ﬄ": "ffl", "ﬆ": "st"}


RUNNING_HEAD_WINDOW = 3     # this book's running head is up to 3 physical lines


def _norm_title(s):
    """'13 Infectious Diseases' -> 'infectious diseases' (for head matching)."""
    s = re.sub(r"^[\d\.\s]+", "", (s or "").strip())
    return re.sub(r"\s+", " ", s).strip().lower()


def _is_running_head(line, page_label, node_titles):
    """
    True only for lines that are provably page furniture:
      a) the bare printed page number for THIS page
      b) a 'CHAPTER 13' / 'PART 1' / 'SECTION 2' marker
      c) a short line equal to this page's own chapter/section title
    All three are anchored to known values (the page label or the TOC), so
    this cannot delete body text that merely happens to be short.
    """
    s = line.strip()
    if not s or len(s) > 70 or re.search(r"[.!?;,]$", s):
        return False
    if s == str(page_label):                                   # (a)
        return True
    # (b) 'CHAPTER 13' alone, or 'CHAPTER 1  Acute Care Setting [3]' combined
    m = re.match(r"^(?:CHAPTER|PART|SECTION|APPENDIX)\s+[\dIVXLC]+\s*(.*)$", s, re.I)
    if m:
        rest = re.sub(r"\s+\d{1,4}$", "", m.group(1)).strip()
        if not rest or _norm_title(rest) in node_titles:
            return True
    if _norm_title(s) and _norm_title(s) in node_titles:        # (c)
        return True
    # combined forms: 'Infectious Diseases 325' / '325 Infectious Diseases'
    m = re.match(r"^(.+?)\s+(\d{1,4})$", s)
    if m and m.group(2) == str(page_label):
        return True
    m = re.match(r"^(\d{1,4})\s+(.+)$", s)
    if m and m.group(1) == str(page_label):
        return True
    return False


def _join_broken_word(prev, nxt):
    """
    Rejoin a word split across a line break where the PDF dropped the hyphen
    ('pul' + 'monary' -> 'pulmonary'). Applied ONLY at a line boundary, and
    only when the joined form is a known word and the fragments are not both
    words on their own. Returns the joined string, or None to keep the space.
    """
    ma = re.search(r"([A-Za-z]{2,})$", prev)
    mb = re.match(r"^([a-z]{2,})", nxt)
    if not (ma and mb):
        return None
    a, b = ma.group(1), mb.group(1)
    if not a[-1].islower():
        return None
    if _known_word(a) and _known_word(b):
        return None                       # both real words - leave alone
    if _known_word(a + b):
        return prev + nxt
    return None


def clean_page(text, page_label, node_titles, top_pat, bot_pat, removal_log):
    """Strip boilerplate + page numbers, de-hyphenate, rejoin wrapped lines."""
    text = unicodedata.normalize("NFC", text)
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    text = text.replace("­", "")
    # em/en/figure spaces and tabs -> plain space (this book's running heads
    # are padded with U+2003 EM SPACE, which plain \s handling missed)
    text = re.sub(r"[ -   　	]", " ", text)

    lines = [l.rstrip() for l in text.split("\n")]
    lines = [l for l in lines if l.strip()]

    # frequency-detected boilerplate at the extremes
    for _ in range(HEADER_FOOTER_LINES):
        if lines and _norm_pattern(lines[0]) in top_pat:
            removal_log.append(("header_freq", lines[0][:70])); lines.pop(0)
        else:
            break
    for _ in range(HEADER_FOOTER_LINES):
        if lines and _norm_pattern(lines[-1]) in bot_pat:
            removal_log.append(("footer_freq", lines[-1][:70])); lines.pop()
        else:
            break

    # page-number / TOC-title anchored running heads and feet.
    # This book prints them as up to 3 separate lines
    # ('CHAPTER 13' / 'Infectious Diseases' / '325'), so scan a 3-line window
    # from each end and keep removing while lines qualify.
    for _ in range(RUNNING_HEAD_WINDOW):
        if lines and _is_running_head(lines[0], page_label, node_titles):
            removal_log.append(("running_head", lines[0][:70])); lines.pop(0)
        else:
            break
    for _ in range(RUNNING_HEAD_WINDOW):
        if lines and _is_running_head(lines[-1], page_label, node_titles):
            removal_log.append(("running_foot", lines[-1][:70])); lines.pop()
        else:
            break

    # isolated page-number lines at the extremes (never mid-page numbers)
    while lines and re.fullmatch(r"[ivxlcIVXLC\d]{1,6}", lines[0].strip()):
        removal_log.append(("pageno", lines[0][:70])); lines.pop(0)
    while lines and re.fullmatch(r"[ivxlcIVXLC\d]{1,6}", lines[-1].strip()):
        removal_log.append(("pageno", lines[-1][:70])); lines.pop()

    # rejoin wrapped lines into paragraphs
    out, buf = [], ""
    bullet = re.compile(r"^\s*([•\-\*▪·]|\(?\d{1,2}[\.\)]|[a-z]\))\s+")
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        starts_new = bool(bullet.match(s)) or (s.isupper() and len(s) < 70)
        if buf and not starts_new:
            if buf.endswith("-") and re.match(r"^[a-z]", s):
                buf = buf[:-1] + s                        # explicit hyphen
            elif re.search(r"[.!?:;]$", buf):
                out.append(buf); buf = s
            else:
                joined = _join_broken_word(buf, s)        # dropped hyphen
                buf = joined if joined else buf + " " + s
        else:
            if buf:
                out.append(buf)
            buf = s
    if buf:
        out.append(buf)

    cleaned = "\n".join(out)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


# --------------------------------------------------------------------------
# 4. CHUNKING (section-aware, never mixes sections)
# --------------------------------------------------------------------------

def _hard_split_tokens(text, limit):
    """Last resort: cut on token count, at a word boundary. Logged by caller."""
    tok = get_tokenizer()
    words, out, cur = text.split(), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if len(tok.encode(cand, add_special_tokens=False)) > limit and cur:
            out.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def split_block(text, stats):
    """Paragraph pieces, each guaranteed <= CHUNK_MAX_TOKENS."""
    parts = []
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        if n_tokens(para) <= CHUNK_MAX_TOKENS:
            parts.append(para)
            continue
        # too big: split on sentences
        cur = ""
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", para):
            cand = (cur + " " + s).strip()
            if n_tokens(cand) <= CHUNK_MAX_TOKENS:
                cur = cand
            else:
                if cur:
                    parts.append(cur)
                if n_tokens(s) > CHUNK_MAX_TOKENS:
                    stats["hard_splits"] += 1
                    parts.extend(_hard_split_tokens(s, CHUNK_MAX_TOKENS))
                    cur = ""
                else:
                    cur = s
        if cur:
            parts.append(cur)
    return parts


def chunk_section(text, stats):
    """
    Accumulate paragraphs up to CHUNK_TARGET_TOKENS with a token-based
    overlap. Never exceeds CHUNK_MAX_TOKENS, so nothing is ever truncated
    by the embedding model.
    """
    pieces = split_block(text, stats)
    chunks, cur = [], ""
    for piece in pieces:
        cand = (cur + " " + piece).strip() if cur else piece
        if not cur:
            cur = piece
        elif n_tokens(cand) <= CHUNK_TARGET_TOKENS:
            cur = cand
        else:
            chunks.append(cur)
            # overlap: carry the tail words of the previous chunk
            words = cur.split()
            tail = ""
            for i in range(len(words) - 1, -1, -1):
                t = " ".join(words[i:])
                if n_tokens(t) > CHUNK_OVERLAP_TOKENS:
                    tail = " ".join(words[i + 1:])
                    break
            cand2 = (tail + " " + piece).strip() if tail else piece
            cur = cand2 if n_tokens(cand2) <= CHUNK_MAX_TOKENS else piece
    if cur:
        chunks.append(cur)

    # merge an orphan tail chunk back into its predecessor when it fits
    if len(chunks) >= 2 and n_tokens(chunks[-1]) < CHUNK_MIN_TOKENS:
        merged = chunks[-2] + " " + chunks[-1]
        if n_tokens(merged) <= CHUNK_MAX_TOKENS:
            chunks[-2] = merged
            chunks.pop()
    return chunks


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main(book_key):
    meta = BOOKS[book_key]
    pdf_path = BOOKS_DIR / meta["source_file"]
    if not pdf_path.exists():
        raise SystemExit(f"Source file not found: {pdf_path}")

    print(f"Book       : {meta['title']}")
    print(f"File       : {meta['source_file']}")
    print(f"Size       : {pdf_path.stat().st_size/1e6:.1f} MB\n")

    print("[1/5] Extracting ...")
    pages, toc, failures = extract_pages(str(pdf_path))
    raw_chars = sum(len(p["text"]) for p in pages)
    near_empty = [p["pdf_page"] for p in pages
                  if p["status"] == "ok" and len(p["text"]) < MIN_PAGE_CHARS]
    label_pages = sum(1 for p in pages if p["page_label_source"] == "label")
    print(f"      pages={len(pages)}  raw_chars={raw_chars:,}  toc_entries={len(toc)}")
    print(f"      failed_pages={len(failures)}  near_empty={len(near_empty)}  "
          f"page_label_resolved={label_pages}/{len(pages)}")

    print("[2/5] Detecting structure ...")
    page_map, entries = build_structure_map(toc, len(pages))
    depth = max([e["level"] for e in entries], default=0)
    print(f"      outline entries={len(entries)}  max_depth={depth}")

    print("[3/5] Cleaning ...")
    top_pat, bot_pat, hf_stats = find_boilerplate(pages)
    print(f"      boilerplate patterns: {len(top_pat)} header / {len(bot_pat)} footer")
    removal_log = []
    for p in pages:
        if p["status"] != "ok":
            p["clean_text"] = ""
            continue
        node = page_map.get(p["pdf_page"])
        # every title on this page's structural path, normalised, so a running
        # head that repeats the chapter/section name can be recognised
        node_titles = {_norm_title(t) for t in (node["path"] if node else [])}
        node_titles.discard("")
        p["clean_text"] = clean_page(p["text"], p["page_label"], node_titles,
                                     top_pat, bot_pat, removal_log)
    clean_chars = sum(len(p["clean_text"]) for p in pages)
    print(f"      clean_chars={clean_chars:,} "
          f"(removed {raw_chars-clean_chars:,} = {100*(raw_chars-clean_chars)/max(1,raw_chars):.1f}%)")
    print(f"      lines removed: {len(removal_log)}")

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXTRACTED_DIR / f"{book_key}_pages.json", "w", encoding="utf-8") as f:
        json.dump({"book": meta, "toc_entries": len(entries), "pages": pages},
                  f, ensure_ascii=False)

    print("[4/5] Chunking (section-aware) ...")
    # group CONSECUTIVE pages sharing the same structural node
    blocks, cur = [], None
    for p in pages:
        if not p["clean_text"]:
            continue
        node = page_map.get(p["pdf_page"])
        key = tuple(node["path"]) if node else ("<front matter>",)
        if cur and cur["key"] == key:
            cur["texts"].append(p["clean_text"])
            cur["page_end"] = p["page_label"]
            cur["pdf_end"] = p["pdf_page"]
        else:
            if cur:
                blocks.append(cur)
            cur = {"key": key, "node": node, "texts": [p["clean_text"]],
                   "page_start": p["page_label"], "page_end": p["page_label"],
                   "pdf_start": p["pdf_page"], "pdf_end": p["pdf_page"],
                   "label_source": p["page_label_source"]}
    if cur:
        blocks.append(cur)

    chunks, seq = [], 0
    stats = {"hard_splits": 0}
    now = datetime.now(timezone.utc).isoformat()
    for b in blocks:
        node = b["node"]
        ctype = classify_content_type(node, has_outline=bool(entries))
        for piece in chunk_section("\n".join(b["texts"]), stats):
            if len(piece) < CHUNK_MIN_CHARS:
                continue
            seq += 1
            path = node["path"] if node else []
            chunks.append({
                "chunk_id": f"{book_key}_{seq:05d}",
                "book": meta["title"],
                "book_id": book_key,
                "author": meta["author"],
                "edition": meta["edition"],
                "chapter": (node or {}).get("chapter"),
                "section": (node or {}).get("section"),
                "subsection": (node or {}).get("subsection"),
                "section_path": " > ".join(path) if path else None,
                "page_start": b["page_start"],
                "page_end": b["page_end"],
                "pdf_page_start": b["pdf_start"],
                "pdf_page_end": b["pdf_end"],
                "page_label_source": b["label_source"],
                "content_type": ctype,
                "subject_area": meta["subject_area"],
                "text": piece,
                "n_chars": len(piece),
                "n_tokens": n_tokens(piece),
                "source_file": meta["source_file"],
                "corpus_version": CORPUS_VERSION,
                "chunk_config_id": CHUNK_CONFIG_ID,
                "extraction_method": "text",
                "created_at": now,
            })

    sizes = [c["n_chars"] for c in chunks]
    toks = [c["n_tokens"] for c in chunks]
    by_type = Counter(c["content_type"] for c in chunks)
    print(f"      blocks={len(blocks)}  chunks={len(chunks)}  "
          f"hard_splits={stats['hard_splits']}")
    print(f"      chars : avg={sum(sizes)//max(1,len(sizes))} "
          f"min={min(sizes) if sizes else 0} max={max(sizes) if sizes else 0}")
    print(f"      tokens: avg={sum(toks)//max(1,len(toks))} "
          f"min={min(toks) if toks else 0} max={max(toks) if toks else 0}")
    print(f"      by content_type: {dict(by_type)}")

    # HARD GATE (blueprint 10.5): refuse to proceed if anything would be
    # silently truncated by the embedding model.
    from config import TOKEN_SAFETY_LIMIT
    over = [c["chunk_id"] for c in chunks if c["n_tokens"] > TOKEN_SAFETY_LIMIT]
    if over:
        raise SystemExit(
            f"ABORT: {len(over)} chunks exceed {TOKEN_SAFETY_LIMIT} tokens and "
            f"would be silently truncated. First: {over[:5]}")
    print(f"      OK: all {len(chunks)} chunks <= {TOKEN_SAFETY_LIMIT} tokens")

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_DIR / f"{book_key}_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)

    print("[5/5] Writing sample + report ...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prose = [c for c in chunks if c["content_type"] == "prose"]
    step = max(1, len(prose) // 10)
    sample = prose[::step][:10]
    with open(REPORTS_DIR / f"{book_key}_sample_chunks.txt", "w", encoding="utf-8") as f:
        f.write(f"SAMPLE CHUNKS - {meta['title']}\n")
        f.write(f"{len(prose)} prose chunks; showing 10 spread evenly\n")
        f.write("=" * 78 + "\n\n")
        for c in sample:
            f.write(f"chunk_id   : {c['chunk_id']}\n")
            f.write(f"chapter    : {c['chapter']}\n")
            f.write(f"section    : {c['section']}\n")
            f.write(f"subsection : {c['subsection']}\n")
            f.write(f"pages      : {c['page_start']}-{c['page_end']} "
                    f"(pdf {c['pdf_page_start']}-{c['pdf_page_end']}, "
                    f"src={c['page_label_source']})\n")
            f.write(f"chars/toks : {c['n_chars']} / {c['n_tokens']}\n")
            f.write("-" * 78 + "\n")
            f.write(c["text"] + "\n")
            f.write("=" * 78 + "\n\n")

    report = {
        "book": meta, "generated_at": now,
        "pages_total": len(pages), "pages_failed": len(failures),
        "pages_near_empty": len(near_empty),
        "page_label_resolved": label_pages,
        "raw_chars": raw_chars, "clean_chars": clean_chars,
        "removed_chars": raw_chars - clean_chars,
        "boilerplate_lines_removed": len(removal_log),
        "header_footer_stats": hf_stats,
        "toc_entries": len(entries), "toc_max_depth": depth,
        "blocks": len(blocks), "chunks_total": len(chunks),
        "chunks_by_content_type": dict(by_type),
        "hard_splits": stats["hard_splits"],
        "chunk_chars_avg": sum(sizes)//max(1, len(sizes)),
        "chunk_chars_min": min(sizes) if sizes else 0,
        "chunk_chars_max": max(sizes) if sizes else 0,
        "chunk_tokens_avg": sum(toks)//max(1, len(toks)),
        "chunk_tokens_min": min(toks) if toks else 0,
        "chunk_tokens_max": max(toks) if toks else 0,
        "removal_log_by_kind": dict(Counter(k for k, _ in removal_log)),
        "config": {"target_tokens": CHUNK_TARGET_TOKENS,
                   "max_tokens": CHUNK_MAX_TOKENS,
                   "min_tokens": CHUNK_MIN_TOKENS,
                   "overlap_tokens": CHUNK_OVERLAP_TOKENS},
    }
    with open(REPORTS_DIR / f"{book_key}_ingest_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print(f"\nDone. {len(chunks)} chunks -> {CHUNKS_DIR / (book_key + '_chunks.json')}")


# Public name for pipeline orchestration (see main.py). Same function -
# main() is kept so the existing CLI behaviour is unchanged.
ingest_book = main


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: py -3.14 ingest_book.py <book_key>")
        print("To see available book keys, run: py -3.14 main.py --list")
        sys.exit(2 if len(sys.argv) < 2 else 0)
    main(sys.argv[1])

