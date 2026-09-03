"""
Settings for the PT-book RAG pipeline.

Values are deliberately aligned with the existing transcript RAG project
(../../rag/config.py) so both use the same embedding model and vector store.
Only the things that genuinely differ for books are changed here.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_DIR = Path(__file__).resolve().parent
_PARENT = PROJECT_DIR.parent

# Source PDFs live in ../books/ if that folder exists, otherwise directly in
# the parent directory. Override with the PT_BOOKS_DIR environment variable.
def _resolve_books_dir():
    import os as _os
    override = _os.environ.get("PT_BOOKS_DIR")
    if override:
        return Path(override)
    candidate = _PARENT / "books"
    return candidate if candidate.is_dir() else _PARENT


BOOKS_DIR = _resolve_books_dir()
DATA_DIR = PROJECT_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
CHUNKS_DIR = DATA_DIR / "chunks"
SQL_DIR = PROJECT_DIR / "sql"
REPORTS_DIR = PROJECT_DIR / "reports"

# ---------------------------------------------------------------- env loading
# Credentials come from the environment, never from this file. A local .env is
# loaded if present; real environment variables always win over .env values.
ENV_FILE = PROJECT_DIR / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE, override=False)
except ImportError:      # python-dotenv absent: plain env vars still work
    pass


def _env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


# ---------------------------------------------------------------- vector db
# Qdrant, running as a SERVER on localhost (not the embedded local-path mode).
# This matches the sibling transcript project (../../rag/config.py), which also
# talks to a Qdrant server over HTTP.
QDRANT_URL = _env("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = _env("QDRANT_API_KEY")        # unset for a local server
QDRANT_TIMEOUT = int(_env("QDRANT_TIMEOUT", "60"))
COLLECTION_PREFIX = _env("COLLECTION_PREFIX", "ptbook_")

# Qdrant builds its HNSW index automatically; no index type to choose.
DISTANCE_METRIC = "cosine"

# ---------------------------------------------------------------- embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"       # same model as the transcript project
EMBEDDING_DIM = 384
# HARD LIMIT of all-MiniLM-L6-v2. Anything longer is SILENTLY truncated.
EMBEDDING_MAX_TOKENS = 256
TOKEN_SAFETY_LIMIT = 250                   # assert every chunk stays under this
EMBED_BATCH_SIZE = 64

# ---------------------------------------------------------------- chunking
# Chunking is TOKEN-aware, not character-aware.
#
# A first run sized chunks by characters (target 1100) assuming ~0.15 tokens
# per character. Measured on this book that assumption was wrong: medical
# textbook prose runs 0.239 tokens/char on average and up to 0.423, so 33.5%
# of chunks blew past the 256-token limit and would have been silently
# truncated. Chunk budgets are therefore expressed in TOKENS.
CHUNK_TARGET_TOKENS = 200
CHUNK_MAX_TOKENS = 250          # hard ceiling, under EMBEDDING_MAX_TOKENS (256)
CHUNK_MIN_TOKENS = 40
CHUNK_OVERLAP_TOKENS = 30

# Character bounds are kept only as cheap pre-filters before tokenizing.
CHUNK_MAX_CHARS = 2000
CHUNK_MIN_CHARS = 150

# ---------------------------------------------------------------- cleaning
HEADER_FOOTER_FREQ_THRESHOLD = 0.30        # pattern on >30% of pages = boilerplate
HEADER_FOOTER_LINES = 2                    # inspect first/last N lines per page
MIN_PAGE_CHARS = 100                       # below this a page is "near-empty"

# Content types excluded from the vector index (kept in the chunk file).
EXCLUDED_CONTENT_TYPES = ("references", "front_matter")

# ---------------------------------------------------------------- corpus
CORPUS_VERSION = "corpus_v1_2026-09-03"
CHUNK_CONFIG_ID = "c1100o150_section_aware"

# ---------------------------------------------------------------- book registry
#
# The registry is DISCOVERED, not hand-listed: every supported file in
# BOOKS_DIR becomes a book automatically. Drop a new PDF into ../books/ and it
# is immediately available - no code change needed.
#
#   py -3.14 main.py --list      shows what was discovered
#   py -3.14 main.py --all       ingests everything
#
# BOOK_META below is an OPTIONAL overlay: it supplies the bibliographic detail
# that cannot be inferred from a filename (author, edition, year) plus a stable
# short book_id. Anything not listed there still works, just with a generated
# id and a title derived from the filename.
#
# Why the overlay matters: every citation the system emits uses these values.
# Fields are left None where they have NOT been verified from the title page -
# null is honest, a guessed citation is not.

import re

# Extensions the ingestion pipeline can read: PDFs via PyMuPDF, DOCX via python-docx, DOC via OLE2 parser.
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".doc")
UNSUPPORTED_EXTENSIONS = ()


# structure_tier records the measured PDF-outline quality:
#   A = rich outline (chapter + section, often deeper)
#   B = chapter-level outline only
#   C = no usable outline; expect book+page citations without chapter names
# It is descriptive metadata; discovery defaults it to "?" for unknown books.

BOOK_META = {
    # filename -> curated metadata
    "Acute Care Handbook for Physical Therapists - Paz, Jaime C. [SRG].pdf": dict(
        id="paz_acute_care", title="Acute Care Handbook for Physical Therapists",
        author="Jaime C. Paz", subject_area="acute_care", tier="A"),

    # ---- orthopaedics / musculoskeletal ----
    "Orthopedic Physical Assessment - David J. Magee [SRG].pdf": dict(
        id="magee_ortho_assessment", title="Orthopedic Physical Assessment",
        author="David J. Magee", subject_area="orthopaedics", tier="A",
        notes="1294 pages, two-column; running head carries the chapter name"),
    "Clinical_Orthpedic_Rehabilitation,_2nd_edition.pdf": dict(
        id="clinical_ortho_rehab", title="Clinical Orthopaedic Rehabilitation",
        edition="2nd", subject_area="orthopaedics", tier="C"),
    "Handbook-of-Orthopedic-Rehabilitation.pdf": dict(
        id="handbook_ortho_rehab", title="Handbook of Orthopedic Rehabilitation",
        subject_area="orthopaedics", tier="B"),
    "orthopedic physicaltherapy secrets.pdf": dict(
        id="ortho_pt_secrets", title="Orthopedic Physical Therapy Secrets",
        subject_area="orthopaedics", tier="B"),
    "physiotherapy in orthopaedics  problem solving approach.pdf": dict(
        id="physio_orthopaedics",
        title="Physiotherapy in Orthopaedics: A Problem-Solving Approach",
        subject_area="orthopaedics", tier="B"),
    "Muscles Testing & Function with Posture & Pain.pdf": dict(
        id="kendall_muscles",
        title="Muscles: Testing and Function with Posture and Pain",
        author="Kendall et al.", subject_area="orthopaedics", tier="B"),
    "Musculoskeletal Assessment - Joint Motion and Muscle Testing (3rd Ed.).pdf": dict(
        id="musculoskeletal_assessment",
        title="Musculoskeletal Assessment: Joint Motion and Muscle Testing",
        edition="3rd", subject_area="orthopaedics", tier="B"),
    "Physical-Therapy-Protocols.pdf": dict(
        id="pt_protocols", title="Physical Therapy Protocols",
        subject_area="orthopaedics", tier="B",
        notes="image-heavy: ~21% of sampled pages had <100 chars"),

    # ---- therapeutic exercise ----
    "Therapeutic Exercise-Carolyn Kischner 5th 2007.pdf": dict(
        id="kisner_therapeutic_exercise", title="Therapeutic Exercise",
        author="Carolyn Kisner", edition="5th", year=2007,
        subject_area="therapeutic_exercise", tier="B"),

    # ---- cardiopulmonary ----
    "Cardiovascular-and-Pulmonary-Physical-Therapy-Watchie-Joanne-SRG.pdf": dict(
        id="watchie_cardiopulm",
        title="Cardiovascular and Pulmonary Physical Therapy",
        author="Joanne Watchie", subject_area="cardiopulmonary", tier="A"),
    "Principles and Practice of Cardiopulmonary Physical Therapy.pdf": dict(
        id="cardiopulm_principles",
        title="Principles and Practice of Cardiopulmonary Physical Therapy",
        subject_area="cardiopulmonary", tier="B"),
    "Clinical Management Notes and Case Histories in Cardiopulmonary Physical Therapy.pdf": dict(
        id="clinical_mgmt_cardiopulm",
        title="Clinical Management Notes and Case Histories in "
              "Cardiopulmonary Physical Therapy",
        subject_area="cardiopulmonary", tier="B"),
    "ECG Notes.pdf": dict(
        id="ecg_notes", title="ECG Notes",
        subject_area="cardiopulmonary", tier="C",
        notes="LOW YIELD: ~39% of sampled pages had <100 chars (image-based). "
              "Review its chunks by hand before trusting retrieval from it."),

    # ---- neurological rehabilitation ----
    "Umphred_s Neurological Rehabilitation - Umphred, Darcy A. [SRG].pdf": dict(
        id="umphred_neuro_rehab", title="Umphred's Neurological Rehabilitation",
        author="Darcy A. Umphred", subject_area="neuro", tier="A"),
    "Susan Sullivan.pdf": dict(
        id="sullivan_physical_rehab", title="Physical Rehabilitation",
        author="Susan B. O'Sullivan", subject_area="neuro", tier="A"),
    "Improving-Functional-Outcomes-in-Physical-Rehabilitation.pdf": dict(
        id="improving_functional_outcomes",
        title="Improving Functional Outcomes in Physical Rehabilitation",
        subject_area="neuro", tier="B"),
    "SCI Functional Status.pdf": dict(
        id="sci_functional_status", title="SCI Functional Status",
        subject_area="neuro", tier="C",
        notes="no PDF outline at all; expect book+page citations only"),
    "Canadian-Stroke-Best-Practice-Recommendations-Webinar-archive-1.pdf": dict(
        id="canadian_stroke",
        title="Canadian Stroke Best Practice Recommendations (Webinar Archive)",
        subject_area="neuro", tier="C",
        notes="9 pages; slide-deck layout extracts poorly"),

    # ---- differential diagnosis ----
    "Differential Diagnosis.pdf": dict(
        id="diff_dx", title="Differential Diagnosis",
        subject_area="differential_dx", tier="C",
        notes="932 pages. POSSIBLE DUPLICATE of diff_dx_pt - compare title "
              "pages and drop one if they are the same work."),
    "Differential Diagnosis in Physical Therapy.pdf": dict(
        id="diff_dx_pt", title="Differential Diagnosis in Physical Therapy",
        subject_area="differential_dx", tier="C",
        notes="470 pages. POSSIBLE DUPLICATE of diff_dx."),

    # ---- general / case-based ----
    "Case Studies in Physiotherapy.pdf": dict(
        id="case_studies_physio", title="Case Studies in Physiotherapy",
        subject_area="general", tier="A"),

    # ---- clinical notes & stations (Word documents) ----
    "Cardio-respiratory stations (1) copy.docx": dict(
        id="cardiorespiratory_stations",
        title="Cardio-Respiratory Clinical Stations and Case Scenarios",
        subject_area="cardiopulmonary", tier="B"),
    "Postural drainage.doc": dict(
        id="postural_drainage",
        title="Postural Drainage Guidelines and Procedures",
        subject_area="cardiopulmonary", tier="B"),
    "ortho fracture rehab.doc": dict(
        id="ortho_fracture_rehab",
        title="Orthopaedic Fracture Rehabilitation Guide",
        author="Dr. Mark McAndrew",
        subject_area="orthopaedics", tier="B"),
}



def _slug(stem, maxlen=44):
    """Filename stem -> a stable, safe book_id for unregistered books."""
    s = stem.lower()
    s = re.sub(r"\[[^\]]*\]", " ", s)          # drop [SRG] style tags
    s = re.sub(r"\(\s*\d+\s*\)|\bcopy\b", " ", s)   # drop "(1)", "copy"
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    s = re.sub(r"_+", "_", s)
    return s[:maxlen].strip("_") or "book"


def _title_from(stem):
    """Readable fallback title when the book is not in BOOK_META."""
    t = re.sub(r"\[[^\]]*\]", " ", stem)
    t = re.sub(r"[_]+", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" -,")
    return t or stem


def discover_books(books_dir=None):
    """
    Scan BOOKS_DIR and build the registry.

    Every supported file becomes an entry. Curated metadata from BOOK_META is
    applied when the filename matches; otherwise sensible values are derived
    from the filename so a newly added book works immediately.
    """
    d = Path(books_dir) if books_dir else BOOKS_DIR
    found = {}
    if not d.is_dir():
        return found
    for path in sorted(d.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        meta = BOOK_META.get(path.name, {})
        key = meta.get("id") or _slug(path.stem)
        if key in found:                       # never let two files collide
            key = f"{key}_{abs(hash(path.name)) % 1000:03d}"
        found[key] = {
            "book_id": key,
            "title": meta.get("title") or _title_from(path.stem),
            "author": meta.get("author"),
            "edition": meta.get("edition"),
            "year": meta.get("year"),
            "source_file": path.name,
            "subject_area": meta.get("subject_area", "unclassified"),
            "structure_tier": meta.get("tier", "?"),
            "notes": meta.get("notes"),
            "curated": bool(meta),
        }
    return found


def unsupported_files(books_dir=None):
    """Files present but not ingestible - reported, never silently skipped."""
    d = Path(books_dir) if books_dir else BOOKS_DIR
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in UNSUPPORTED_EXTENSIONS)


BOOKS = discover_books()
