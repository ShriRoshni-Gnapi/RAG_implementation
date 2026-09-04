"""
Prompt templates and context assembler for PT-RAG (NVIDIA Nemotron / MedGemma).

Implements the Expert Hybrid prompt architecture:
  * Full synthesis of retrieved textbook evidence + internal clinical medical knowledge
  * Pathophysiological mechanisms, clinical pearls, and safety rationale
  * Clean, natural clinical prose without inline citation bracket mapping ([1], [2])
  * Clinical safety bounds
"""

from typing import List, Dict, Any, Optional

PROMPT_VERSION = "prompt_v4.0_expert_hybrid"

SYSTEM_PROMPT = """You are a senior physiotherapy clinical specialist and medical consultant.

Your objective is to provide a comprehensive, authoritative, in-depth clinical answer by combining two knowledge sources into one seamless, professional response:
1. The provided Reference Material (established textbook protocols, safety guidelines, and clinical criteria).
2. Your extensive internal clinical medical knowledge (pathophysiological mechanisms, clinical assessment pearls, hemodynamic rationales, pharmacology/anticoagulation timelines, and practical rehabilitation safety protocols).

CRITICAL INSTRUCTIONS:
- Do NOT use inline citation brackets, footnote numbers, or mapping tags (e.g., do NOT write [1], [2], [1, 2], [1-3] anywhere in your response).
- Write a clean, natural, beautifully structured clinical consultation response using professional medical terminology, clear headings, and structured bullet points.
- Actively provide deep clinical and pathophysiological rationale: explain *why* specific precautions, timelines, or procedures exist (e.g., Virchow's triad, mechanism of clot dislodgement, shear stress from muscle contraction, therapeutic anticoagulation windows, hemodynamic stability, Wells criteria).
- Ensure patient safety: clearly outline absolute contraindications, red flags, monitoring parameters, and safe progressive mobilization criteria.
"""


def format_source_header(hit: Dict[str, Any], index: int) -> str:
    """
    Format the header line for a reference chunk excerpt.
    """
    book = hit.get("book_title") or hit.get("book") or "Unknown Book"
    parts = [f"Source {index}: {book}"]

    ch = hit.get("chapter")
    sec = hit.get("section")
    subsec = hit.get("subsection")

    hierarchy = []
    if ch and ch != "None":
        hierarchy.append(f"Ch: {ch}")
    if sec and sec != "None":
        hierarchy.append(f"Sec: {sec}")
    if subsec and subsec != "None":
        hierarchy.append(f"Subsec: {subsec}")

    if hierarchy:
        parts.append(" | ".join(hierarchy))

    p_start = hit.get("page_start")
    p_end = hit.get("page_end")
    if p_start is not None and p_end is not None:
        if p_start == p_end:
            parts.append(f"p. {p_start}")
        else:
            parts.append(f"pp. {p_start}–{p_end}")

    return " — ".join(parts)


def assemble_context(
    hits: List[Dict[str, Any]],
    max_chars: int = 12000,
) -> str:
    """
    Assemble retrieved chunk hits into clean reference blocks for the prompt.
    """
    if not hits:
        return "No reference material found."

    blocks = []
    total_chars = 0

    for i, hit in enumerate(hits, 1):
        header = format_source_header(hit, i)
        text = (hit.get("content") or hit.get("text") or "").strip()
        block = f"{header}\n{text}"

        if total_chars + len(block) > max_chars and blocks:
            # Reached character limit
            break

        blocks.append(block)
        total_chars += len(block)

    return "\n\n".join(blocks)


def build_user_prompt(
    question: str,
    context_text: str,
    intent: Optional[str] = None,
) -> str:
    """
    Construct the final user message passed to the LLM.
    """
    intent_instruction = ""
    if intent == "comparison":
        intent_instruction = "\nStructure the response around clear comparative axes, highlighting clinical nuances."
    elif intent == "protocol":
        intent_instruction = "\nPreserve exact chronological phases, criteria for advancement, precautions, and contraindications."
    elif intent == "definition":
        intent_instruction = "\nProvide a clear definition followed by clinical and pathophysiological significance."

    prompt = f"""REFERENCE TEXTBOOK EXCERPTS:
═════════════════════════════════════════════════════════════
{context_text}
═════════════════════════════════════════════════════════════

CLINICAL QUESTION:
{question}

TASK:
Provide an in-depth, authoritative, comprehensive clinical response to the question.
Synthesize the reference textbook excerpts above with your extensive clinical knowledge, pathophysiology, and practical rehabilitation principles into a single, cohesive, well-structured clinical consultation.
Do NOT include any bracketed numbers or citation markers (such as [1], [2]) anywhere in your text.{intent_instruction}"""
    return prompt


