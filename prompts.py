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

    author = hit.get("author")
    if author:
        parts.append(f"Author: {author}")

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


# --------------------------------------------------------------------------
# SOAP note -> textbook references (reference-retrieval mode)
# --------------------------------------------------------------------------

REFERENCE_SYSTEM_PROMPT = """You are a Physiotherapy Textbook Reference Retrieval Assistant.
Your task is to analyse the complete SOAP note and identify relevant references from the retrieved physiotherapy textbook passages.
Instructions:

1. Read the complete Subjective, Objective and Assessment sections.
2. Identify important clinical terms, symptoms, examination findings and assessment concepts.
3. Match these concepts against the retrieved textbook passages.
4. Return only references directly relevant to the complete SOAP note.
5. For each reference identify the relevant supporting sentences of the passage by their numbers. The system adds the book title, author, chapter, section and page number, and prints those sentences.
6. Do not mention which SOAP section a reference belongs to.
7. Do not generate diagnoses, treatment plans, clinical advice or additional explanations.
8. Do not use your own medical knowledge to create references.
9. Never invent book titles, authors, chapters or page numbers.
10. If no relevant reference is found, explicitly state that no matching reference was found.

Output only a combined list of relevant textbook references."""

NO_REFERENCE_TEXT = "No matching reference was found."
NO_SUMMARY_TEXT = "No supported summary was available for the retrieved references."

# At most this many sentences are printed per source. The model is asked to stay
# within it and extract_source_sentences enforces it. Raise it if a real run
# shows good sentences being left out.
MAX_SENTENCES_PER_SOURCE = 3


def assemble_numbered_context(hits: List[Dict[str, Any]], sentences: List[List[Any]]) -> str:
    """
    Reference blocks with every sentence numbered, for sentence picking.

    `sentences` holds passages.Sentence items. The same numbering is used to
    print the picked sentences, so a number always means the same text.
    """
    blocks = []
    for i, (hit, sents) in enumerate(zip(hits, sentences), 1):
        lines = "\n".join(f"  ({k}) {s.text}" for k, s in enumerate(sents, 1))
        blocks.append(f"{format_source_header(hit, i)}\n{lines}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Detailed summary of a cited passage
# --------------------------------------------------------------------------

INSUFFICIENT_TEXT = "INSUFFICIENT"

SUMMARY_SYSTEM_PROMPT = """You explain physiotherapy textbook passages.
Write a detailed summary of the retrieved passage you are given, using only that passage.

1. Read the complete passage before writing.
2. Explain the main concept and the important supporting points.
3. Describe the clinical or technical meaning of the passage in clear, simple language.
4. Explain terminology only where the passage itself explains it.
5. Explain how the different points in the passage are connected.
6. Preserve the author's intended meaning.
7. Do not use general medical knowledge, other books, assumptions or anything the passage does not state.
8. Do not copy the original sentences; explain them in your own words.
9. Write a well-structured explanation of two to four short paragraphs, not a one-line summary.
10. If the passage does not give enough information to explain a point, leave that point out. Never guess.
11. Do not define or explain a term unless the passage itself defines or explains it. Do not add what a condition, test or structure "is".
12. State what the passage says. Do not draw conclusions from it: no sentences that begin "This suggests", "This means", "This highlights", "This ensures", "These indicate" or similar."""


def build_summary_user_prompt(numbered_passage: str, focus: Optional[List[int]] = None) -> str:
    """
    User message for the detailed summary of one citation's passage(s).

    The model must end every paragraph with the numbers of the passage
    sentences it is based on. The caller checks those numbers and the
    paragraph against the passage and drops any paragraph it cannot support;
    the numbers themselves are never printed.

    `focus` lists the sentences that were chosen as relevant to the note. The
    summary is written about their ideas and the text that directly explains
    them, not about every other topic in the retrieved chunk.
    """
    focus_text = ""
    if focus:
        numbers = ", ".join(str(k) for k in focus)
        focus_text = (
            f"\nSentences {numbers} were chosen as the ones that matter for this note. Write the summary about "
            "the ideas in those sentences and the parts of the passage that directly explain them. Leave out "
            "parts of the passage about other conditions, procedures or patient groups."
        )
    return f"""RETRIEVED PASSAGE (every sentence is numbered):
═════════════════════════════════════════════════════════════
{numbered_passage}
═════════════════════════════════════════════════════════════
{focus_text}
Write the detailed summary now. Put each paragraph on its own line and end it with the numbers of the passage sentences it is based on, in square brackets, for example [2, 5]. Every paragraph must end with such numbers. Use only facts and terms that appear in the passage, and no headings, bullet points or other text. If the passage does not contain enough information for a meaningful summary, reply only: {INSUFFICIENT_TEXT}"""


def build_reference_user_prompt(soap_note: str, context_text: str) -> str:
    """
    User message for reference mode: the numbered passages plus the SOAP note.

    The model only returns "Source N: sentence numbers". The caller prints
    each source's book details from stored metadata and the picked sentences
    exactly as stored, so nothing in the list is written by the model and
    nothing from outside the books can appear.
    """
    return f"""RETRIEVED TEXTBOOK PASSAGES:
═════════════════════════════════════════════════════════════
{context_text}
═════════════════════════════════════════════════════════════

SOAP NOTE:
{soap_note}

Return the combined list of relevant references. Write one line per reference: the source label exactly as written above, then the numbers of that source's sentences that directly support a finding in the note, separated by commas (for example "Source 3: 2, 4"). Write only the label and the numbers: no book details and no other words.
Be strict. Include a source only if its sentences are about the same condition, body region and kind of finding as the note. Leave out a source that merely shares a term (for example "stairs", "swelling" or "pain") but describes a different condition, patient group or procedure. A short list is better than a long one.
Pick at most {MAX_SENTENCES_PER_SOURCE} sentences per source, most relevant first. Do not pick a sentence that: gives advice on treatment or management; states a general fact not tied to a finding in the note (for example norms for sex or age); or only makes sense after the sentence before it (it begins with "This", "These", "It" or "However"). Each picked sentence must support a finding by itself.
If none of the passages is relevant, reply only: {NO_REFERENCE_TEXT}"""


