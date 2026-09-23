"""
Citation Engine Module:
Injects in-text citation markers ([1], [2]) into generated text and generates
scholarly bibliographies in APA 7th, MLA 9th, and IEEE formats, plus BibTeX export.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field

from source_search import SourceDocument
from source_validator import ValidationResult, VerificationStatus

logger = logging.getLogger("CitationAssistant.CitationEngine")


class FormattedCitation(BaseModel):
    """Structured citation reference item."""
    citation_number: int
    source: SourceDocument
    formatted_text: str
    style: str


class CitationResult(BaseModel):
    """Complete output of citation injection and formatting."""
    cited_answer: str
    references: List[FormattedCitation]
    raw_references_text: str
    bibtex_export: str
    total_sources_cited: int


class CitationEngine:
    """Handles in-text citation injection and multi-style bibliography generation."""

    def __init__(self, default_style: str = "APA"):
        self.default_style = default_style.upper()

    def process_citations(
        self,
        original_text: str,
        validation_results: List[ValidationResult],
        style: Optional[str] = None
    ) -> CitationResult:
        """
        Inject citation markers into text and format bibliography references.
        """
        citation_style = (style or self.default_style).upper()

        # Step 1: Deduplicate unique sources and assign citation indices [1], [2], ...
        unique_sources: List[SourceDocument] = []
        source_url_to_num: Dict[str, int] = {}
        claim_to_citation_nums: Dict[str, List[int]] = {}

        for val in validation_results:
            # Only attach citations to Supported or Partially Supported claims
            if val.status in (VerificationStatus.SUPPORTED, VerificationStatus.PARTIALLY_SUPPORTED):
                claim_nums = []
                for src in val.supporting_sources:
                    if src.url not in source_url_to_num:
                        citation_idx = len(unique_sources) + 1
                        source_url_to_num[src.url] = citation_idx
                        unique_sources.append(src)
                        claim_nums.append(citation_idx)
                    else:
                        claim_nums.append(source_url_to_num[src.url])
                if claim_nums:
                    claim_to_citation_nums[val.claim_id] = sorted(list(set(claim_nums)))

        # Step 2: In-text citation injection into sentences
        cited_text = self._inject_intext_citations(original_text, validation_results, claim_to_citation_nums)

        # Step 3: Format references list according to chosen style
        formatted_refs: List[FormattedCitation] = []
        ref_lines: List[str] = []

        for idx, src in enumerate(unique_sources, start=1):
            formatted_entry = self._format_reference_entry(src, idx, citation_style)
            formatted_refs.append(
                FormattedCitation(
                    citation_number=idx,
                    source=src,
                    formatted_text=formatted_entry,
                    style=citation_style,
                )
            )
            ref_lines.append(formatted_entry)

        raw_references_text = "\n\n".join(ref_lines)
        bibtex_text = self._generate_bibtex(unique_sources)

        return CitationResult(
            cited_answer=cited_text,
            references=formatted_refs,
            raw_references_text=raw_references_text,
            bibtex_export=bibtex_text,
            total_sources_cited=len(unique_sources),
        )

    def _inject_intext_citations(
        self,
        text: str,
        validations: List[ValidationResult],
        claim_to_nums: Dict[str, List[int]]
    ) -> str:
        """
        Injects numeric citations like [1] or [1, 2] directly following supported statements.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        augmented_sentences: List[str] = []

        for sentence in sentences:
            sentence_clean = sentence.strip()
            citation_markers_for_sentence = []

            for val in validations:
                if val.claim_id in claim_to_nums:
                    # Check if claim or part of claim resides in this sentence
                    claim_words = set(re.findall(r'\w+', val.claim_text.lower()))
                    sentence_words = set(re.findall(r'\w+', sentence_clean.lower()))
                    overlap = len(claim_words.intersection(sentence_words)) / max(len(claim_words), 1)

                    if overlap >= 0.5:
                        citation_markers_for_sentence.extend(claim_to_nums[val.claim_id])

            if citation_markers_for_sentence:
                unique_nums = sorted(list(set(citation_markers_for_sentence)))
                marker_str = f" [{', '.join(map(str, unique_nums))}]"
                
                # Place marker before terminal punctuation if present
                if sentence_clean and sentence_clean[-1] in ".!?":
                    punct = sentence_clean[-1]
                    augmented = sentence_clean[:-1] + marker_str + punct
                else:
                    augmented = sentence_clean + marker_str
                augmented_sentences.append(augmented)
            else:
                augmented_sentences.append(sentence_clean)

        return " ".join(augmented_sentences)

    def _format_reference_entry(self, source: SourceDocument, num: int, style: str) -> str:
        """Format an individual citation entry into APA, MLA, or IEEE."""
        author_or_org = source.author or source.publisher or "Author Unknown"
        year = source.publication_year or "n.d."
        title = source.title
        url = source.url

        if style == "APA":
            # APA 7th Edition: Author/Org. (Year). Title. Publisher. URL
            return f"[{num}] {author_or_org}. ({year}). *{title}*. {source.publisher}. [{url}]({url})"

        elif style == "MLA":
            # MLA 9th Edition: Author/Org. "Title." Publisher, Year, URL.
            return f"[{num}] {author_or_org}. \"{title}.\" *{source.publisher}*, {year}, [{url}]({url})."

        elif style == "IEEE":
            # IEEE: [1] Author/Org, "Title," Publisher, Year. [Online]. Available: URL
            return f"[{num}] {author_or_org}, \"{title},\" *{source.publisher}*, {year}. [Online]. Available: [{url}]({url})"

        else:
            return f"[{num}] {author_or_org} ({year}). {title}. Available: {url}"

    def _generate_bibtex(self, sources: List[SourceDocument]) -> str:
        """Generate BibTeX entries for academic users."""
        entries = []
        for i, src in enumerate(sources, start=1):
            key = f"source{i}_{re.sub(r'[^a-zA-Z0-9]', '', (src.publisher or 'ref'))[:8].lower()}"
            entry = (
                f"@misc{{{key},\n"
                f"  title = {{{src.title}}},\n"
                f"  author = {{{src.author or src.publisher}}},\n"
                f"  publisher = {{{src.publisher}}},\n"
                f"  year = {{{src.publication_year or '2024'}}},\n"
                f"  url = {{{src.url}}},\n"
                f"  note = {{Accessed via Citation Assistant}}\n"
                f"}}"
            )
            entries.append(entry)
        return "\n\n".join(entries)
