"""
Source Validation & Entailment Module:
Performs Natural Language Inference (NLI) and evidence grounding to determine whether
retrieved sources substantiate, partially substantiate, or fail to support factual claims.
"""

import re
import logging
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from llm import LLMClient
from claim_extractor import Claim
from source_search import SourceDocument
from config import config

logger = logging.getLogger("CitationAssistant.SourceValidator")


class VerificationStatus(str, Enum):
    SUPPORTED = "Supported"
    PARTIALLY_SUPPORTED = "Partially Supported"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence"
    UNSUPPORTED = "Unsupported"


class ValidationResult(BaseModel):
    """Structured validation outcome for a single claim."""
    claim_id: str
    claim_text: str
    status: VerificationStatus
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanation of why evidence supports or fails to support the claim")
    supporting_sources: List[SourceDocument] = Field(default_factory=list)


class SourceValidator:
    """Evaluates claim-evidence entailment and assigns verifiable status."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def validate_claim(
        self,
        claim: Claim,
        candidate_sources: List[SourceDocument]
    ) -> ValidationResult:
        """
        Validate whether candidate sources substantiate the claim.
        """
        if not candidate_sources:
            return ValidationResult(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence_score=0.2,
                reasoning="No relevant sources could be retrieved for this claim.",
                supporting_sources=[],
            )

        # If in mock mode, use heuristic entailment
        if self.llm.provider == "mock":
            return self._heuristic_validate(claim, candidate_sources)

        # Build prompt with candidate evidence snippets
        sources_context = "\n".join([
            f"Source [{doc.source_id}] (Title: {doc.title}, Publisher: {doc.publisher}):\nSnippet: \"{doc.snippet}\""
            for doc in candidate_sources
        ])

        system_prompt = (
            "You are an impartial academic fact-checker and citation auditor. "
            "Your objective is to determine whether the provided evidence snippets substantiate a factual claim. "
            "You must assign one of the following exact statuses:\n"
            "- Supported: The evidence directly confirms and grounds the claim.\n"
            "- Partially Supported: The evidence confirms core aspects but omits specific nuances or metrics.\n"
            "- Insufficient Evidence: The snippets mention the topic but lack specific data to prove the claim.\n"
            "- Unsupported: The evidence contradicts the claim or provides zero substantiation."
        )

        user_prompt = f"""
Factual Claim to Evaluate:
\"\"\"{claim.claim_text}\"\"\"

Retrieved Evidence Sources:
{sources_context}

Evaluate the entailment between the claim and the retrieved sources.

Respond with JSON format:
{{
  "status": "Supported", // One of: "Supported", "Partially Supported", "Insufficient Evidence", "Unsupported"
  "confidence_score": 0.95, // Float between 0.0 and 1.0
  "reasoning": "Clear 1-2 sentence explanation describing how the evidence substantiates or fails to substantiate the claim.",
  "supported_source_ids": ["{candidate_sources[0].source_id}"] // IDs of sources that genuinely support the claim
}}
"""
        try:
            val_data = self.llm.generate_json(user_prompt, system_prompt=system_prompt)
            raw_status = val_data.get("status", "Insufficient Evidence")
            
            # Map to Enum
            status_enum = VerificationStatus.INSUFFICIENT_EVIDENCE
            for member in VerificationStatus:
                if member.value.lower() == raw_status.lower():
                    status_enum = member
                    break

            confidence = float(val_data.get("confidence_score", 0.7))
            reasoning = val_data.get("reasoning", "Evidence evaluated via LLM verification.")
            supported_ids = set(val_data.get("supported_source_ids", []))

            # Filter matching supporting sources
            matched_sources = [
                doc for doc in candidate_sources
                if doc.source_id in supported_ids or (status_enum == VerificationStatus.SUPPORTED and len(candidate_sources) == 1)
            ]
            if not matched_sources and status_enum in (VerificationStatus.SUPPORTED, VerificationStatus.PARTIALLY_SUPPORTED):
                matched_sources = [candidate_sources[0]]

            return ValidationResult(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                status=status_enum,
                confidence_score=confidence,
                reasoning=reasoning,
                supporting_sources=matched_sources,
            )

        except Exception as e:
            logger.warning(f"LLM validation failed: {e}. Using heuristic validator.")
            return self._heuristic_validate(claim, candidate_sources)

    def _heuristic_validate(self, claim: Claim, sources: List[SourceDocument]) -> ValidationResult:
        """Semantic overlap entailment fallback for offline mock testing."""
        claim_words = set(re.findall(r'\w+', claim.claim_text.lower()))
        best_overlap = 0.0
        best_source = sources[0] if sources else None

        for doc in sources:
            snippet_words = set(re.findall(r'\w+', (doc.snippet + " " + doc.title).lower()))
            overlap = len(claim_words.intersection(snippet_words)) / max(len(claim_words), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_source = doc

        if best_overlap >= 0.4:
            status = VerificationStatus.SUPPORTED
            confidence = min(0.92, 0.6 + best_overlap * 0.4)
            reason = f"The source from {best_source.publisher} directly discusses key concepts and aligns with the assertion."
            supporting = [best_source]
        elif best_overlap >= 0.2:
            status = VerificationStatus.PARTIALLY_SUPPORTED
            confidence = 0.65
            reason = f"The source from {best_source.publisher} provides relevant context, though full claim details require deeper verification."
            supporting = [best_source]
        else:
            status = VerificationStatus.INSUFFICIENT_EVIDENCE
            confidence = 0.40
            reason = "Retrieved excerpts provide insufficient context to firmly substantiate this claim."
            supporting = []

        return ValidationResult(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            status=status,
            confidence_score=confidence,
            reasoning=reason,
            supporting_sources=supporting,
        )
