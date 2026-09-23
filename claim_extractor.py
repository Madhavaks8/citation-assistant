"""
Claim Extraction Module:
Decomposes generated answers into discrete, verifiable atomic factual claims,
filtering out opinions and generating targeted search queries.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from llm import LLMClient
from config import config

logger = logging.getLogger("CitationAssistant.ClaimExtractor")


class Claim(BaseModel):
    """Structured representation of an extracted factual claim."""
    claim_id: str = Field(description="Unique identifier, e.g. C1, C2")
    claim_text: str = Field(description="The atomic factual assertion")
    original_sentence: str = Field(description="The source sentence in the generated answer")
    search_query: str = Field(description="Targeted scholarly search query to find supporting evidence")
    is_factual: bool = Field(default=True, description="True if verifiable factual claim, False if subjective")


class ClaimExtractor:
    """Extracts atomic factual claims and search queries from generated text."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def extract_claims(self, text: str, user_question: Optional[str] = None) -> List[Claim]:
        """
        Extract factual claims from the text using structured LLM reasoning
        with rule-based fallback.
        """
        if not text or not text.strip():
            return []

        # If LLM is mock, use rule-based extractor
        if self.llm.provider == "mock":
            return self._rule_based_extract(text)

        system_prompt = (
            "You are an expert academic research assistant specializing in fact extraction and citation grounding. "
            "Your task is to analyze an AI-generated answer, decompose it into individual atomic factual claims, "
            "and generate precise, search-engine-ready scholarly queries for each claim."
        )

        user_prompt = f"""
Analyze the following generated answer to the question: "{user_question or 'N/A'}"

Generated Answer:
\"\"\"{text}\"\"\"

Instructions:
1. Break down the text into distinct, atomic, verifiable factual claims.
2. Filter out non-verifiable opinions, vague rhetorical flourishes, and introductory filler.
3. For each claim, formulate a targeted search query (3-7 words) optimized for academic/authoritative sources.
4. Limit to the most important claims (maximum {config.MAX_CLAIMS_TO_PROCESS}).

Return a JSON array of objects with the exact schema:
[
  {{
    "claim_id": "C1",
    "claim_text": "Exact atomic factual statement",
    "original_sentence": "The sentence in the text where this claim appears",
    "search_query": "Targeted search query for academic/gov evidence",
    "is_factual": true
  }}
]
"""
        try:
            raw_claims = self.llm.generate_json(user_prompt, system_prompt=system_prompt)
            if isinstance(raw_claims, dict) and "claims" in raw_claims:
                raw_claims = raw_claims["claims"]

            claims: List[Claim] = []
            for i, item in enumerate(raw_claims):
                if i >= config.MAX_CLAIMS_TO_PROCESS:
                    break
                if isinstance(item, dict):
                    cid = item.get("claim_id") or f"C{i+1}"
                    ctext = item.get("claim_text", "").strip()
                    orig = item.get("original_sentence", ctext).strip()
                    query = item.get("search_query", ctext).strip()
                    is_fact = item.get("is_factual", True)
                    
                    if ctext and is_fact:
                        claims.append(
                            Claim(
                                claim_id=cid,
                                claim_text=ctext,
                                original_sentence=orig,
                                search_query=query,
                                is_factual=is_fact
                            )
                        )
            
            if claims:
                logger.info(f"Extracted {len(claims)} claims via LLM.")
                return claims

        except Exception as e:
            logger.warning(f"LLM claim extraction failed: {e}. Falling back to sentence decomposition.")

        return self._rule_based_extract(text)

    def _rule_based_extract(self, text: str) -> List[Claim]:
        """
        Deterministic sentence-based segmentation fallback when LLM is unavailable.
        """
        # Split text into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        claims: List[Claim] = []
        
        idx = 1
        for sentence in sentences:
            clean_s = sentence.strip()
            # Ignore short or rhetorical sentences
            if len(clean_s.split()) < 4 or clean_s.endswith("?"):
                continue

            # Generate a search query by removing stop words / punctuation
            query_words = [
                w for w in re.sub(r'[^\w\s]', '', clean_s).split()
                if w.lower() not in {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at", "by", "for", "with", "that", "this", "to", "it"}
            ]
            search_query = " ".join(query_words[:6])

            claims.append(
                Claim(
                    claim_id=f"C{idx}",
                    claim_text=clean_s,
                    original_sentence=clean_s,
                    search_query=search_query or clean_s,
                    is_factual=True,
                )
            )
            idx += 1
            if len(claims) >= config.MAX_CLAIMS_TO_PROCESS:
                break

        logger.info(f"Extracted {len(claims)} claims via rule-based segmenter.")
        return claims
