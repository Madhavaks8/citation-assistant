"""
Claim Extraction Module:
Decomposes generated answers into discrete, verifiable atomic factual claims,
filters out opinions, classifies claim types, and generates multiple high-precision
claim-specific search queries.
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
    search_query: str = Field(default="", description="Primary targeted scholarly search query")
    search_queries: List[str] = Field(default_factory=list, description="2-3 targeted search query variants")
    claim_type: str = Field(default="factual", description="Type of claim: factual, numerical, statistical, causal, definitional, comparative, historical")
    is_factual: bool = Field(default=True, description="True if verifiable factual claim, False if subjective")

    def model_post_init(self, __context: Any) -> None:
        """Ensure search_query and search_queries remain synchronized."""
        if not self.search_queries and self.search_query:
            self.search_queries = [self.search_query]
        elif self.search_queries and not self.search_query:
            self.search_query = self.search_queries[0]


class ClaimExtractor:
    """Extracts atomic factual claims, metadata, and 2-3 precision search queries from generated text."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def extract_claims(self, text: str, user_question: Optional[str] = None) -> List[Claim]:
        """
        Extract atomic factual claims from the text using structured LLM reasoning
        with rule-based fallback.
        """
        if not text or not text.strip():
            return []

        # If LLM is mock, use enhanced rule-based extractor
        if self.llm.provider == "mock":
            return self._rule_based_extract(text)

        system_prompt = (
            "You are an expert academic research assistant specializing in fact extraction and citation grounding. "
            "Your task is to analyze an AI-generated answer, decompose it into individual ATOMIC factual claims, "
            "classify each claim's type, and generate 2-3 precise, search-engine-ready scholarly queries for each claim.\n\n"
            "CRITICAL RULES FOR ATOMIC CLAIM EXTRACTION:\n"
            "1. An atomic claim must express ONE independently verifiable factual proposition.\n"
            "2. If a sentence asserts multiple distinct, separable facts (e.g. 'AI is used for personalized learning and automated grading in schools'), "
            "decompose it into separate claims ('AI is used for personalized learning in schools', 'AI is used for automated grading in schools').\n"
            "3. If a sentence describes a single unified mechanism or tightly connected facts (e.g. 'Large language models are trained using self-supervised learning on large text datasets'), "
            "keep it as a single claim. Do NOT split tightly connected facts unnecessarily.\n"
            "4. NEVER invent or hallucinate facts that were not stated in the generated answer.\n"
            "5. Filter out subjective opinions, conversational filler, and rhetorical questions.\n"
            "6. Classify claim_type as one of: 'factual', 'numerical', 'statistical', 'causal', 'definitional', 'comparative', 'historical'.\n"
            "7. For each claim, generate 2-3 distinct, highly targeted search queries (4-8 words each):\n"
            "   - Query 1: Direct technical terms and entities.\n"
            "   - Query 2: Scholarly synonyms or academic phrasing.\n"
            "   - Query 3: Mechanism/evidence context phrasing.\n"
            "   - Queries must retain specific entities, numbers, dates, technical terms, mechanisms, and outcomes.\n"
            "   - DO NOT collapse queries into generic topics (e.g., do NOT generate 'AI education' or 'artificial intelligence')."
        )

        user_prompt = f"""
Analyze the following generated answer to the question: "{user_question or 'N/A'}"

Generated Answer:
\"\"\"{text}\"\"\"

Instructions:
Decompose the text into atomic factual claims (maximum {config.MAX_CLAIMS_TO_PROCESS}).
For each claim, provide 2-3 specific search queries and metadata.

Return a JSON array of objects with the exact schema:
[
  {{
    "claim_id": "C1",
    "claim_text": "Exact atomic factual statement",
    "original_sentence": "The source sentence from the text",
    "claim_type": "factual",
    "search_queries": [
      "query variant 1 with specific technical entities",
      "query variant 2 with scholarly synonyms",
      "query variant 3 with mechanism or evidence terms"
    ],
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
                    ctype = item.get("claim_type", "factual").strip().lower()
                    is_fact = item.get("is_factual", True)
                    
                    # Extract queries list
                    queries = item.get("search_queries", [])
                    if isinstance(queries, str):
                        queries = [queries]
                    elif not isinstance(queries, list):
                        queries = []
                    
                    if not queries and "search_query" in item:
                        queries = [item["search_query"]]

                    # Filter and clean queries
                    clean_queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
                    if not clean_queries:
                        clean_queries = self._generate_query_variants_for_claim(ctext)

                    if ctext and is_fact:
                        claims.append(
                            Claim(
                                claim_id=cid,
                                claim_text=ctext,
                                original_sentence=orig,
                                search_query=clean_queries[0] if clean_queries else ctext,
                                search_queries=clean_queries,
                                claim_type=ctype if ctype in {"factual", "numerical", "statistical", "causal", "definitional", "comparative", "historical"} else "factual",
                                is_factual=is_fact,
                            )
                        )
            
            if claims:
                logger.info(f"Extracted {len(claims)} atomic claims via LLM.")
                return claims

        except Exception as e:
            logger.warning(f"LLM claim extraction failed: {e}. Falling back to semantic rule-based segmenter.")

        return self._rule_based_extract(text)

    def _rule_based_extract(self, text: str) -> List[Claim]:
        """
        Semantic sentence and clause decomposition fallback when LLM is in mock mode or unavailable.
        Decomposes compound sentences into atomic propositions while preserving connected concepts.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        claims: List[Claim] = []
        claim_idx = 1

        for sentence in sentences:
            clean_s = sentence.strip()
            # Ignore short or rhetorical sentences
            if len(clean_s.split()) < 4 or clean_s.endswith("?"):
                continue

            # Check if sentence has compound independent factual propositions
            sub_clauses = self._decompose_sentence_into_propositions(clean_s)

            for clause in sub_clauses:
                if len(clause.split()) < 4:
                    continue

                claim_type = self._classify_claim_type(clause)
                queries = self._generate_query_variants_for_claim(clause)

                claims.append(
                    Claim(
                        claim_id=f"C{claim_idx}",
                        claim_text=clause,
                        original_sentence=clean_s,
                        search_query=queries[0] if queries else clause,
                        search_queries=queries,
                        claim_type=claim_type,
                        is_factual=True,
                    )
                )
                claim_idx += 1
                if len(claims) >= config.MAX_CLAIMS_TO_PROCESS:
                    break

            if len(claims) >= config.MAX_CLAIMS_TO_PROCESS:
                break

        logger.info(f"Extracted {len(claims)} atomic claims via semantic rule-based segmenter.")
        return claims

    def _decompose_sentence_into_propositions(self, sentence: str) -> List[str]:
        """
        Decompose compound sentences with separable factual propositions (e.g., 'X does A and it also does B').
        Preserves unified structures like 'X is trained on Y using Z'.
        """
        # Patterns where two distinct actions/propositions are coordinated
        # Example: "AI is used for personalized learning, and automated grading is also performed."
        # Example: "AI is used for personalized learning and automated grading in schools."
        
        # Check coordinated verb phrases e.g. "is used for X and Y in schools" or "can provide X and automate Y"
        match_coordination = re.match(
            r'^(.*?\b(?:is used for|can|enables|helps to|is)\b)\s+([^,]+?)\s+(?:and|, and|as well as)\s+([^,]+?(\s+in\s+.*|\s+for\s+.*|\s+to\s+.*)?)$',
            sentence,
            re.IGNORECASE
        )
        if match_coordination:
            prefix, part1, part2, context = match_coordination.groups()
            context_str = context or ""
            
            # Verify if part1 and part2 are distinct activities (e.g. "personalized learning" vs "automated grading")
            words1 = part1.strip().split()
            words2 = part2.strip().split()
            if len(words1) >= 2 and len(words2) >= 2 and not ("using" in part2 or "trained" in sentence):
                clean_part1 = part1.strip()
                clean_part2 = part2.strip()
                
                # Formulate 2 atomic claims
                c1 = f"{prefix.strip()} {clean_part1}{context_str}".strip()
                # Ensure c1 ends cleanly
                if not c1.endswith("."):
                    c1 += "."
                
                # For c2, reconstruct complete grammatical predicate
                # Extract subject from prefix
                subj_match = re.match(r'^([^,]+?\b(?:AI|Artificial intelligence|Large language models|Intelligent systems|Systems|Models)\b)', prefix, re.IGNORECASE)
                subj = subj_match.group(1).strip() if subj_match else "AI systems"
                
                c2 = f"{subj} {clean_part2}".strip()
                if not c2.endswith("."):
                    c2 += "."
                
                return [c1, c2]

        # Check compound sentence with ", and they/it"
        compound_split = re.split(r',\s+(?:and\s+(?:they|it|furthermore|additionally)\s+)', sentence, flags=re.IGNORECASE)
        if len(compound_split) == 2 and len(compound_split[0].split()) >= 5 and len(compound_split[1].split()) >= 4:
            s1 = compound_split[0].strip()
            s2 = compound_split[1].strip()
            if not s1.endswith("."):
                s1 += "."
            if not s2.endswith("."):
                s2 += "."
            return [s1, s2]

        return [sentence]

    def _classify_claim_type(self, text: str) -> str:
        """Classify factual claim type based on semantic indicators."""
        text_lower = text.lower()
        if re.search(r'\b(\d+(\.\d+)?%|\d{4}|millions?|billions?|fold|percent)\b', text_lower):
            if "%" in text_lower or "percent" in text_lower or "rate" in text_lower:
                return "statistical"
            return "numerical"
        elif any(w in text_lower for w in ["causes", "caused", "leads to", "results in", "driver of", "due to", "because"]):
            return "causal"
        elif any(w in text_lower for w in ["is defined as", "utilizes", "refers to", "consists of", "principles of"]):
            return "definitional"
        elif any(w in text_lower for w in ["compared to", "higher than", "faster than", "improves upon", "versus"]):
            return "comparative"
        elif re.search(r'\b(in \d{4}|century|historically|past decade|originated)\b', text_lower):
            return "historical"
        return "factual"

    def _generate_query_variants_for_claim(self, claim_text: str) -> List[str]:
        """
        Generate 2-3 high-precision, claim-specific search queries.
        Preserves technical entities, relationships, mechanisms, and metrics.
        Captures both subject entities and specific predicate outcomes/mechanisms.
        """
        # Remove terminal punctuation
        clean_text = re.sub(r'[.!?]+$', '', claim_text.strip())

        # Stop words to remove (filler/grammatical glue only; keep domain terms)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "and", "or", "but",
            "if", "because", "as", "until", "while", "of", "at", "by", "for",
            "with", "about", "against", "between", "into", "through", "during",
            "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "can", "will", "just", "should", "now",
            "furthermore", "additionally", "moreover", "specifically", "utilizes",
            "using", "enables", "helps", "based", "tailored"
        }

        # Extract meaningful tokens
        raw_tokens = re.findall(r'[a-zA-Z0-9_\-]+', clean_text)
        content_tokens = [t for t in raw_tokens if t.lower() not in stop_words]

        if not content_tokens:
            return [clean_text]

        # Extract head (subject area) and tail (mechanism / outcome)
        head_tokens = content_tokens[:min(4, len(content_tokens))]
        tail_tokens = content_tokens[max(0, len(content_tokens)-4):]

        # Variant 1: Complete salient tokens (up to 8 content words)
        q1 = " ".join(content_tokens[:8])

        # Variant 2: Head entities + Tail mechanisms/outcomes (bridges subject and specific predicate)
        combined_head_tail = []
        for t in head_tokens + tail_tokens:
            if t not in combined_head_tail:
                combined_head_tail.append(t)
        q2 = " ".join(combined_head_tail[:7])

        # Variant 3: Specific Mechanism & Outcome focus (tail tokens + core subject)
        q3_tokens = []
        if head_tokens:
            q3_tokens.append(head_tokens[0])
            if len(head_tokens) > 1 and head_tokens[1].lower() in {"intelligence", "language", "tutoring", "learning"}:
                q3_tokens.append(head_tokens[1])
        for t in tail_tokens:
            if t not in q3_tokens:
                q3_tokens.append(t)
        q3 = " ".join(q3_tokens[:6])

        # Deduplicate while preserving order
        queries: List[str] = []
        for q in [q1, q2, q3]:
            q_clean = q.strip()
            if q_clean and q_clean not in queries:
                queries.append(q_clean)

        # Ensure at least 2 distinct variants
        if len(queries) == 1:
            queries.append(f"{queries[0]} research empirical")

        return queries[:3]

