"""
Pipeline Orchestrator Module:
Coordinates Answer Generation, Claim Extraction, Evidence Retrieval,
Source Validation, and Citation Formatting into a single end-to-end pipeline.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field

from llm import LLMClient
from claim_extractor import ClaimExtractor, Claim
from source_search import SourceSearchEngine, SourceDocument
from source_validator import SourceValidator, ValidationResult, VerificationStatus
from citation_engine import CitationEngine, CitationResult
from config import config

logger = logging.getLogger("CitationAssistant.Pipeline")


class PipelineMetrics(BaseModel):
    """Execution metrics and verification statistics."""
    total_claims: int
    supported_claims: int
    partially_supported_claims: int
    insufficient_or_unsupported: int
    grounding_rate_pct: float
    total_sources_retrieved: int
    unique_sources_cited: int
    processing_time_seconds: float


class PipelineResult(BaseModel):
    """Complete output package from the citation assistant pipeline."""
    user_query: str
    citation_style: str
    raw_answer: str
    claims: List[Claim]
    validations: List[ValidationResult]
    citation_result: CitationResult
    metrics: PipelineMetrics


class CitationAssistantPipeline:
    """Master pipeline orchestrator for generative AI source citations."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        search_engine: Optional[SourceSearchEngine] = None,
        validator: Optional[SourceValidator] = None,
        citation_engine: Optional[CitationEngine] = None,
    ):
        self.llm = llm_client or LLMClient()
        self.claim_extractor = ClaimExtractor(llm_client=self.llm)
        self.search_engine = search_engine or SourceSearchEngine()
        self.validator = validator or SourceValidator(llm_client=self.llm)
        self.citation_engine = citation_engine or CitationEngine()

    def run(
        self,
        query: str,
        citation_style: str = "APA",
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> PipelineResult:
        """
        Execute the complete 5-stage citation pipeline.
        """
        start_time = time.time()
        logger.info(f"Starting pipeline execution for query: '{query}' [Style: {citation_style}]")

        def update_progress(msg: str, frac: float):
            if progress_callback:
                progress_callback(msg, frac)

        # -------------------------------------------------------------
        # Stage 1: Generative AI Answer Synthesis
        # -------------------------------------------------------------
        update_progress("Synthesizing Generative AI answer...", 0.15)
        system_prompt = (
            "You are an authoritative academic researcher. Provide a comprehensive, well-structured, "
            "and factual answer to the user's question. Focus on objective facts, empirical findings, "
            "and established institutional insights."
        )
        raw_answer = self.llm.generate_text(query, system_prompt=system_prompt)
        logger.info("Stage 1 completed: Raw answer generated.")

        # -------------------------------------------------------------
        # Stage 2: Claim Extraction & Decomposition
        # -------------------------------------------------------------
        update_progress("Extracting atomic factual claims...", 0.35)
        claims = self.claim_extractor.extract_claims(raw_answer, user_question=query)
        logger.info(f"Stage 2 completed: {len(claims)} claims extracted.")

        # -------------------------------------------------------------
        # Stage 3 & 4: Evidence Retrieval & Entailment Validation
        # -------------------------------------------------------------
        update_progress("Retrieving authoritative sources and verifying claims...", 0.60)
        validations: List[ValidationResult] = []
        total_sources_count = 0

        for i, claim in enumerate(claims):
            step_frac = 0.60 + (0.25 * (i / max(len(claims), 1)))
            update_progress(f"Verifying claim {i+1}/{len(claims)}: '{claim.claim_text[:40]}...' ", step_frac)
            
            # Retrieve candidate sources
            candidate_sources = self.search_engine.search_for_claim(claim.claim_id, claim.search_query)
            total_sources_count += len(candidate_sources)

            # Validate entailment
            val_result = self.validator.validate_claim(claim, candidate_sources)
            validations.append(val_result)

        logger.info("Stage 3 & 4 completed: Sources retrieved and validated.")

        # -------------------------------------------------------------
        # Stage 5: In-Text Citation Injection & Bibliography Formatting
        # -------------------------------------------------------------
        update_progress("Attaching in-text citations and formatting bibliography...", 0.90)
        citation_res = self.citation_engine.process_citations(
            original_text=raw_answer,
            validation_results=validations,
            style=citation_style
        )
        logger.info("Stage 5 completed: Citations attached.")

        # -------------------------------------------------------------
        # Compute Verification Metrics
        # -------------------------------------------------------------
        supported_count = sum(1 for v in validations if v.status == VerificationStatus.SUPPORTED)
        partially_supported_count = sum(1 for v in validations if v.status == VerificationStatus.PARTIALLY_SUPPORTED)
        insufficient_count = len(validations) - (supported_count + partially_supported_count)
        
        grounding_rate = (
            ((supported_count + 0.5 * partially_supported_count) / max(len(validations), 1)) * 100.0
        )
        elapsed_sec = round(time.time() - start_time, 2)

        metrics = PipelineMetrics(
            total_claims=len(claims),
            supported_claims=supported_count,
            partially_supported_claims=partially_supported_count,
            insufficient_or_unsupported=insufficient_count,
            grounding_rate_pct=round(grounding_rate, 1),
            total_sources_retrieved=total_sources_count,
            unique_sources_cited=citation_res.total_sources_cited,
            processing_time_seconds=elapsed_sec,
        )

        update_progress("Pipeline complete!", 1.0)

        return PipelineResult(
            user_query=query,
            citation_style=citation_style,
            raw_answer=raw_answer,
            claims=claims,
            validations=validations,
            citation_result=citation_res,
            metrics=metrics,
        )
