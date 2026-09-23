"""
Unit and Integration Test Suite for Generative AI Source Citation Assistant.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config, AppConfig
from llm import LLMClient
from claim_extractor import ClaimExtractor, Claim
from source_search import SourceSearchEngine, SourceDocument
from source_validator import SourceValidator, VerificationStatus, ValidationResult
from citation_engine import CitationEngine
from pipeline import CitationAssistantPipeline


class TestConfiguration:
    """Test configuration and domain scoring rules."""

    def test_domain_authority_scoring(self):
        assert AppConfig.get_authority_score("https://www.cdc.gov/flu") == 98
        assert AppConfig.get_authority_score("https://stanford.edu/research") == 90
        assert AppConfig.get_authority_score("https://nature.com/articles/s41586") == 99
        assert AppConfig.get_authority_score("https://en.wikipedia.org/wiki/AI") == 75
        assert AppConfig.get_authority_score("https://random-blog.com/post") == 50


class TestLLMClient:
    """Test LLM client abstraction and mock fallback."""

    def test_mock_generation(self):
        client = LLMClient(provider="mock")
        resp = client.generate_text("How does AI impact education?")
        assert len(resp) > 20
        assert "education" in resp.lower() or "learning" in resp.lower()


class TestClaimExtraction:
    """Test atomic claim extraction and query generation."""

    def test_rule_based_claim_extraction(self):
        extractor = ClaimExtractor(llm_client=LLMClient(provider="mock"))
        sample_text = (
            "Artificial intelligence is rapidly expanding in modern schools. "
            "Adaptive algorithms allow teachers to personalize student curriculum. "
            "Automated grading tools save instructional hours each week."
        )
        claims = extractor.extract_claims(sample_text)
        assert len(claims) >= 2
        assert all(isinstance(c, Claim) for c in claims)
        assert claims[0].claim_id == "C1"
        assert len(claims[0].search_query) > 0


class TestSourceSearch:
    """Test search engine retrieval and ranking."""

    def test_fallback_and_scoring(self):
        search_engine = SourceSearchEngine(max_results_per_claim=2)
        docs = search_engine.search_for_claim("C1", "artificial intelligence education personalized learning")
        assert len(docs) >= 1
        assert isinstance(docs[0], SourceDocument)
        assert docs[0].authority_score >= 50
        assert docs[0].url.startswith("http")


class TestSourceValidator:
    """Test entailment verification and status mapping."""

    def test_validation_logic(self):
        validator = SourceValidator(llm_client=LLMClient(provider="mock"))
        claim = Claim(
            claim_id="C1",
            claim_text="AI systems personalize learning pathways for students.",
            original_sentence="AI systems personalize learning pathways for students.",
            search_query="AI personalized learning students",
        )
        supporting_doc = SourceDocument(
            source_id="S1",
            claim_id="C1",
            title="Personalized Learning with AI",
            url="https://www.unesco.org/ai-education",
            publisher="UNESCO",
            publication_year="2023",
            snippet="AI systems personalize learning pathways and curricula for students based on real-time mastery.",
            authority_score=98,
        )

        val_result = validator.validate_claim(claim, [supporting_doc])
        assert isinstance(val_result, ValidationResult)
        assert val_result.status in (VerificationStatus.SUPPORTED, VerificationStatus.PARTIALLY_SUPPORTED)
        assert len(val_result.supporting_sources) > 0


class TestCitationEngine:
    """Test in-text citation injection and bibliography formatting."""

    def test_apa_mla_ieee_formatting(self):
        engine = CitationEngine()
        doc = SourceDocument(
            source_id="S1",
            claim_id="C1",
            title="UNESCO Artificial Intelligence Guidance",
            url="https://www.unesco.org/ai",
            publisher="UNESCO",
            publication_year="2023",
            snippet="AI supports personalized student learning.",
            authority_score=98,
        )
        val = ValidationResult(
            claim_id="C1",
            claim_text="AI supports personalized student learning.",
            status=VerificationStatus.SUPPORTED,
            confidence_score=0.95,
            reasoning="Direct match.",
            supporting_sources=[doc],
        )

        # Test APA
        res_apa = engine.process_citations("AI supports personalized student learning.", [val], style="APA")
        assert "[1]" in res_apa.cited_answer
        assert "(2023)" in res_apa.raw_references_text
        assert "UNESCO" in res_apa.raw_references_text

        # Test MLA
        res_mla = engine.process_citations("AI supports personalized student learning.", [val], style="MLA")
        assert "UNESCO." in res_mla.raw_references_text

        # Test IEEE
        res_ieee = engine.process_citations("AI supports personalized student learning.", [val], style="IEEE")
        assert "[Online]. Available:" in res_ieee.raw_references_text

        # Test BibTeX
        assert "@misc{" in res_apa.bibtex_export


class TestEndToEndPipeline:
    """Test full pipeline integration."""

    def test_pipeline_execution(self):
        pipeline = CitationAssistantPipeline(llm_client=LLMClient(provider="mock"))
        result = pipeline.run("How does AI affect education?", citation_style="APA")

        assert result.user_query == "How does AI affect education?"
        assert len(result.claims) > 0
        assert len(result.validations) == len(result.claims)
        assert result.metrics.total_claims == len(result.claims)
        assert result.metrics.processing_time_seconds >= 0.0
        assert len(result.citation_result.cited_answer) > 0
