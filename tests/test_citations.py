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


class TestPhase2AAtomicExtraction:
    """Dedicated tests for Phase 2A atomic claim extraction and query specificity."""

    def test_atomic_claims_split(self):
        """Test A: Compound sentences with separable factual assertions are decomposed into atomic claims."""
        extractor = ClaimExtractor(llm_client=LLMClient(provider="mock"))
        compound_text = "AI is used for personalized learning and automated grading in schools."
        claims = extractor.extract_claims(compound_text)
        
        # Should produce 2 atomic claims
        assert len(claims) == 2
        claim_texts = [c.claim_text.lower() for c in claims]
        assert any("personalized learning" in ct for ct in claim_texts)
        assert any("automated grading" in ct or "grading" in ct for ct in claim_texts)

    def test_connected_facts_remain_single_claim(self):
        """Test B: Tightly connected facts describing a single mechanism remain one unified claim."""
        extractor = ClaimExtractor(llm_client=LLMClient(provider="mock"))
        connected_text = "Large language models are trained using self-supervised learning on large text datasets."
        claims = extractor.extract_claims(connected_text)
        
        # Should remain 1 coherent claim
        assert len(claims) == 1
        assert "self-supervised learning" in claims[0].claim_text.lower()
        assert "large text datasets" in claims[0].claim_text.lower()

    def test_query_specificity(self):
        """Test C: Generated search queries must preserve technical entities and not collapse to generic topics."""
        extractor = ClaimExtractor(llm_client=LLMClient(provider="mock"))
        text = "Large language models are trained using self-supervised learning on large text datasets."
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1
        claim = claims[0]
        
        # Verify queries contain technical terms
        primary_q = claim.search_query.lower()
        all_q_text = " ".join([q.lower() for q in claim.search_queries])
        
        # Must NOT be just a generic topic like "artificial intelligence"
        assert primary_q != "artificial intelligence"
        assert primary_q != "ai"
        
        # Must retain specific entities/mechanisms
        assert "language" in all_q_text or "models" in all_q_text or "llm" in all_q_text
        assert "self" in all_q_text or "supervised" in all_q_text or "trained" in all_q_text

    def test_multiple_query_variants(self):
        """Test D: Extractor must generate 2-3 distinct query variants per claim."""
        extractor = ClaimExtractor(llm_client=LLMClient(provider="mock"))
        text = "AI systems provide personalized learning experiences based on individual student needs."
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1
        claim = claims[0]
        
        # Check that 2-3 query variants are generated
        assert len(claim.search_queries) >= 2
        assert len(claim.search_queries) <= 3
        # Check that variants are distinct strings
        assert len(set(claim.search_queries)) == len(claim.search_queries)

    def test_no_hallucinated_facts(self):
        """Test E: Extracted claims must not invent facts absent from the input."""
        extractor = ClaimExtractor(llm_client=LLMClient(provider="mock"))
        original_text = "Quantum computing utilizes principles of quantum mechanics such as superposition."
        claims = extractor.extract_claims(original_text)
        assert len(claims) >= 1
        for claim in claims:
            # Check key keywords come directly from the source sentence
            assert "quantum" in claim.claim_text.lower()
            assert "superposition" in claim.claim_text.lower()


class TestPhase2AProblematicExamples:
    """Test the specific observed problematic examples from real test cases."""

    def test_generative_ai_deep_neural_networks(self):
        """Example 1: Generative AI deep neural networks multimedia claim."""
        extractor = ClaimExtractor(llm_client=LLMClient(provider="mock"))
        text = "Generative artificial intelligence utilizes deep neural networks to synthesize human-like text and multimedia."
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1
        claim = claims[0]
        
        # Queries must target deep neural networks and text/multimedia synthesis
        combined_queries = " ".join([q.lower() for q in claim.search_queries])
        assert "neural" in combined_queries or "generative" in combined_queries
        assert "text" in combined_queries or "multimedia" in combined_queries or "synthesize" in combined_queries

        # Ensure search engine returns precision sources
        search_engine = SourceSearchEngine()
        sources = search_engine.search_for_claim(claim.claim_id, claim.search_queries)
        assert len(sources) >= 1
        # The retrieved source must directly discuss deep generative models or synthesis
        snippet_all = " ".join([s.snippet.lower() + " " + s.title.lower() for s in sources])
        assert "generative" in snippet_all or "neural" in snippet_all

    def test_personalized_learning_paths(self):
        """Example 2: AI personalized learning paths education claim."""
        extractor = ClaimExtractor(llm_client=LLMClient(provider="mock"))
        text = "Artificial intelligence is transforming education by enabling personalized learning paths tailored to individual student speeds."
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1
        claim = claims[0]
        
        # Queries must target education, personalized learning, student speeds
        combined_queries = " ".join([q.lower() for q in claim.search_queries])
        assert "personalized" in combined_queries or "learning" in combined_queries
        assert "education" in combined_queries or "student" in combined_queries

        # Ensure search engine returns education sources (NOT healthcare)
        search_engine = SourceSearchEngine()
        sources = search_engine.search_for_claim(claim.claim_id, claim.search_queries)
        assert len(sources) >= 1
        snippet_all = " ".join([s.snippet.lower() + " " + s.title.lower() for s in sources])
        assert "healthcare" not in sources[0].title.lower()
        assert "education" in snippet_all or "learning" in snippet_all or "tutoring" in snippet_all

    def test_llm_self_supervised_corpora(self):
        """Example 3: LLM vast text corpora self-supervised learning claim."""
        extractor = ClaimExtractor(llm_client=LLMClient(provider="mock"))
        text = "Large language models are trained on vast corpora of text data using self-supervised learning techniques."
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1
        claim = claims[0]
        
        combined_queries = " ".join([q.lower() for q in claim.search_queries])
        assert "self" in combined_queries or "supervised" in combined_queries
        assert "corpora" in combined_queries or "text" in combined_queries or "models" in combined_queries


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

