"""
Configuration module for Generative AI Source Citation Assistant.
Supports Google Colab Secrets (userdata), OS environment variables, and local .env files.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load local .env if available
load_dotenv()

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve credentials safely:
    1. Try Google Colab userdata (if running in Colab)
    2. Try OS environment variables
    3. Return default fallback
    """
    try:
        from google.colab import userdata  # type: ignore
        val = userdata.get(key)
        if val:
            return val
    except Exception:
        pass
    
    return os.environ.get(key, default)


class AppConfig:
    """Application configuration and hyperparameter settings."""

    # Provider & Model Settings
    DEFAULT_PROVIDER: str = get_secret("LLM_PROVIDER", "gemini")  # "gemini", "openai", or "mock"
    GEMINI_API_KEY: Optional[str] = get_secret("GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = get_secret("OPENAI_API_KEY")
    TAVILY_API_KEY: Optional[str] = get_secret("TAVILY_API_KEY")
    
    # Model names
    GEMINI_MODEL: str = get_secret("GEMINI_MODEL", "gemini-1.5-flash")
    OPENAI_MODEL: str = get_secret("OPENAI_MODEL", "gpt-4o-mini")

    # Pipeline Thresholds & Limits
    MAX_CLAIMS_TO_PROCESS: int = int(get_secret("MAX_CLAIMS_TO_PROCESS", "6"))
    MAX_SEARCH_RESULTS_PER_CLAIM: int = int(get_secret("MAX_SEARCH_RESULTS", "3"))
    MIN_ENTICEMENT_CONFIDENCE: float = 0.65
    REQUEST_TIMEOUT: int = 15  # seconds for external search & web requests

    # Supported Citation Styles
    SUPPORTED_CITATION_STYLES: List[str] = ["APA", "MLA", "IEEE"]
    DEFAULT_CITATION_STYLE: str = "APA"

    # Domain Authority Scoring Rules
    # Higher scores denote higher scholarly or institutional authority
    AUTHORITY_DOMAIN_SCORES: Dict[str, int] = {
        # Top-level authoritative domains
        ".gov": 95,
        ".edu": 90,
        ".mil": 90,
        ".org": 70,
        
        # Major International & Scientific Organizations
        "who.int": 98,
        "unesco.org": 98,
        "worldbank.org": 95,
        "un.org": 95,
        "oecd.org": 95,
        "cdc.gov": 98,
        "nih.gov": 98,
        "nasa.gov": 98,
        "nist.gov": 95,

        # Academic Publishers & Repositories
        "nature.com": 99,
        "science.org": 99,
        "sciencedirect.com": 95,
        "ieee.org": 95,
        "acm.org": 95,
        "springer.com": 92,
        "wiley.com": 92,
        "tandfonline.com": 90,
        "oup.com": 92,
        "cambridge.org": 92,
        "cell.com": 95,
        "thelancet.com": 98,
        "arxiv.org": 90,
        "biorxiv.org": 88,
        "ncbi.nlm.nih.gov": 98,
        "pubmed.ncbi.nlm.nih.gov": 98,
        "jstor.org": 95,
        "frontiersin.org": 88,
        "mdpi.com": 80,
        "plos.org": 90,
        
        # Encyclopedia & Reference Knowledge Bases
        "wikipedia.org": 75,
        "britannica.com": 85,
        "plato.stanford.edu": 95,
        "investopedia.com": 75,
    }

    @classmethod
    def get_authority_score(cls, url: str) -> int:
        """
        Calculate an authority score (0 - 100) based on URL domain.
        """
        if not url:
            return 30
        
        url_lower = url.lower()
        score = 50  # base score for arbitrary web sources

        # Check explicit domain match
        for domain, domain_score in cls.AUTHORITY_DOMAIN_SCORES.items():
            if domain in url_lower:
                score = max(score, domain_score)

        return score


config = AppConfig()
