"""
LLM abstraction layer supporting Google Gemini, OpenAI, and a deterministic Mock provider.
Ensures zero crashes in Google Colab and easy switching between models.
"""

import json
import logging
from typing import Dict, Any, Optional
from config import config, get_secret

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CitationAssistant.LLM")


class LLMClient:
    """Unified client for interacting with Generative AI models."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.provider = (provider or config.DEFAULT_PROVIDER).lower()
        
        # Dynamically fetch key from get_secret (Colab userdata / env var) if not explicitly provided
        if self.provider == "gemini":
            self.api_key = api_key or get_secret("GEMINI_API_KEY") or config.GEMINI_API_KEY
            self.model_name = model_name or get_secret("GEMINI_MODEL") or config.GEMINI_MODEL
        elif self.provider == "openai":
            self.api_key = api_key or get_secret("OPENAI_API_KEY") or config.OPENAI_API_KEY
            self.model_name = model_name or get_secret("OPENAI_MODEL") or config.OPENAI_MODEL
        else:
            self.api_key = None
            self.model_name = "mock-deterministic"
        
        self._gemini_client = None
        self._openai_client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize provider SDKs safely with fallback protection."""
        if self.provider == "gemini":
            if not self.api_key:
                logger.warning(
                    "No GEMINI_API_KEY detected. Operating in mock/fallback mode until API key is provided."
                )
                self.provider = "mock"
                return
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._gemini_client = genai.GenerativeModel(self.model_name)
                logger.info(f"Initialized Google Gemini model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}. Falling back to mock mode.")
                self.provider = "mock"

        elif self.provider == "openai":
            if not self.api_key:
                logger.warning("No OPENAI_API_KEY detected. Falling back to mock mode.")
                self.provider = "mock"
                return
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.api_key)
                logger.info(f"Initialized OpenAI model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}. Falling back to mock mode.")
                self.provider = "mock"
        else:
            self.provider = "mock"
            logger.info("Using deterministic Mock LLM provider.")

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.3) -> str:
        """Generate plain text output from the model."""
        if self.provider == "gemini" and self._gemini_client:
            try:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = self._gemini_client.generate_content(
                    full_prompt,
                    generation_config={"temperature": temperature}
                )
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini API error: {e}. Using fallback generator.")
                return self._mock_generate(prompt)

        elif self.provider == "openai" and self._openai_client:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                resp = self._openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI API error: {e}. Using fallback generator.")
                return self._mock_generate(prompt)

        else:
            return self._mock_generate(prompt)

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Any:
        """Generate structured JSON output from the model."""
        json_instruction = (
            "\nIMPORTANT: Respond with valid, parseable JSON ONLY. Do not include markdown code block formatting (e.g. ```json ... ```) or conversational commentary."
        )
        combined_prompt = f"{prompt}\n{json_instruction}"
        raw_output = self.generate_text(combined_prompt, system_prompt=system_prompt, temperature=0.1)

        # Clean markdown codeblocks if model returns them
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Direct JSON parse failed: {e}. Attempting substring extraction.")
            start = cleaned.find("{")
            start_arr = cleaned.find("[")
            
            if start != -1 and (start_arr == -1 or start < start_arr):
                end = cleaned.rfind("}")
                if end != -1:
                    return json.loads(cleaned[start : end + 1])
            elif start_arr != -1:
                end = cleaned.rfind("]")
                if end != -1:
                    return json.loads(cleaned[start_arr : end + 1])
            
            raise ValueError(f"Could not parse valid JSON from LLM response: {raw_output}")

    def _mock_generate(self, prompt: str) -> str:
        """Deterministic mock generator for offline testing and demos without API keys."""
        prompt_lower = prompt.lower()
        if "education" in prompt_lower or "learning" in prompt_lower:
            return (
                "Artificial intelligence is transforming education by enabling personalized learning paths tailored to individual student speeds. "
                "Intelligent tutoring systems have been shown to improve student learning outcomes and retention rates. "
                "Furthermore, educational institutions are deploying AI to automate administrative grading and provide real-time feedback."
            )
        elif "climate" in prompt_lower or "warming" in prompt_lower:
            return (
                "Global surface temperatures have increased significantly over the past century due to greenhouse gas emissions. "
                "The Intergovernmental Panel on Climate Change reports that human activities are the primary driver of observed global warming. "
                "Renewable energy technologies like solar and wind power are expanding globally to reduce carbon dioxide emissions."
            )
        elif "quantum" in prompt_lower:
            return (
                "Quantum computing utilizes principles of quantum mechanics such as superposition and entanglement. "
                "Quantum processors have demonstrated quantum computational supremacy on specific specialized mathematical tasks. "
                "Researchers are developing fault-tolerant quantum algorithms for cryptography and molecular simulation."
            )
        else:
            return (
                "Generative artificial intelligence utilizes deep neural networks to synthesize human-like text and multimedia. "
                "Large language models are trained on vast corpora of text data using self-supervised learning techniques. "
                "Ensuring factual grounding and verifiable source citations remains a central challenge in deploying AI responsibly."
            )
