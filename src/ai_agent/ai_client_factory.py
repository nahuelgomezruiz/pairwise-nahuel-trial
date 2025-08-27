"""Factory for creating AI clients from different providers."""

import logging
import os
from typing import Optional, Dict, Any
from threading import Lock
from enum import Enum

# Import LangSmith tracer
try:
    from .langsmith_tracer import tracer
except ImportError:
    tracer = None
    
logger = logging.getLogger(__name__)

# System prompt used across all AI providers for consistency
# To change the role/context for all providers, modify this single constant
SYSTEM_PROMPT = "You are an expert essay grader."


class AIProvider(Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    GEMINI = "gemini"


class BaseAIClient:
    """Base class for AI clients."""
    
    def __init__(self, model: str = None):
        self.model = model
        
    def complete(self, prompt: str, **kwargs) -> str:
        """Make a completion request to the AI model."""
        raise NotImplementedError
        
    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        return self.model
    
    def grade_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Grade with retry logic for robustness."""
        import time
        
        for attempt in range(max_retries):
            try:
                return self.complete(prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
                
    def parse_grading_response(self, response: str) -> Dict[str, Any]:
        """Parse a grading response to extract score and reasoning."""
        import re
        
        # Try to extract final score
        score_pattern = r'(?:FINAL\s+)?SCORE:\s*(\d+(?:\.\d+)?)'
        score_match = re.search(score_pattern, response, re.IGNORECASE)
        
        if score_match:
            total_score = float(score_match.group(1))
        else:
            # Fallback
            total_score = 3.5
            
        # Try to extract reasoning
        reasoning_pattern = r'REASONING:(.*?)(?:FINAL\s+SCORE:|$)'
        reasoning_match = re.search(reasoning_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            reasoning = response.strip()
            
        return {
            'total_score': total_score,
            'reasoning': reasoning,
            'category_scores': {}  # Will be populated for component-based grading
        }


class OpenAIClient(BaseAIClient):
    """OpenAI API client."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        super().__init__(model)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
            
        # Import OpenAI library
        try:
            import openai
            self.openai = openai
            client = openai.OpenAI(api_key=self.api_key)
            
            # Wrap with LangSmith if available
            if tracer:
                self.client = tracer.wrap_openai_client(client)
            else:
                self.client = client
                
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
            
    def complete(self, prompt: str, temperature: float = 0.3, **kwargs) -> str:
        """Make a completion request to OpenAI."""
        try:
            # Check if this is a model that only supports temperature=1.0 (o3 or gpt-5-mini)
            # These models use internal reasoning mechanisms instead of temperature variation
            is_temp_restricted_model = 'o3' in self.model.lower() or 'gpt-5-mini' in self.model.lower()
            
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Some models only support default temperature (1.0)
            if not is_temp_restricted_model:
                request_params["temperature"] = temperature
            # For restricted models, we don't set temperature parameter (uses default 1.0)
            
            # Add reasoning effort for gpt-5-mini (medium effort for comparison tasks)
            # Reference: https://platform.openai.com/docs/guides/reasoning
            # Medium effort provides balanced reasoning for comparison tasks
            if 'gpt-5-mini' in self.model.lower():
                request_params["reasoning_effort"] = "medium"
                logger.info("Using medium reasoning effort for gpt-5-mini")
            
            # Add any other kwargs
            for key, value in kwargs.items():
                if key != "temperature" or not is_temp_restricted_model:  # Skip temperature for restricted models
                    request_params[key] = value
            
            response = self.client.chat.completions.create(**request_params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicClient(BaseAIClient):
    """Anthropic Claude API client."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: str = None):
        super().__init__(model)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
            
        # Import Anthropic library
        try:
            import anthropic
            self.anthropic = anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Wrap with LangSmith if available
            if tracer:
                self.client = tracer.wrap_anthropic_client(client)
            else:
                self.client = client
                
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
            
    def complete(self, prompt: str, temperature: float = 0.3, max_tokens: int = 4000, **kwargs) -> str:
        """Make a completion request to Anthropic."""
        try:
            # For rubric parsing and prompt generation, use Opus
            if "extract" in prompt.lower() or "generate" in prompt.lower():
                model_to_use = "claude-3-opus-20240229"
            else:
                # For grading, use the specified model (default Sonnet)
                model_to_use = self.model
                
            response = self.client.messages.create(
                model=model_to_use,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class GeminiClient(BaseAIClient):
    """Google Gemini API client."""
    
    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None):
        super().__init__(model)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
            
        # Import Google Generative AI library
        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=self.api_key)
            self.model_instance = genai.GenerativeModel(
                model,
                system_instruction=SYSTEM_PROMPT
            )
            
            # Apply tracing decorator if available
            if tracer:
                self._complete_impl = tracer.trace_llm_call("Gemini", model)(self._raw_complete)
            else:
                self._complete_impl = self._raw_complete
                
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
            
    def complete(self, prompt: str, temperature: float = 0.3, **kwargs) -> str:
        """Make a completion request to Gemini."""
        return self._complete_impl(prompt, temperature, **kwargs)
    
    def _raw_complete(self, prompt: str, temperature: float = 0.3, **kwargs) -> str:
        """Raw completion request."""
        try:
            # Convert common parameters to Gemini's expected format
            gemini_kwargs = {}
            for key, value in kwargs.items():
                if key == "max_tokens":
                    gemini_kwargs["max_output_tokens"] = value
                else:
                    gemini_kwargs[key] = value
            
            generation_config = self.genai.GenerationConfig(
                temperature=temperature,
                **gemini_kwargs
            )
            
            response = self.model_instance.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


class AIClientFactory:
    """Factory for creating AI clients."""
    
    # Simple process-wide client cache to enable HTTP connection reuse and reduce creation overhead
    _client_cache: Dict[str, BaseAIClient] = {}
    _cache_lock: Lock = Lock()

    # Default models for each provider
    DEFAULT_MODELS = {
        AIProvider.OPENAI: "gpt-4o-mini",
        AIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
        AIProvider.GEMINI: "gemini-1.5-flash"
    }
    
    # Model aliases for convenience
    MODEL_ALIASES = {
        # OpenAI aliases
        "gpt4": "gpt-4o",
        "gpt4-mini": "gpt-4o-mini",
        "gpt-4": "gpt-4o",
        "gpt-3.5": "gpt-3.5-turbo",
        
        # Anthropic aliases  
        "claude": "claude-3-5-sonnet-20241022",
        "claude-opus": "claude-3-opus-20240229",
        "claude-sonnet": "claude-3-5-sonnet-20241022",
        "claude-haiku": "claude-3-haiku-20240307",
        "opus": "claude-3-opus-20240229",
        "sonnet": "claude-3-5-sonnet-20241022",
        "haiku": "claude-3-haiku-20240307",
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "sonnet-4": "claude-sonnet-4-20250514",
        
        # Gemini aliases
        "gemini": "gemini-1.5-flash",
        "gemini-pro": "gemini-1.5-pro",
        "gemini-flash": "gemini-1.5-flash",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash"
    }
    
    @classmethod
    def get_client(cls, model_spec: Optional[str] = None) -> BaseAIClient:
        """
        Get an AI client based on model specification.
        
        Args:
            model_spec: Model specification string. Can be:
                       - None: Use default (OpenAI GPT-4o-mini)
                       - Provider name: "openai", "anthropic", "gemini"
                       - Model name: "gpt-4", "claude-opus", "gemini-pro", etc.
                       - Full spec: "openai:gpt-4", "anthropic:claude-opus", etc.
                       
        Returns:
            Configured AI client
        """
        if not model_spec:
            # Default to OpenAI
            return OpenAIClient()
            
        # Resolve aliases
        model_spec = cls.MODEL_ALIASES.get(model_spec.lower(), model_spec)
        
        # Check if it's a provider:model format
        if ":" in model_spec:
            provider_str, model = model_spec.split(":", 1)
            provider = cls._get_provider(provider_str)
        else:
            # Try to infer provider from model name
            provider, model = cls._infer_provider_from_model(model_spec)
            
        # Reuse or create the appropriate client
        cache_key = f"{provider.value}:{model}"
        with cls._cache_lock:
            cached = cls._client_cache.get(cache_key)
            if cached is not None:
                return cached
            if provider == AIProvider.OPENAI:
                client = OpenAIClient(model=model)
            elif provider == AIProvider.ANTHROPIC:
                client = AnthropicClient(model=model)
            elif provider == AIProvider.GEMINI:
                client = GeminiClient(model=model)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            cls._client_cache[cache_key] = client
            return client
            
    @classmethod
    def _get_provider(cls, provider_str: str) -> AIProvider:
        """Get provider enum from string."""
        provider_str = provider_str.lower()
        for provider in AIProvider:
            if provider.value == provider_str:
                return provider
        raise ValueError(f"Unknown provider: {provider_str}")
        
    @classmethod
    def _infer_provider_from_model(cls, model: str) -> tuple:
        """Infer provider from model name."""
        model_lower = model.lower()
        
        # Check if it's just a provider name
        for provider in AIProvider:
            if provider.value == model_lower:
                return provider, cls.DEFAULT_MODELS[provider]
                
        # Check model patterns
        if "gpt" in model_lower:
            return AIProvider.OPENAI, model
        elif "claude" in model_lower:
            return AIProvider.ANTHROPIC, model
        elif "gemini" in model_lower:
            return AIProvider.GEMINI, model
        else:
            # Default to OpenAI with the given model name
            logger.warning(f"Could not infer provider for model '{model}', defaulting to OpenAI")
            return AIProvider.OPENAI, model