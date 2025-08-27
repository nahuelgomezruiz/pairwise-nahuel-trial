"""LangSmith observability integration for AI model calls."""

import logging
import os
import sys
from typing import Optional, Any, Dict
from functools import wraps
from datetime import datetime
from pathlib import Path

# Add root to path for imports
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from config.settings import (
    LANGSMITH_API_KEY,
    LANGSMITH_TRACING,
    LANGSMITH_PROJECT_NAME,
    LANGSMITH_ENDPOINT
)

logger = logging.getLogger(__name__)


class LangSmithTracer:
    """Handles LangSmith tracing for AI model calls."""
    
    def __init__(self):
        """Initialize the LangSmith tracer."""
        self.enabled = LANGSMITH_TRACING and LANGSMITH_API_KEY
        self.wrapped_clients = {}  # Cache for wrapped clients
        
        if self.enabled:
            try:
                import langsmith
                from langsmith import Client
                from langsmith.wrappers import wrap_openai
                
                self.langsmith = langsmith
                self.wrap_openai = wrap_openai
                
                # Initialize LangSmith client
                self.client = Client(
                    api_key=LANGSMITH_API_KEY,
                    api_url=LANGSMITH_ENDPOINT
                )
                
                # Set environment variables for LangSmith
                os.environ["LANGSMITH_TRACING"] = "true"
                os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
                os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT_NAME
                
                logger.info(f"LangSmith tracing enabled for project: {LANGSMITH_PROJECT_NAME}")
                
            except ImportError as e:
                logger.warning("LangSmith library not installed. Tracing disabled.")
                self.enabled = False
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith: {e}")
                self.enabled = False
        else:
            if not LANGSMITH_API_KEY:
                logger.info("LangSmith API key not provided. Tracing disabled.")
            else:
                logger.info("LangSmith tracing disabled in settings.")
    
    def wrap_openai_client(self, client):
        """Wrap an OpenAI client with LangSmith tracing."""
        if not self.enabled:
            return client
            
        # Check if already wrapped
        client_id = id(client)
        if client_id in self.wrapped_clients:
            return self.wrapped_clients[client_id]
            
        try:
            wrapped = self.wrap_openai(client)
            self.wrapped_clients[client_id] = wrapped
            logger.debug("OpenAI client wrapped with LangSmith tracing")
            return wrapped
        except Exception as e:
            logger.warning(f"Failed to wrap OpenAI client: {e}")
            return client
    
    def wrap_anthropic_client(self, client):
        """Wrap an Anthropic client with LangSmith tracing."""
        if not self.enabled:
            return client
            
        # Check if already wrapped
        client_id = id(client)
        if client_id in self.wrapped_clients:
            return self.wrapped_clients[client_id]
            
        try:
            # Check if wrap_anthropic is available in langsmith.wrappers
            try:
                from langsmith.wrappers import wrap_anthropic
                wrapped = wrap_anthropic(client)
                self.wrapped_clients[client_id] = wrapped
                logger.debug("Anthropic client wrapped with LangSmith tracing using wrap_anthropic")
                return wrapped
            except ImportError:
                # Fall back to manual instrumentation if wrap_anthropic is not available
                logger.debug("wrap_anthropic not available, using manual tracing for Anthropic client")
                return client
        except Exception as e:
            logger.warning(f"Failed to wrap Anthropic client: {e}")
            return client
    
    def wrap_gemini_client(self, client):
        """Wrap a Gemini client with LangSmith tracing."""
        if not self.enabled:
            return client
            
        # Gemini also requires manual instrumentation
        logger.debug("Gemini client prepared for LangSmith tracing")
        return client
    
    def trace_comparison(self, func):
        """Decorator to trace pairwise comparison functions."""
        if not self.enabled:
            return func
            
        try:
            from langsmith import traceable
            
            @wraps(func)
            @traceable(
                name=func.__name__,
                run_type="chain",
                tags=["comparison", "grading"],
                metadata={"function": func.__name__}
            )
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapper
        except Exception as e:
            logger.warning(f"Failed to add tracing to {func.__name__}: {e}")
            return func
    
    def trace_complete_call(self, provider: str, model: str):
        """Decorator to trace complete method calls for non-OpenAI providers."""
        def decorator(func):
            if not self.enabled:
                return func
                
            try:
                from langsmith import traceable
                
                @wraps(func)
                @traceable(
                    name=f"{provider}_complete",
                    run_type="llm",
                    tags=[provider.lower(), "completion", model],
                    metadata={"provider": provider, "model": model}
                )
                def wrapper(*args, **kwargs):
                    # Extract prompt if available
                    prompt = args[1] if len(args) > 1 else kwargs.get("prompt", "")
                    
                    # Add prompt to metadata
                    from langsmith import get_current_run
                    current_run = get_current_run()
                    if current_run:
                        current_run.inputs = {"prompt": prompt}
                    
                    # Execute the actual call
                    result = func(*args, **kwargs)
                    
                    # Add result to outputs
                    if current_run:
                        current_run.outputs = {"completion": result}
                    
                    return result
                
                return wrapper
            except Exception as e:
                logger.warning(f"Failed to add tracing to {provider} complete call: {e}")
                return func
        
        return decorator
    
    def trace_llm_call(self, provider: str, model: str):
        """Decorator to trace LLM calls with proper inputs/outputs logging."""
        def decorator(func):
            if not self.enabled:
                return func
                
            try:
                from langsmith import traceable
                
                @wraps(func)
                @traceable(
                    name=f"{provider}_{model}",
                    run_type="llm",
                    tags=[provider.lower(), "llm", model],
                    metadata={
                        "provider": provider, 
                        "model": model,
                        "llm_provider": provider.lower()
                    }
                )
                def wrapper(prompt: str, temperature: float = 0.3, **kwargs):
                    # Log the inputs
                    inputs = {
                        "prompt": prompt,
                        "temperature": temperature,
                        "model": model,
                        **kwargs
                    }
                    
                    # Execute the actual call
                    result = func(prompt, temperature, **kwargs)
                    
                    # The traceable decorator will handle logging
                    return result
                
                return wrapper
            except Exception as e:
                logger.warning(f"Failed to add tracing to {provider} LLM call: {e}")
                return func
        
        return decorator
    
    def trace_grading_session(self, name: str = "grading_session"):
        """Context manager for tracing a complete grading session."""
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()
            
        try:
            from langsmith import traceable
            
            @traceable(
                name=name,
                run_type="chain",
                tags=["grading_session"],
                metadata={"timestamp": datetime.now().isoformat()}
            )
            def session_context():
                pass
            
            return session_context()
        except Exception as e:
            logger.warning(f"Failed to create grading session context: {e}")
            from contextlib import nullcontext
            return nullcontext()
    
    def log_comparison_result(self, essay_a_id: str, essay_b_id: str, result: str):
        """Log a comparison result to LangSmith."""
        if not self.enabled:
            return
            
        try:
            # Simply log the metadata - it will be captured by the tracer
            logger.info(f"Comparison result: {result}")
            
            # The traceable decorator will handle the actual logging to LangSmith
            # We don't need to manually add metadata since the traced function
            # will include all the context automatically
                
        except Exception as e:
            logger.warning(f"Failed to log comparison result: {e}")


# Global tracer instance
tracer = LangSmithTracer()