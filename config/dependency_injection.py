"""Dependency injection configuration for modular architecture."""

import logging
from typing import Dict, Any, Optional, Type
from pathlib import Path

from src.ai_agent.ai_client_factory import AIClientFactory
from src.data_management import ClusterManager, RubricManager, DataLoader
from src.essay_grading import PairwiseGrader
from src.integrations import SheetsIntegration, KaggleIntegration

logger = logging.getLogger(__name__)


class DIContainer:
    """Dependency injection container for managing component dependencies."""
    
    def __init__(self):
        """Initialize the DI container."""
        self._services = {}
        self._singletons = {}
        self._config = {}
        
    def register_config(self, config: Dict[str, Any]):
        """Register configuration settings."""
        self._config.update(config)
        logger.info("Configuration registered")
        
    def register_singleton(self, service_type: Type, instance: Any):
        """Register a singleton instance."""
        self._singletons[service_type] = instance
        logger.debug(f"Registered singleton: {service_type.__name__}")
        
    def register_service(self, service_type: Type, factory: callable):
        """Register a service factory."""
        self._services[service_type] = factory
        logger.debug(f"Registered service: {service_type.__name__}")
        
    def get(self, service_type: Type):
        """Get a service instance."""
        # Check singletons first
        if service_type in self._singletons:
            return self._singletons[service_type]
            
        # Check registered services
        if service_type in self._services:
            instance = self._services[service_type](self)
            return instance
            
        # Try to create default instances
        return self._create_default_instance(service_type)
        
    def _create_default_instance(self, service_type: Type):
        """Create default instance for common types."""
        if service_type == DataLoader:
            return DataLoader()
        elif service_type == RubricManager:
            return RubricManager()
        elif service_type == ClusterManager:
            return ClusterManager(self.get(DataLoader))
        elif service_type == PairwiseGrader:
            model = self._config.get('model', 'openai:gpt-5-mini')
            tracer = self._config.get('tracer', None)
            return PairwiseGrader(
                model=model,
                cluster_manager=self.get(ClusterManager),
                rubric_manager=self.get(RubricManager),
                tracer=tracer
            )
        else:
            raise ValueError(f"No factory registered for {service_type}")


class ConfigManager:
    """Enhanced configuration management with dependency injection support."""
    
    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        """Initialize configuration manager."""
        self.config = self._load_base_config()
        if config_overrides:
            self.config.update(config_overrides)
            
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration from settings."""
        from config.settings import (
            PROJECT_ROOT, DATA_DIR, DEFAULT_MODEL, BATCH_SIZE,
            LANGSMITH_TRACING, KAGGLE_COMPETITION
        )
        
        return {
            'project_root': PROJECT_ROOT,
            'data_dir': DATA_DIR,
            'model': DEFAULT_MODEL,
            'batch_size': BATCH_SIZE,
            'langsmith_tracing': LANGSMITH_TRACING,
            'kaggle_competition': KAGGLE_COMPETITION
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
        
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values."""
        self.config.update(updates)
        
    def create_di_container(self) -> DIContainer:
        """Create a DI container with this configuration."""
        container = DIContainer()
        container.register_config(self.config)
        
        # Register default factories
        self._register_default_factories(container)
        
        return container
        
    def _register_default_factories(self, container: DIContainer):
        """Register default service factories."""
        
        def create_sheets_integration(di_container):
            credentials_dict = di_container._config.get('sheets_credentials_dict')
            credentials_path = di_container._config.get('sheets_credentials_path')
            return SheetsIntegration(credentials_path, credentials_dict)
            
        def create_kaggle_integration(di_container):
            competition = di_container._config.get('kaggle_competition')
            return KaggleIntegration(competition)
            
        container.register_service(SheetsIntegration, create_sheets_integration)
        container.register_service(KaggleIntegration, create_kaggle_integration)


# Global configuration instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_di_container() -> DIContainer:
    """Get a DI container with default configuration."""
    return get_config_manager().create_di_container()