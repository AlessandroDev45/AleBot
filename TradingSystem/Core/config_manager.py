import os
import json
import logging
from typing import Any, Dict
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self):
        """Initialize configuration manager"""
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'Config',
                'config.json'
            )
            
            with open(config_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
            
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access"""
        return self.config[key]
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return self.config.get(key, default)
        
    def update(self, key: str, value: Any) -> None:
        """Update configuration value"""
        try:
            self.config[key] = value
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            raise
            
    def save(self) -> None:
        """Save configuration to file"""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'Config',
                'config.json'
            )
            
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise
            
    def get_risk_params(self) -> Dict[str, Any]:
        """Get risk management parameters"""
        return self.config.get('risk', {})