"""
Logger utility for irrigation predictor
Provides structured logging with file and console output
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


class Logger:
    """Custom logger with file and console output"""
    
    def __init__(self, name="irrigation_predictor", config=None):
        """
        Initialize logger
        
        Args:
            name: Logger name
            config: Configuration dictionary with logging settings
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Default configuration
        if config is None:
            config = {
                'level': 'INFO',
                'file': 'logs/collection.log',
                'console': True
            }
        
        # Set logging level
        level = getattr(logging, config.get('level', 'INFO').upper())
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler
        log_file = config.get('file', 'logs/collection.log')
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        if config.get('console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)
    
    def debug(self, message, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)
    
    def section(self, title):
        """Log a section separator"""
        separator = "=" * 80
        self.logger.info(separator)
        self.logger.info(f"  {title}")
        self.logger.info(separator)


def get_logger(name="irrigation_predictor", config=None):
    """
    Get or create a logger instance
    
    Args:
        name: Logger name
        config: Configuration dictionary
        
    Returns:
        Logger instance
    """
    return Logger(name, config)
