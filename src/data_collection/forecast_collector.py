"""
Forecast Data Collector
Collects weather forecast data (up to 16 days)
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict
from src.utils.logger import get_logger
from src.utils.helpers import (
    load_yaml, save_json, get_location_dir, format_duration
)
from src.data_collection.api_client import OpenMeteoClient
from src.data_collection.rate_limiter import RateLimiter


class ForecastCollector:
    """
    Collects weather forecast data
    """
    
    def __init__(self, config_path: str = "config/config.yaml", logger=None):
        """
        Initialize forecast collector
        
        Args:
            config_path: Path to configuration file
            logger: Logger instance
        """
        self.logger = logger or get_logger()
        self.config = load_yaml(config_path)
        
        # Initialize components
        self.rate_limiter = RateLimiter(self.config['rate_limits'], self.logger)
        self.api_client = OpenMeteoClient(
            self.config['api'],
            self.rate_limiter,
            self.logger
        )
        
        # Configuration
        self.data_config = self.config['data_collection']
        
        self.logger.info("Forecast Collector initialized")
    
    def collect(
        self,
        location_id: str,
        location_config: Dict
    ) -> Dict:
        """
        Collect forecast data for a location
        
        Args:
            location_id: Location identifier
            location_config: Location configuration
            
        Returns:
            Collection statistics
        """
        self.logger.section(f"COLLECTING FORECAST DATA: {location_config['name']}")
        
        start_time = time.time()
        
        # Prepare directories
        raw_dir = get_location_dir('data/raw/forecast', location_id)
        
        try:
            # Fetch forecast data
            self.logger.info("Fetching forecast data...")
            
            data = self.api_client.get_forecast(
                latitude=location_config['latitude'],
                longitude=location_config['longitude'],
                hourly_params=self.data_config['parameters']['hourly'],
                daily_params=self.data_config['parameters'].get('daily'),
                forecast_days=self.data_config['forecast']['forecast_days'],
                past_days=self.data_config['forecast'].get('past_days', 0),
                timezone=location_config.get('timezone', 'UTC')
            )
            
            # Save raw data
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"forecast_{timestamp}.json"
            filepath = Path(raw_dir) / filename
            save_json(data, str(filepath))
            
            self.logger.info(f"✓ Saved: {filename}")
            
            # Calculate statistics
            duration = time.time() - start_time
            stats = {
                'location_id': location_id,
                'collected_at': timestamp,
                'file': str(filepath),
                'forecast_days': self.data_config['forecast']['forecast_days'],
                'duration_seconds': duration,
                'duration_formatted': format_duration(duration),
                'rate_limiter_stats': self.rate_limiter.get_stats()
            }
            
            # Print statistics
            self._print_stats(stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to collect forecast data: {e}")
            raise
    
    def _print_stats(self, stats: Dict):
        """Print collection statistics"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FORECAST COLLECTION COMPLETE!")
        self.logger.info("=" * 80)
        self.logger.info(f"Location: {stats['location_id']}")
        self.logger.info(f"Collected at: {stats['collected_at']}")
        self.logger.info(f"Forecast days: {stats['forecast_days']}")
        self.logger.info(f"Duration: {stats['duration_formatted']}")
        self.logger.info(f"File: {stats['file']}")
        self.logger.info("-" * 80)
        self.logger.info("Rate Limiter Stats:")
        rl_stats = stats['rate_limiter_stats']
        self.logger.info(f"  Total calls: {rl_stats['total_calls']}")
        self.logger.info(f"  Calls today: {rl_stats['calls_today']}")
        self.logger.info(f"  Remaining today: {rl_stats['remaining_today']}")
        self.logger.info("=" * 80)
