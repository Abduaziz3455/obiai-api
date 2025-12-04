"""
Historical Data Collector
Collects 1 year of historical weather data with resume capability
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from src.utils.logger import get_logger
from src.utils.helpers import (
    load_yaml, save_json, load_json,
    get_date_range, ensure_dir, get_location_dir,
    get_checkpoint_path, format_duration, calculate_progress
)
from src.data_collection.api_client import OpenMeteoClient
from src.data_collection.rate_limiter import RateLimiter


class HistoricalCollector:
    """
    Collects historical weather data with checkpoint/resume support
    """
    
    def __init__(self, config_path: str = "config/config.yaml", logger=None):
        """
        Initialize historical collector
        
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
        self.resume_config = self.config['resume']
        
        self.logger.info("Historical Collector initialized")
    
    def collect(
        self,
        location_id: str,
        location_config: Dict,
        resume: bool = True
    ) -> Dict:
        """
        Collect historical data for a location
        """
        self.logger.section(f"COLLECTING HISTORICAL DATA: {location_config['name']}")
        
        start_time = time.time()
        raw_dir = get_location_dir('data/raw/historical', location_id)
        checkpoint = self._load_checkpoint(location_id, resume)
        date_ranges = self._get_collection_ranges(checkpoint)
        
        if not date_ranges:
            self.logger.info("✓ All data already collected!")
            return self._get_collection_stats(checkpoint, time.time() - start_time)
        
        self.logger.info(f"Data ranges to collect: {len(date_ranges)}")
        
        for i, (start_date, end_date) in enumerate(date_ranges, 1):
            self.logger.info(f"\n[{i}/{len(date_ranges)}] Collecting: {start_date} to {end_date}")
            
            try:
                # USE ARCHIVE API (not forecast)
                data = self.api_client.get_historical(
                    latitude=location_config['latitude'],
                    longitude=location_config['longitude'],
                    start_date=start_date,
                    end_date=end_date,
                    hourly_params=self.data_config['parameters']['hourly'],
                    daily_params=self.data_config['parameters'].get('daily'),
                    timezone=location_config.get('timezone', 'UTC')
                )
                
                # Save raw data
                filename = f"{start_date}_to_{end_date}.json"
                filepath = Path(raw_dir) / filename
                save_json(data, str(filepath))
                
                self.logger.info(f"✓ Saved: {filename}")
                
                # Update checkpoint
                checkpoint['completed_ranges'].append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'collected_at': datetime.now().isoformat(),
                    'file': str(filepath)
                })
                checkpoint['last_completed_date'] = end_date
                checkpoint['total_api_calls'] += 1
                
                # Save checkpoint
                if i % self.resume_config.get('auto_save_interval', 5) == 0:
                    self._save_checkpoint(location_id, checkpoint)
                    self.logger.info(f"Checkpoint saved (progress: {i}/{len(date_ranges)})")
                
                # Progress update
                progress = calculate_progress(i, len(date_ranges))
                self.logger.info(f"Progress: {progress:.1f}%")
                
                # Rate limit info
                remaining = self.rate_limiter.get_remaining_calls()
                self.logger.debug(f"Remaining calls - Day: {remaining['day']}, Month: {remaining['month']}")
                
            except Exception as e:
                self.logger.error(f"Failed to collect {start_date} to {end_date}: {e}")
                checkpoint['errors'].append({
                    'date_range': f"{start_date} to {end_date}",
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                self._save_checkpoint(location_id, checkpoint)
                raise
        
        # Mark as complete
        checkpoint['status'] = 'completed'
        checkpoint['completed_at'] = datetime.now().isoformat()
        self._save_checkpoint(location_id, checkpoint)
        
        # Print statistics
        duration = time.time() - start_time
        stats = self._get_collection_stats(checkpoint, duration)
        self._print_stats(stats)
        
        return stats
    
    def _load_checkpoint(self, location_id: str, resume: bool) -> Dict:
        """Load or create checkpoint"""
        checkpoint_path = get_checkpoint_path(
            self.resume_config['checkpoint_dir'],
            location_id,
            'historical'
        )
        
        if resume and Path(checkpoint_path).exists():
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = load_json(checkpoint_path)
            self.logger.info(f"Resuming from: {checkpoint['last_completed_date']}")
            return checkpoint
        else:
            self.logger.info("Starting new collection")
            return {
                'location_id': location_id,
                'collection_type': 'historical',
                'status': 'in_progress',
                'started_at': datetime.now().isoformat(),
                'last_completed_date': None,
                'completed_ranges': [],
                'total_api_calls': 0,
                'errors': []
            }
    
    def _save_checkpoint(self, location_id: str, checkpoint: Dict):
        """Save checkpoint"""
        checkpoint_path = get_checkpoint_path(
            self.resume_config['checkpoint_dir'],
            location_id,
            'historical'
        )
        checkpoint['updated_at'] = datetime.now().isoformat()
        save_json(checkpoint, checkpoint_path)
    
    def _get_collection_ranges(self, checkpoint: Dict) -> List[tuple]:
        """Get date ranges that need to be collected"""
        start_date = self.data_config['historical']['start_date']
        end_date = self.data_config['historical']['end_date']
        chunk_days = self.data_config['historical']['chunk_size_days']
        
        # Get all ranges
        all_ranges = get_date_range(start_date, end_date, chunk_days)
        
        # If no checkpoint, return all
        if not checkpoint['last_completed_date']:
            return all_ranges
        
        # Filter already completed ranges
        last_completed = checkpoint['last_completed_date']
        
        # Return ranges after last completed
        remaining = []
        for start, end in all_ranges:
            if start > last_completed:
                remaining.append((start, end))
        
        return remaining
    
    def _get_collection_stats(self, checkpoint: Dict, duration: float) -> Dict:
        """Get collection statistics"""
        return {
            'location_id': checkpoint['location_id'],
            'status': checkpoint['status'],
            'started_at': checkpoint['started_at'],
            'completed_at': checkpoint.get('completed_at'),
            'duration_seconds': duration,
            'duration_formatted': format_duration(duration),
            'total_ranges': len(checkpoint['completed_ranges']),
            'total_api_calls': checkpoint['total_api_calls'],
            'total_errors': len(checkpoint['errors']),
            'rate_limiter_stats': self.rate_limiter.get_stats()
        }
    
    def _print_stats(self, stats: Dict):
        """Print collection statistics"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COLLECTION COMPLETE!")
        self.logger.info("=" * 80)
        self.logger.info(f"Location: {stats['location_id']}")
        self.logger.info(f"Status: {stats['status']}")
        self.logger.info(f"Duration: {stats['duration_formatted']}")
        self.logger.info(f"Date ranges collected: {stats['total_ranges']}")
        self.logger.info(f"Total API calls: {stats['total_api_calls']}")
        self.logger.info(f"Errors: {stats['total_errors']}")
        self.logger.info("-" * 80)
        self.logger.info("Rate Limiter Stats:")
        rl_stats = stats['rate_limiter_stats']
        self.logger.info(f"  Total calls: {rl_stats['total_calls']}")
        self.logger.info(f"  Calls today: {rl_stats['calls_today']}")
        self.logger.info(f"  Calls this month: {rl_stats['calls_this_month']}")
        self.logger.info(f"  Remaining today: {rl_stats['remaining_today']}")
        self.logger.info(f"  Remaining this month: {rl_stats['remaining_this_month']}")
        self.logger.info("=" * 80)