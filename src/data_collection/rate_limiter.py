"""
Rate Limiter for Open-Meteo API
Manages API call limits to stay within free tier restrictions
"""

import time
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Optional
from src.utils.logger import get_logger


class RateLimiter:
    """
    Manages API rate limits with safety buffer
    Tracks calls per minute, hour, day, and month
    """
    
    def __init__(self, config: Dict, logger=None):
        """
        Initialize rate limiter
        
        Args:
            config: Rate limit configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or get_logger()
        
        # Extract limits with safety buffer
        safety = config.get('safety_buffer', 0.8)
        self.limits = {
            'minute': int(config.get('calls_per_minute', 600) * safety),
            'hour': int(config.get('calls_per_hour', 5000) * safety),
            'day': int(config.get('calls_per_day', 10000) * safety),
            'month': int(config.get('calls_per_month', 300000) * safety)
        }
        
        # Track API calls with timestamps
        self.call_history = {
            'minute': deque(maxlen=self.limits['minute']),
            'hour': deque(maxlen=self.limits['hour']),
            'day': [],
            'month': []
        }
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'total_wait_time': 0.0,
            'calls_today': 0,
            'calls_this_month': 0
        }
        
        self.logger.info(f"Rate limiter initialized with limits: {self.limits}")
    
    def wait_if_needed(self):
        """
        Check rate limits and wait if necessary
        Raises exception if daily or monthly limit would be exceeded
        """
        now = datetime.now()
        
        # Clean old entries
        self._clean_old_entries(now)
        
        # Check monthly limit (hard stop)
        if len(self.call_history['month']) >= self.limits['month']:
            raise Exception("Monthly API limit reached! Cannot make more calls this month.")
        
        # Check daily limit (hard stop)
        if len(self.call_history['day']) >= self.limits['day']:
            raise Exception("Daily API limit reached! Cannot make more calls today.")
        
        # Check hour limit (wait)
        if len(self.call_history['hour']) >= self.limits['hour']:
            wait_time = self._calculate_wait_time('hour')
            self.logger.warning(f"Hour limit reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            self._clean_old_entries(datetime.now())
        
        # Check minute limit (wait)
        if len(self.call_history['minute']) >= self.limits['minute']:
            wait_time = self._calculate_wait_time('minute')
            self.logger.warning(f"Minute limit reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            self._clean_old_entries(datetime.now())
    
    def record_call(self):
        """Record an API call"""
        now = datetime.now()
        
        # Add to all tracking queues
        self.call_history['minute'].append(now)
        self.call_history['hour'].append(now)
        self.call_history['day'].append(now)
        self.call_history['month'].append(now)
        
        # Update statistics
        self.stats['total_calls'] += 1
        self.stats['calls_today'] = len(self.call_history['day'])
        self.stats['calls_this_month'] = len(self.call_history['month'])
    
    def _clean_old_entries(self, now: datetime):
        """Remove expired entries from tracking"""
        # Clean minute (keep last 60 seconds)
        cutoff_minute = now - timedelta(seconds=60)
        while self.call_history['minute'] and self.call_history['minute'][0] < cutoff_minute:
            self.call_history['minute'].popleft()
        
        # Clean hour (keep last 60 minutes)
        cutoff_hour = now - timedelta(minutes=60)
        while self.call_history['hour'] and self.call_history['hour'][0] < cutoff_hour:
            self.call_history['hour'].popleft()
        
        # Clean day (keep last 24 hours)
        cutoff_day = now - timedelta(hours=24)
        self.call_history['day'] = [t for t in self.call_history['day'] if t >= cutoff_day]
        
        # Clean month (keep last 30 days)
        cutoff_month = now - timedelta(days=30)
        self.call_history['month'] = [t for t in self.call_history['month'] if t >= cutoff_month]
    
    def _calculate_wait_time(self, period: str) -> float:
        """
        Calculate how long to wait before next call
        
        Args:
            period: 'minute' or 'hour'
            
        Returns:
            Wait time in seconds
        """
        if not self.call_history[period]:
            return 0.0
        
        oldest_call = self.call_history[period][0]
        now = datetime.now()
        
        if period == 'minute':
            elapsed = (now - oldest_call).total_seconds()
            return max(0, 60 - elapsed + 1)  # +1 for safety
        else:  # hour
            elapsed = (now - oldest_call).total_seconds()
            return max(0, 3600 - elapsed + 1)
    
    def get_remaining_calls(self) -> Dict[str, int]:
        """
        Get remaining API calls for each period
        
        Returns:
            Dictionary with remaining calls
        """
        now = datetime.now()
        self._clean_old_entries(now)
        
        return {
            'minute': self.limits['minute'] - len(self.call_history['minute']),
            'hour': self.limits['hour'] - len(self.call_history['hour']),
            'day': self.limits['day'] - len(self.call_history['day']),
            'month': self.limits['month'] - len(self.call_history['month'])
        }
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        remaining = self.get_remaining_calls()
        
        return {
            'total_calls': self.stats['total_calls'],
            'calls_today': self.stats['calls_today'],
            'calls_this_month': self.stats['calls_this_month'],
            'remaining_today': remaining['day'],
            'remaining_this_month': remaining['month'],
            'limits': self.limits
        }
    
    def print_stats(self):
        """Print current statistics"""
        stats = self.get_stats()
        
        self.logger.info("=" * 60)
        self.logger.info("Rate Limiter Statistics")
        self.logger.info("=" * 60)
        self.logger.info(f"Total API calls: {stats['total_calls']}")
        self.logger.info(f"Calls today: {stats['calls_today']} / {self.limits['day']}")
        self.logger.info(f"Calls this month: {stats['calls_this_month']} / {self.limits['month']}")
        self.logger.info(f"Remaining today: {stats['remaining_today']}")
        self.logger.info(f"Remaining this month: {stats['remaining_this_month']}")
        self.logger.info("=" * 60)
