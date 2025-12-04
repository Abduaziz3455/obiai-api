"""
Open-Meteo API Client
Handles all API requests with retry logic and error handling
"""

import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.utils.logger import get_logger


class OpenMeteoClient:
    """
    Client for Open-Meteo API
    Handles historical and forecast weather data requests
    """
    
    def __init__(self, config: Dict, rate_limiter=None, logger=None):
        """
        Initialize API client
        
        Args:
            config: API configuration
            rate_limiter: RateLimiter instance
            logger: Logger instance
        """
        self.config = config
        self.base_url = config.get('base_url', 'https://api.open-meteo.com/v1')
        self.timeout = config.get('timeout', 30)
        self.rate_limiter = rate_limiter
        self.logger = logger or get_logger()
        
        # Retry settings
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        self.logger.info(f"API Client initialized: {self.base_url}")
    
    def get_historical(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        hourly_params: List[str],
        daily_params: Optional[List[str]] = None,
        timezone: str = "UTC"
    ) -> Dict[str, Any]:
        """
        Fetch historical weather data

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            hourly_params: List of hourly parameters
            daily_params: List of daily parameters (optional)
            timezone: Timezone for data

        Returns:
            API response as dictionary
        """
        # Historical API uses different base URL
        endpoint = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date,
            'end_date': end_date,
            'timezone': timezone
        }
        
        # Add parameters
        if hourly_params:
            params['hourly'] = ','.join(hourly_params)
        if daily_params:
            params['daily'] = ','.join(daily_params)
        
        return self._make_request(endpoint, params, "historical")
    
    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        hourly_params: List[str],
        daily_params: Optional[List[str]] = None,
        forecast_days: int = 16,
        past_days: int = 0,
        timezone: str = "UTC"
    ) -> Dict[str, Any]:
        """
        Fetch weather forecast data
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            hourly_params: List of hourly parameters
            daily_params: List of daily parameters (optional)
            forecast_days: Number of forecast days (max 16)
            past_days: Number of past days to include (max 92)
            timezone: Timezone for data
            
        Returns:
            API response as dictionary
        """
        endpoint = f"{self.base_url}/forecast"
        
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'forecast_days': min(forecast_days, 16),
            'timezone': timezone
        }
        
        if past_days > 0:
            params['past_days'] = min(past_days, 92)
        
        # Add parameters
        if hourly_params:
            params['hourly'] = ','.join(hourly_params)
        if daily_params:
            params['daily'] = ','.join(daily_params)
        
        return self._make_request(endpoint, params, "forecast")
    
    def _make_request(
        self,
        endpoint: str,
        params: Dict,
        request_type: str
    ) -> Dict[str, Any]:
        """
        Make API request with retry logic
        
        Args:
            endpoint: API endpoint URL
            params: Query parameters
            request_type: Type of request (for logging)
            
        Returns:
            API response as dictionary
            
        Raises:
            Exception: If request fails after all retries
        """
        # Check rate limits
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()
        
        # Try request with retries
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Making {request_type} request (attempt {attempt + 1}/{self.max_retries})")
                self.logger.debug(f"URL: {endpoint}")
                self.logger.debug(f"Params: {params}")
                
                # Make request
                response = requests.get(
                    endpoint,
                    params=params,
                    timeout=self.timeout
                )
                
                # Check response status
                if response.status_code == 200:
                    data = response.json()
                    
                    # Validate response
                    if self._validate_response(data):
                        # Record successful call
                        if self.rate_limiter:
                            self.rate_limiter.record_call()
                        
                        self.logger.debug(f"{request_type.capitalize()} request successful")
                        return data
                    else:
                        raise ValueError("Invalid response format")
                
                elif response.status_code == 429:
                    # Rate limit exceeded
                    self.logger.warning(f"Rate limit exceeded (429). Waiting {self.retry_delay * 2} seconds...")
                    time.sleep(self.retry_delay * 2)
                    
                else:
                    # Other error
                    self.logger.error(f"API error: {response.status_code} - {response.text}")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise Exception(f"API request failed: {response.status_code}")
            
            except requests.exceptions.Timeout:
                self.logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise Exception("Request timed out after all retries")
            
            except requests.exceptions.ConnectionError as e:
                self.logger.warning(f"Connection error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise Exception("Connection failed after all retries")
            
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
        
        raise Exception("Request failed after all retries")
    
    def _validate_response(self, data: Dict) -> bool:
        """
        Validate API response format
        
        Args:
            data: Response data
            
        Returns:
            True if valid, False otherwise
        """
        # Check for error
        if isinstance(data, dict) and data.get('error'):
            self.logger.error(f"API returned error: {data.get('reason', 'Unknown error')}")
            return False
        
        # Check required fields
        required_fields = ['latitude', 'longitude']
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        # Check if has data
        if 'hourly' not in data and 'daily' not in data:
            self.logger.error("Response has no hourly or daily data")
            return False
        
        return True
    
    def test_connection(self) -> bool:
        """
        Test API connection with a simple request
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info("Testing API connection...")
            
            # Simple test request (Karmana location, 1 day)
            data = self.get_historical(
                latitude=40.48,
                longitude=65.355,
                start_date="2024-11-01",
                end_date="2024-11-01",
                hourly_params=['temperature_2m']
            )
            
            self.logger.info("✓ API connection successful")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ API connection failed: {e}")
            return False
