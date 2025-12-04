"""
Helper utilities for irrigation predictor
Common functions used across modules
"""

import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary with configuration
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        file_path: Path to JSON file
        indent: JSON indentation
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def get_date_range(start_date: str, end_date: str, chunk_days: int = 30) -> List[Tuple[str, str]]:
    """
    Split date range into chunks
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        chunk_days: Size of each chunk in days
        
    Returns:
        List of (start, end) date tuples
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    chunks = []
    current = start
    
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d")
        ))
        current = chunk_end
    
    return chunks


def ensure_dir(directory: str):
    """
    Ensure directory exists
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_location_dir(base_dir: str, location_id: str) -> str:
    """
    Get directory path for location
    
    Args:
        base_dir: Base directory
        location_id: Location identifier
        
    Returns:
        Directory path for location
    """
    location_dir = Path(base_dir) / location_id
    ensure_dir(location_dir)
    return str(location_dir)


def format_file_size(bytes_size: int) -> str:
    """
    Format bytes to human readable size
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def calculate_progress(current: int, total: int) -> float:
    """
    Calculate progress percentage
    
    Args:
        current: Current count
        total: Total count
        
    Returns:
        Progress percentage (0-100)
    """
    if total == 0:
        return 0.0
    return (current / total) * 100


def format_duration(seconds: float) -> str:
    """
    Format duration in human readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def validate_date_format(date_str: str) -> bool:
    """
    Validate date string format (YYYY-MM-DD)
    
    Args:
        date_str: Date string
        
    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def get_checkpoint_path(checkpoint_dir: str, location_id: str, data_type: str) -> str:
    """
    Get checkpoint file path
    
    Args:
        checkpoint_dir: Checkpoint directory
        location_id: Location identifier
        data_type: Type of data ('historical' or 'forecast')
        
    Returns:
        Checkpoint file path
    """
    ensure_dir(checkpoint_dir)
    return str(Path(checkpoint_dir) / f"{location_id}_{data_type}_checkpoint.json")
