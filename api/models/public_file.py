"""
Pydantic models for public file operations.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List


class PublicFileUploadResponse(BaseModel):
    """Response model for successful file upload."""
    success: bool = True
    filename: str = Field(..., description="Name of the uploaded file")
    file_path: str = Field(..., description="Relative path to access the file")
    url: str = Field(..., description="Full URL to access the file")
    size: int = Field(..., description="File size in bytes")
    message: str = Field(..., description="Success message")


class PublicFileInfo(BaseModel):
    """Model for public file information."""
    filename: str = Field(..., description="Name of the file")
    file_path: str = Field(..., description="Relative path to access the file")
    url: str = Field(..., description="Full URL to access the file")
    size: int = Field(..., description="File size in bytes")
    modified: datetime = Field(..., description="Last modified timestamp")


class PublicFileListResponse(BaseModel):
    """Response model for listing public files."""
    success: bool = True
    total: int = Field(..., description="Total number of files")
    files: List[PublicFileInfo] = Field(..., description="List of files")


class PublicFileDeleteResponse(BaseModel):
    """Response model for file deletion."""
    success: bool = True
    filename: str = Field(..., description="Name of the deleted file")
    message: str = Field(..., description="Success message")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    detail: str = Field(..., description="Error message")
