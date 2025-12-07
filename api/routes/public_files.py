"""
Public file management API endpoints.
"""
import os
import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, status
from fastapi.responses import JSONResponse

from api.models.public_file import (
    PublicFileUploadResponse,
    PublicFileListResponse,
    PublicFileInfo,
    PublicFileDeleteResponse,
    ErrorResponse,
)


router = APIRouter()

# Constants
PUBLIC_DIR = "public"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
FORBIDDEN_EXTENSIONS = {'.exe', '.bat', '.sh', '.cmd', '.com', '.msi', '.scr', '.pif'}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal and invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)

    # Remove or replace invalid characters
    filename = re.sub(r'[^\w\s\-\.]', '_', filename)

    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')

    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"

    return filename


def is_allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed.

    Args:
        filename: Filename to check

    Returns:
        True if allowed, False otherwise
    """
    ext = os.path.splitext(filename.lower())[1]
    return ext not in FORBIDDEN_EXTENSIONS


def get_file_url(filename: str, request) -> str:
    """
    Generate full URL for a public file.

    Args:
        filename: Name of the file
        request: FastAPI request object

    Returns:
        Full URL to access the file
    """
    base_url = str(request.base_url).rstrip('/')
    return f"{base_url}/public/{filename}"


@router.post(
    "/public/files",
    response_model=PublicFileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload file to public folder",
    description="Upload a file to the public folder with an optional custom name. Files are accessible to everyone.",
    responses={
        201: {"description": "File uploaded successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Server error", "model": ErrorResponse},
    }
)
async def upload_public_file(
    file: UploadFile = File(..., description="File to upload"),
    custom_name: Optional[str] = Form(None, description="Custom filename (optional, keeps original if not provided)")
):
    """
    Upload a file to the public folder.

    Args:
        file: File to upload
        custom_name: Optional custom filename (will be sanitized)

    Returns:
        PublicFileUploadResponse with file details and access URL

    Raises:
        400: Invalid file or filename
        500: Server error
    """
    try:
        # Determine filename
        if custom_name:
            # Get extension from original file
            ext = os.path.splitext(file.filename)[1]
            # Sanitize custom name and add extension
            filename = sanitize_filename(custom_name)
            if not filename.endswith(ext):
                filename = f"{filename}{ext}"
        else:
            filename = sanitize_filename(file.filename)

        # Validate file extension
        if not is_allowed_file(filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Forbidden extensions: {', '.join(FORBIDDEN_EXTENSIONS)}"
            )

        # Create public directory if it doesn't exist
        os.makedirs(PUBLIC_DIR, exist_ok=True)

        # Check if file already exists
        file_path = os.path.join(PUBLIC_DIR, filename)
        if os.path.exists(file_path):
            # Add timestamp to make filename unique
            name_part, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{name_part}_{timestamp}{ext}"
            file_path = os.path.join(PUBLIC_DIR, filename)

        # Read and validate file size
        content = await file.read()
        file_size = len(content)

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024)}MB"
            )

        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )

        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Generate response
        from fastapi import Request
        from starlette.requests import Request as StarletteRequest

        # Since we don't have request object in parameters, construct URL manually
        file_url = f"/public/{filename}"

        return PublicFileUploadResponse(
            success=True,
            filename=filename,
            file_path=file_url,
            url=file_url,
            size=file_size,
            message=f"File '{filename}' uploaded successfully to public folder"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )


@router.get(
    "/public/files",
    response_model=PublicFileListResponse,
    summary="List all public files",
    description="Retrieve a list of all files in the public folder with their details.",
)
async def list_public_files():
    """
    List all files in the public folder.

    Returns:
        PublicFileListResponse with list of all public files

    Raises:
        500: Server error
    """
    try:
        # Create public directory if it doesn't exist
        os.makedirs(PUBLIC_DIR, exist_ok=True)

        files = []

        # List all files in public directory
        for filename in os.listdir(PUBLIC_DIR):
            file_path = os.path.join(PUBLIC_DIR, filename)

            # Skip directories and hidden files
            if os.path.isfile(file_path) and not filename.startswith('.'):
                stat_info = os.stat(file_path)

                files.append(PublicFileInfo(
                    filename=filename,
                    file_path=f"/public/{filename}",
                    url=f"/public/{filename}",
                    size=stat_info.st_size,
                    modified=datetime.fromtimestamp(stat_info.st_mtime)
                ))

        # Sort by modified date (newest first)
        files.sort(key=lambda x: x.modified, reverse=True)

        return PublicFileListResponse(
            success=True,
            total=len(files),
            files=files
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list public files: {str(e)}"
        )


@router.delete(
    "/public/files/{filename}",
    response_model=PublicFileDeleteResponse,
    summary="Delete a public file",
    description="Delete a specific file from the public folder.",
    responses={
        200: {"description": "File deleted successfully"},
        404: {"description": "File not found", "model": ErrorResponse},
        500: {"description": "Server error", "model": ErrorResponse},
    }
)
async def delete_public_file(filename: str):
    """
    Delete a file from the public folder.

    Args:
        filename: Name of the file to delete

    Returns:
        PublicFileDeleteResponse confirming deletion

    Raises:
        404: File not found
        500: Server error
    """
    try:
        # Sanitize filename to prevent directory traversal
        filename = sanitize_filename(filename)
        file_path = os.path.join(PUBLIC_DIR, filename)

        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{filename}' not found in public folder"
            )

        # Check if it's a file (not a directory)
        if not os.path.isfile(file_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'{filename}' is not a file"
            )

        # Delete file
        os.remove(file_path)

        return PublicFileDeleteResponse(
            success=True,
            filename=filename,
            message=f"File '{filename}' deleted successfully from public folder"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}"
        )
