"""
API routes for the AI Counter backend.
"""
import os
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from app.ocr import OCRProcessor

logger = logging.getLogger(__name__)

# Initialize OCR processor
ocr_processor = OCRProcessor()

router = APIRouter(prefix="/api", tags=["api"])

# Create uploads directory if it doesn't exist
# Use absolute path relative to server directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Upload directory: {UPLOAD_DIR.absolute()}")


@router.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    utility_type: str = Form(...),
):
    """
    Upload an image file for OCR processing.
    
    Args:
        file: Image file to upload
        utility_type: Type of utility meter (gas, water, electricity)
    
    Returns:
        JSON response with file information
    """
    logger.info("=" * 50)
    logger.info("Received upload request")
    logger.info(f"Utility type received: '{utility_type}' (type: {type(utility_type)})")
    logger.info(f"File filename: {file.filename}")
    logger.info(f"File content_type: {file.content_type}")
    logger.info(f"File size: {file.size if hasattr(file, 'size') else 'unknown'}")
    
    # Validate utility type
    valid_types = ["gas", "water", "electricity"]
    utility_type_lower = utility_type.lower() if utility_type else ""
    logger.info(f"Utility type (lowercase): '{utility_type_lower}'")
    logger.info(f"Valid types: {valid_types}")
    
    if utility_type_lower not in valid_types:
        logger.error(f"Invalid utility_type: '{utility_type}' (lowercase: '{utility_type_lower}')")
        logger.error(f"Valid types are: {valid_types}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid utility_type. Must be one of: {', '.join(valid_types)}. Received: '{utility_type}'",
        )
    
    # Validate file type
    logger.info(f"Validating file content_type: {file.content_type}")
    logger.info(f"File filename: {file.filename}")
    
    # Check content-type first
    is_valid_image = False
    if file.content_type and file.content_type.startswith("image/"):
        is_valid_image = True
        logger.info("File validated by content-type")
    else:
        # Fallback: check file extension
        if file.filename:
            valid_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.heif', '.gif', '.webp']
            file_extension = Path(file.filename).suffix.lower()
            if file_extension in valid_extensions:
                is_valid_image = True
                logger.info(f"File validated by extension: {file_extension}")
            else:
                logger.error(f"Invalid file extension: {file_extension}")
        else:
            logger.error("No filename provided and content_type is invalid")
    
    if not is_valid_image:
        logger.error(f"Invalid file. content_type: {file.content_type}, filename: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File must be an image. Received content_type: {file.content_type}, filename: {file.filename}",
        )
    
    logger.info("Validation passed, processing file...")
    
    try:
        # Generate unique filename
        file_extension = Path(file.filename).suffix if file.filename else ".jpg"
        import uuid
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        file_size = len(content)
        
        logger.info(
            f"Image saved: {unique_filename}, "
            f"type: {utility_type}, size: {file_size} bytes"
        )
        
        # Perform OCR recognition
        logger.info("Starting OCR recognition...")
        reading_value_str, confidence_score, raw_text, normalization_status = ocr_processor.recognize_digits(
            file_path, min_confidence=60, utility_type=utility_type_lower
        )
        
        if reading_value_str is not None:
            logger.info(
                f"OCR successful: reading_value={reading_value_str}, "
                f"confidence={confidence_score}%, status={normalization_status}"
            )
        else:
            logger.warning(
                f"OCR failed: confidence={confidence_score}%, "
                f"status={normalization_status}, raw_text='{raw_text[:100] if raw_text else None}'"
            )
        
        logger.info("=" * 50)
        
        # Build response
        response_data = {
            "status": "success",
            "message": "Image uploaded and processed successfully",
            "file_path": str(file_path),
            "filename": unique_filename,
            "utility_type": utility_type.lower(),
            "file_size": file_size,
            "content_type": file.content_type,
        }
        
        # Add OCR results if available
        if reading_value_str is not None:
            # Return both string and integer representations
            response_data["reading_value"] = reading_value_str  # String with leading zeros
            response_data["reading_value_int"] = int(reading_value_str)  # Integer for calculations
            response_data["confidence_score"] = confidence_score
            response_data["normalization_status"] = normalization_status
            if raw_text:
                response_data["raw_text"] = raw_text
            
            # Adjust message based on normalization status
            if normalization_status == "ok":
                response_data["message"] = "Image uploaded and processed successfully"
            elif normalization_status == "decreased":
                response_data["message"] = "Reading detected, but value decreased from previous reading"
            elif normalization_status == "too_large_jump":
                response_data["message"] = "Reading detected, but value increased abnormally - please verify"
        else:
            response_data["reading_value"] = None
            response_data["reading_value_int"] = None
            response_data["confidence_score"] = confidence_score
            response_data["normalization_status"] = normalization_status
            response_data["raw_text"] = raw_text
            
            # Provide helpful error message based on status
            status_messages = {
                "no_digits": "No digits found in image",
                "bad_length": "Reading length doesn't match expected format",
                "file_not_found": "Image file not found",
                "error": "OCR processing error occurred",
            }
            response_data["message"] = status_messages.get(
                normalization_status, 
                "Image uploaded but OCR recognition failed"
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data,
        )
    
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}",
        )

