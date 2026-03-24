"""Image analysis endpoints for flood detection with AI-powered descriptions."""
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import Optional

from ..image_utils import validate_image, ImageValidationError
from ..openai_client import analyze_flood_image, OpenAIIntegrationError
from ..schemas import ImageAnalysisResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["image-analysis"])


@router.post("/analyze-flood-image", response_model=ImageAnalysisResponse)
async def analyze_flood_image_endpoint(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, BMP, GIF, WebP)"),
    barangay: Optional[str] = Form(None, description="Optional barangay name for context")
):
    """
    Analyze an uploaded image for flood detection using GPT-4o-mini vision.
    
    Steps:
    1. Validates image format and size
    2. Sends image to GPT-4o-mini for analysis
    3. GPT determines if flood is present
    4. Returns comprehensive analysis with recommendations
    
    **Supported formats**: JPEG, PNG, BMP, GIF, WebP
    **Maximum file size**: 5MB
    **Minimum resolution**: 100x100 pixels
    **Maximum resolution**: 10000x10000 pixels
    """
    try:
        # Read file
        image_bytes = await file.read()
        
        if not image_bytes:
            raise ImageValidationError("Empty file uploaded")
        
        logger.info(f"Analyzing image. File size: {len(image_bytes)} bytes")
        
        # Validate image
        try:
            image, image_info = validate_image(image_bytes)
        except ImageValidationError as exc:
            logger.warning(f"Image validation failed: {exc}")
            raise HTTPException(
                status_code=400,
                detail=f"Image validation failed: {str(exc)}"
            ) from exc
        
        # Analyze with GPT-4o-mini
        try:
            analysis = analyze_flood_image(image=image, barangay=barangay)
        except OpenAIIntegrationError as exc:
            logger.error(f"GPT analysis failed: {exc}")
            raise HTTPException(
                status_code=503,
                detail=f"AI analysis service temporarily unavailable: {str(exc)}"
            ) from exc
        
        # Build response
        response = ImageAnalysisResponse(
            is_flood=analysis["is_flood"],
            flood_classification="FLOOD" if analysis["is_flood"] else "NO FLOOD",
            water_percentage=0.0 if not analysis["is_flood"] else 50.0,  # GPT handles this
            description=analysis["description"],
            short_summary=analysis["short_summary"],
            severity=analysis["severity"],
            confidence=analysis["confidence"],
            has_people=analysis["has_people"],
            has_structures=analysis["has_structures"],
            water_type=analysis["water_type"],
            recommendations=analysis["recommendations"],
            image_info=image_info,
            barangay=barangay,
        )
        
        logger.info(f"Analysis complete: is_flood={analysis['is_flood']}, severity={analysis['severity']}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error in flood image analysis: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during analysis: {str(exc)}"
        ) from exc


@router.get("/health/image-analysis")
def image_analysis_health():
    """Health check for image analysis service."""
    try:
        from ..openai_client import get_openai_client
        client = get_openai_client()
        
        return {
            "status": "healthy",
            "service": "image-analysis",
            "openai_configured": True,
            "model": "gpt-4o-mini",
            "message": "Image analysis service is operational"
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "service": "image-analysis",
            "openai_configured": False,
            "error": str(exc),
            "message": "Image analysis service is not operational"
        }

