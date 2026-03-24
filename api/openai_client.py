"""OpenAI integration for flood image analysis and description generation."""
import logging
import os
import json
from typing import Dict, Optional
from openai import OpenAI, APIError
import base64
from PIL import Image
import io

logger = logging.getLogger(__name__)


class OpenAIIntegrationError(Exception):
    """Raised when OpenAI integration fails."""
    pass


def get_openai_client() -> OpenAI:
    """
    Get or initialize OpenAI client.
    
    Returns:
        OpenAI client instance
        
    Raises:
        OpenAIIntegrationError: If API key not configured
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIIntegrationError("OPENAI_API_KEY not set in environment variables")
    
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as exc:
        raise OpenAIIntegrationError(f"Failed to initialize OpenAI client: {str(exc)}") from exc


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string for OpenAI API.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded image string
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return base64.standard_b64encode(buffer.read()).decode("utf-8")


def analyze_flood_image(
    image: Image.Image,
    barangay: Optional[str] = None
) -> Dict:
    """
    Use GPT-4o-mini to analyze flood image and generate description.
    
    Args:
        image: PIL Image object
        barangay: Optional barangay name for context
        
    Returns:
        Dictionary with:
            - is_flood: Whether flood was detected
            - description: Detailed description
            - short_summary: Brief summary
            - severity: Flood severity (mild/moderate/severe/none)
            - confidence: Confidence level (0-100)
            - has_people: Whether people are at risk
            - has_structures: Whether structures are affected
            - recommendations: List of recommended actions
            
    Raises:
        OpenAIIntegrationError: If API call fails
    """
    try:
        client = get_openai_client()
        
        # Convert image to base64
        image_base64 = image_to_base64(image)
        
        # Prepare context
        context = ""
        if barangay:
            context = f"Location: {barangay}. "
        
        logger.info("Sending image to GPT-4o-mini for analysis...")
        
        # Call GPT-4o-mini with vision using OpenAI Chat API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": f"""{context}Analyze this image and determine if it shows flooding. Respond ONLY with valid JSON (no markdown, no other text):

{{
    "is_flood": boolean - true if image shows flooding or large water accumulation,
    "description": "2-3 sentence detailed description of flood conditions if it's a flood and answer like you are a resident reporting the flood and humanize it like a real person would, otherwise explanation why it's not",
    "short_summary": "1 sentence summary (max 80 characters)",
    "severity": "mild", "moderate", or "severe" (only if is_flood is true, else "none"),
    "confidence": number 0-100 indicating how confident you are,
    "has_people": boolean - are people visible and potentially at risk,
    "has_structures": boolean - are buildings or infrastructure affected,
    "water_type": "description of water type: clear, murky, turbulent, stagnant, brown/muddy, or if no flood: 'none'",
    "recommendations": ["action 1", "action 2", "action 3"] - practical actions for local authorities if flood, else empty array
}}"""
                        }
                    ],
                }
            ],
        )
        
        # Extract response text
        response_text = response.choices[0].message.content
        logger.info(f"GPT response received: {len(response_text)} characters")
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
            logger.info(f"Successfully parsed GPT response: is_flood={result.get('is_flood')}")
        except json.JSONDecodeError as parse_error:
            logger.warning(f"Failed to parse GPT response as JSON: {response_text}")
            logger.warning(f"Parse error: {parse_error}")
            # Fallback response
            result = {
                "is_flood": False,
                "description": "Unable to analyze image. Please try again or upload a clearer image.",
                "short_summary": "Analysis unavailable",
                "severity": "none",
                "confidence": 0,
                "has_people": False,
                "has_structures": False,
                "water_type": "unknown",
                "recommendations": []
            }
        
        return result
        
    except APIError as api_error:
        logger.error(f"OpenAI API error: {api_error}")
        raise OpenAIIntegrationError(f"OpenAI API error: {str(api_error)}") from api_error
    except OpenAIIntegrationError:
        # Re-raise our custom errors
        raise
    except Exception as exc:
        logger.error(f"Unexpected error in image analysis: {exc}", exc_info=True)
        raise OpenAIIntegrationError(f"Failed to analyze image: {str(exc)}") from exc


