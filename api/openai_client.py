"""OpenAI integration for flood image analysis and description generation."""
import logging
import os
import json
from typing import Any, Dict, Optional
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


def _safe_json_dict(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from model text output, tolerating fenced responses."""

    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object response from OpenAI.")
    return parsed


def generate_alert_copy_from_flood_data(
    barangay: str,
    prediction: Dict[str, Any],
    recent_reports: Optional[list[dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate alert title/description/severity from current barangay flood data."""

    try:
        client = get_openai_client()
        reports = recent_reports or []
        context_payload = {
            "barangay": barangay,
            "prediction": prediction,
            "recent_reports": reports[:3],
        }

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=400,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a disaster-risk assistant for barangay flood alerts. "
                        "Write concise, clear, public-safety alert copy."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Create an alert draft from this JSON data. Respond only as valid JSON with this schema: "
                        '{"title": string, "description": string, "severity": "critical"|"moderate"|"low", "reason": string}. '
                        "Do not invent fields and keep description under 320 characters. "
                        f"Data: {json.dumps(context_payload, ensure_ascii=True)}"
                    ),
                },
            ],
        )

        response_text = response.choices[0].message.content
        parsed = _safe_json_dict(response_text)

        title = str(parsed.get("title") or "Flood Alert").strip()[:120]
        description = str(parsed.get("description") or "Monitor flood conditions and stay prepared.").strip()[:320]
        severity = str(parsed.get("severity") or "low").strip().lower()
        reason = str(parsed.get("reason") or "Generated from prediction data.").strip()[:240]

        if severity not in {"critical", "moderate", "low"}:
            severity = "low"

        return {
            "title": title or "Flood Alert",
            "description": description or "Monitor flood conditions and stay prepared.",
            "severity": severity,
            "reason": reason or "Generated from prediction data.",
        }

    except APIError as api_error:
        logger.error(f"OpenAI API error during alert generation: {api_error}")
        raise OpenAIIntegrationError(f"OpenAI API error: {str(api_error)}") from api_error
    except OpenAIIntegrationError:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error in alert generation: {exc}", exc_info=True)
        raise OpenAIIntegrationError(f"Failed to generate alert copy: {str(exc)}") from exc


