"""Image validation and processing utilities."""
import io
from typing import Tuple, Dict
from PIL import Image


class ImageValidationError(Exception):
    """Raised when image validation fails."""
    pass


def validate_image(image_bytes: bytes, max_size_mb: float = 5.0) -> Tuple[Image.Image, Dict]:
    """
    Validate and process an uploaded image.
    
    Args:
        image_bytes: Raw image bytes
        max_size_mb: Maximum allowed file size in MB
    
    Returns:
        Tuple of (PIL Image object, metadata dict)
        
    Raises:
        ImageValidationError: If image is invalid
    """
    # Check file size
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ImageValidationError(f"Image too large: {size_mb:.2f}MB (max: {max_size_mb}MB)")
    
    # Try to open image
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ImageValidationError(f"Failed to read image: {str(exc)}") from exc
    
    # Check format
    if image.format not in ["JPEG", "PNG", "BMP", "GIF", "WEBP"]:
        raise ImageValidationError(f"Unsupported image format: {image.format}")
    
    # Check dimensions
    width, height = image.size
    if width < 100 or height < 100:
        raise ImageValidationError(f"Image too small: {width}x{height} (min: 100x100)")
    if width > 10000 or height > 10000:
        raise ImageValidationError(f"Image too large: {width}x{height} (max: 10000x10000)")
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    metadata = {
        "format": image.format,
        "size": {"width": width, "height": height},
        "file_size_mb": round(size_mb, 2),
    }
    
    return image, metadata

