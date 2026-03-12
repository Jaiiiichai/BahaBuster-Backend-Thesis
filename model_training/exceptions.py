"""Custom exceptions used across the training pipeline."""


class ModelNotFoundError(Exception):
    """Raised when no trained model is available for the requested barangay."""

    def __init__(self, barangay: str, available: list[str]):
        self.barangay = barangay
        self.available = available
        super().__init__(f"No trained model for barangay '{barangay}'.")
