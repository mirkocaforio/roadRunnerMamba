from datetime import datetime


class OutOfMemoryError(Exception):
    """Custom exception for handling out of memory errors during processing."""
    
    def __init__(self, message="Out of memory during processing."):
        super().__init__(message)
        self.timestamp = datetime.now()
        self.message = message
    
    def __str__(self):
        return f"{self.message}"