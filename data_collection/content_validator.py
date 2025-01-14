import logging
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ResolutionValidationResult:
    file_name: str
    is_valid: bool
    validation_errors: List[str]

def validate_resolution_content(content: str) -> List[str]:
    """
    Validate the content of a resolution file.
    
    Args:
        content: The text content of the resolution file
        
    Returns:
        List of validation error messages, empty if content is valid
    """
    validation_errors = []
    
    if len(content) < 100:
        validation_errors.append("Resolution content too short (minimum 100 characters)")
    if not content.strip():
        validation_errors.append("Resolution file is empty or contains only whitespace")
        
    return validation_errors

def validate_resolution_content(resolutions_dir: Path) -> Dict[str, ResolutionValidationResult]:
    logger = logging.getLogger(__name__)
    logger.info("Starting resolution content validation...")
    
    validation_results = {}
    
    for resolution_file in resolutions_dir.glob("*.txt"):
        try:
            content = resolution_file.read_text(encoding="utf-8")
            validation_errors = validate_resolution_content(content)
            
            validation_results[resolution_file.name] = ResolutionValidationResult(
                file_name=resolution_file.name,
                is_valid=len(validation_errors) == 0,
                validation_errors=validation_errors
            )
            
        except Exception as e:
            validation_results[resolution_file.name] = ResolutionValidationResult(
                file_name=resolution_file.name,
                is_valid=False,
                validation_errors=[f"Error reading resolution file: {str(e)}"]
            )
    
    return validation_results 