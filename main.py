import logging
from pathlib import Path
from data_collection.resolution_collector import collect_central_bank_resolutions
from data_collection.content_validator import validate_resolution_content

def configure_logging() -> None:
    """Configure logging settings for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/central_bank_collector.log'),
            logging.StreamHandler()
        ]
    )

def initialize_directories() -> tuple[Path, Path]:
    """Create and return necessary directory paths for data storage."""
    data_dir = Path("data")
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    return raw_data_dir, processed_data_dir

def main() -> None:
    """Main execution function for the Central Bank resolution collection process."""
    configure_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initiating Central Bank resolutions collection process...")
        
        raw_data_dir, processed_data_dir = initialize_directories()
        
        # Collect and process resolutions
        collect_central_bank_resolutions(raw_data_dir)
        validation_results = validate_resolution_content(processed_data_dir)
        
        # Process validation results
        for file_result in validation_results.values():
            if not file_result.is_valid:
                logger.warning(
                    f"Content validation failed for {file_result.file_name}: "
                    f"{', '.join(file_result.errors)}"
                )
        
        logger.info("Resolution collection process completed successfully")
        
    except Exception as e:
        logger.error("Critical error in resolution collection process", exc_info=True)
        raise

if __name__ == "__main__":
    main()
