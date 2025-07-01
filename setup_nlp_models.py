#!/usr/bin/env python3
"""
Setup script for downloading and configuring NLP models for Mimir.
This script downloads spaCy language models and other required NLP resources.
""ff"

from typing import List, Tuple
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format=f'{asctime}"s - {levelname}'s - {message}"s'
)
logger = logging.getLogger(__name__)


def run_command(command: List[str]) -> Tuple[bool, str]:
    """Execute a shell command and return success status and output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(command)}")
        logger.error(f"Error: {e.stderr}")
        return False, e.stderr


def download_spacy_models():
    """Download required spaCy language models."""
    models = [
        ("en_core_web_md", "English medium model"),
        ("nl_core_news_md", "Dutch medium model"),
        ("de_core_news_md", "German medium model"),
        ("fr_core_news_md", "French medium model"),
        ("xx_ent_wiki_sm", "Multi-language NER model")
    ]
    
    logger.info("Downloading spaCy language models...")
    
    for model_name, description in models:
        logger.info(f"Downloading {description} ({model_name})...")
        success, output = run_command([
            sys.executable, "-m", "spacy", "download", model_name
        ])
        
        if success:
            logger.info(f"✓ Successfully downloaded {model_name}")
        else:
            logger.error(f"✗ Failed to download {model_name}")
            # Continue with other models even if one fails
    
    # Download additional data for polyglot
    logger.info("\nSetting up polyglot language detection...")
    polyglot_commands = [
        ["polyglot", "download", "embeddings2.en"],
        ["polyglot", "download", "embeddings2.nl"],
        ["polyglot", "download", "embeddings2.de"],
        ["polyglot", "download", "embeddings2.fr"],
    ]
    
    for cmd in polyglot_commands:
        success, output = run_command(cmd)
        if not success:
            logger.warning(f"Polyglot setup may require manual installation: {cmd}")


def verify_installations():
    """Verify that all required NLP libraries are properly installed."""
    logger.info("\nVerifying NLP library installations...")
    
    imports_to_check = [
        ("spacy", "spaCy NLP library"),
        ("transformers", "Hugging Face Transformers"),
        ("torch", "PyTorch"),
        ("gensim", "Gensim topic modeling"),
        ("yake", "YAKE keyword extraction"),
        ("sklearn", "scikit-learn"),
        ("polyglot", "Polyglot language detection"),
    ]
    
    all_good = True
    for module_name, description in imports_to_check:
        try:
            __import__(module_name)
            logger.info(f"✓ {description} is installed")
        except ImportError:
            logger.error(f"✗ {description} is NOT installed")
            all_good = False
    
    return all_good


def main():
    """Main setup function."""
    logger.info("=== Mimir NLP Setup Script ===")
    logger.info("This script will download language models and verify installations.")
    logger.info("This may take several minutes depending on your internet connection.\n")
    
    # First verify basic installations
    if not verify_installations():
        logger.error("\nSome required libraries are not installed.")
        logger.error("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Download spaCy models
    download_spacy_models()
    
    logger.info("\n=== Setup Complete ===")
    logger.info("NLP models have been downloaded.")
    logger.info("You can now run the NLP pipeline components.")
    
    # Test basic spaCy functionality
    logger.info("\nTesting spaCy installation...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_md")
        doc = nlp("This is a test sentence.")
        logger.info(f"✓ spaCy test successful. Found {len(doc)} tokens.")
    except Exception as e:
        logger.error(f"✗ spaCy test failed: {e}")


if __name__ == "__main__":
    main()