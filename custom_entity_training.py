"""
Custom entity training pipeline for industry-specific entities.
Allows training custom NER models for domain-specific entity recognition.
"""

import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import spacy
from spacy.scorer import Scorer
from spacy.training import Example
from spacy.util import compounding, minibatch

logger = logging.getLogger(__name__)

# Import custom exceptions
from ai_exceptions import (ConfigurationError, DataValidationError,
                           InvalidInputError, MimirAIException,
                           ModelLoadingError, ResourceNotFoundError,
                           TrainingError, validate_text_input)

# Constants
MIN_TRAINING_EXAMPLES = 20
DEFAULT_N_ITER = 100
DEFAULT_DROP_RATE = 0.5
DEFAULT_BATCH_SIZE = 8
MIN_BATCH_SIZE = 4.0
BATCH_COMPOUND_FACTOR = 1.001
LOG_ITERATION_INTERVAL = 20
EVALUATION_SUBSET_SIZE = 10
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
MAX_ENTITY_LENGTH = 100
MIN_ENTITY_LENGTH = 1


class CustomEntityTrainer:
    """Train custom NER models for industry-specific entities."""
    
    def __init__(self, base_model: str = "en_core_web_md", 
                 model_output_dir: str = "models/custom_ner"):
        """
        Initialize the custom entity trainer.
        
        Args:
            base_model: Base spaCy model to start from
            model_output_dir: Directory to save trained models
        """
        self.base_model = base_model
        # Validate and create model directory
        try:
            self.model_output_dir = Path(model_output_dir)
            # Security check: ensure path is not trying to escape
            if '..' in str(self.model_output_dir):
                raise ConfigurationError('model_output_dir', 'Path cannot contain ".."')
            self.model_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ConfigurationError('model_output_dir', f'Failed to create directory: {e}')
        
        # Custom entity types for different industries
        self.industry_entities = {
            'finance': ['FINANCIAL_INSTRUMENT', 'REGULATORY_BODY', 'FINANCIAL_METRIC'],
            'technology': ['TECHNOLOGY', 'PROGRAMMING_LANGUAGE', 'FRAMEWORK', 'PROTOCOL'],
            'healthcare': ['MEDICAL_CONDITION', 'TREATMENT', 'DRUG', 'MEDICAL_DEVICE'],
            'legal': ['LAW', 'COURT', 'LEGAL_ENTITY', 'CASE'],
            'energy': ['ENERGY_SOURCE', 'ENERGY_COMPANY', 'ENERGY_TECHNOLOGY']
        }
        
        # Training data storage
        self.training_data_file = self.model_output_dir / "training_data.json"
        self.load_training_data()
    
    def load_training_data(self) -> None:
        """Load existing training data if available."""
        if self.training_data_file.exists():
            try:
                with open(self.training_data_file, 'r') as f:
                    self.training_data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in training data file: {e}")
                self.training_data = defaultdict(list)
            except Exception as e:
                logger.error(f"Failed to load training data: {e}")
                self.training_data = defaultdict(list)
        else:
            self.training_data = defaultdict(list)
    
    def save_training_data(self) -> None:
        """Save training data to disk."""
        try:
            with open(self.training_data_file, 'w') as f:
                json.dump(dict(self.training_data), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
            raise MimirAIException(f"Failed to save training data", {'error': str(e)})
    
    def add_training_example(self, text: str, entities: List[Tuple[int, int, str]], 
                           industry: str = 'general') -> None:
        """
        Add a training example for custom entities.
        
        Args:
            text: The text containing entities
            entities: List of (start, end, label) tuples
            industry: Industry category for the example
            
        Raises:
            InvalidInputError: If input validation fails
            DataValidationError: If entity validation fails
        """
        # Validate text
        text = validate_text_input(text, min_length=1)
        
        # Validate entities
        for start, end, label in entities:
            if start < 0 or end > len(text):
                raise DataValidationError(
                    'entity_bounds',
                    f'Entity bounds ({start}, {end}) exceed text length ({len(text)})',
                    {'start': start, 'end': end, 'text_length': len(text)}
                )
            if end <= start:
                raise DataValidationError(
                    'entity_bounds',
                    f'Invalid entity bounds: end ({end}) must be greater than start ({start})',
                    {'start': start, 'end': end}
                )
            if not label:
                raise DataValidationError(
                    'entity_label',
                    'Entity label cannot be empty'
                )
        training_example = {
            'text': text,
            'entities': entities,
            'timestamp': datetime.utcnow().isoformat(),
            'industry': industry
        }
        
        self.training_data[industry].append(training_example)
        self.save_training_data()
        
        logger.info(f"Added training example for {industry}: {len(entities)} entities")
    
    def prepare_training_data(self, industry: str) -> List[Tuple[str, Dict]]:
        """
        Prepare training data in spaCy format.
        
        Args:
            industry: Industry to prepare data for
            
        Returns:
            List of (text, annotations) tuples
        """
        if industry not in self.training_data:
            logger.warning(f"No training data for industry: {industry}")
            return []
        
        training_data = []
        for example in self.training_data[industry]:
            text = example['text']
            entities = example['entities']
            
            # Convert to spaCy format
            annotations = {"entities": entities}
            training_data.append((text, annotations))
        
        return training_data
    
    def train_custom_model(self, industry: str, n_iter: int = DEFAULT_N_ITER, 
                          drop_rate: float = DEFAULT_DROP_RATE, batch_size: int = DEFAULT_BATCH_SIZE) -> Optional[str]:
        """
        Train a custom NER model for specific industry.
        
        Args:
            industry: Industry to train model for
            n_iter: Number of training iterations
            drop_rate: Dropout rate for training
            batch_size: Batch size for training
            
        Returns:
            Path to saved model or None if training failed
        """
        training_data = self.prepare_training_data(industry)
        
        if len(training_data) < MIN_TRAINING_EXAMPLES:
            raise TrainingError(
                f"{industry}_ner",
                f"Insufficient training data: {len(training_data)} examples, need at least {MIN_TRAINING_EXAMPLES}"
            )
        
        try:
            # Load base model
            try:
                nlp = spacy.load(self.base_model)
            except Exception as e:
                raise ModelLoadingError(self.base_model, str(e))
            
            # Get or create NER component
            if "ner" not in nlp.pipe_names:
                ner = nlp.create_pipe("ner")
                nlp.add_pipe(ner)
            else:
                ner = nlp.get_pipe("ner")
            
            # Add custom labels
            for label in self.industry_entities.get(industry, []):
                ner.add_label(label)
            
            # Prepare training
            other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
            
            # Train the model
            logger.info(f"Starting training for {industry} with {len(training_data)} examples")
            
            with nlp.disable_pipes(*other_pipes):
                optimizer = nlp.begin_training()
                
                for iteration in range(n_iter):
                    random.shuffle(training_data)
                    losses = {}
                    
                    # Create minibatches
                    batches = minibatch(
                        training_data,
                        size=compounding(MIN_BATCH_SIZE,
                        batch_size,
                        BATCH_COMPOUND_FACTOR)
                    ))
                    
                    for batch in batches:
                        examples = []
                        for text, annotations in batch:
                            doc = nlp.make_doc(text)
                            example = Example.from_dict(doc, annotations)
                            examples.append(example)
                        
                        # Update model
                        nlp.update(examples, sgd=optimizer, drop=drop_rate, losses=losses)
                    
                    if iteration % LOG_ITERATION_INTERVAL == 0:
                        logger.info(f"Iteration {iteration}, Loss: {losses.get('ner', 0):.3f}")
            
            # Save the model
            model_name = f"{industry}_ner_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.model_output_dir / model_name
            nlp.to_disk(model_path)
            
            logger.info(f"Model saved to: {model_path}")
            
            # Evaluate model
            self._evaluate_model(nlp, training_data[:EVALUATION_SUBSET_SIZE])
            
            return str(model_path)
            
        except TrainingError:
            raise
        except ModelLoadingError:
            raise
        except Exception as e:
            raise TrainingError(f"{industry}_ner", str(e))
    
    def _evaluate_model(self, nlp, test_data: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Evaluate the trained model."""
        scorer = Scorer()
        examples = []
        
        for text, annotations in test_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        
        scores = scorer.score(examples)
        
        logger.info("Model Evaluation:")
        logger.info(f"  - Precision: {scores['ents_p']:.3f}")
        logger.info(f"  - Recall: {scores['ents_r']:.3f}")
        logger.info(f"  - F-Score: {scores['ents_f']:.3f}")
    
    def load_custom_model(self, model_path: str) -> Optional[Any]:
        """Load a trained custom model.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Loaded model or None
            
        Raises:
            ResourceNotFoundError: If model path doesn't exist
            ModelLoadingError: If model loading fails
        """
        # Validate path
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise ResourceNotFoundError('model', str(model_path))
        
        try:
            nlp = spacy.load(model_path)
            logger.info(f"Loaded custom model from: {model_path}")
            return nlp
        except Exception as e:
            raise ModelLoadingError(str(model_path), str(e))
    
    def extract_with_custom_model(self, text: str, model_path: str) -> List[Dict]:
        """
        Extract entities using a custom trained model.
        
        Args:
            text: Text to extract entities from
            model_path: Path to custom model
            
        Returns:
            List of extracted entities
        """
        nlp = self.load_custom_model(model_path)
        if not nlp:
            return []
        
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 1.0  # TODO: Implement confidence scoring
            })
        
        return entities
    
    def generate_training_data_from_patterns(self, industry: str, 
                                           patterns_file: Optional[str] = None):
        """
        Generate training data using pattern matching for quick bootstrapping.
        
        Args:
            industry: Industry to generate data for
            patterns_file: Optional file with patterns
        """
        # Default patterns for different industries
        default_patterns = {
            'finance': {
                'FINANCIAL_INSTRUMENT': ['stock', 'bond', 'ETF', 'option', 'futures', 
                                       'derivative', 'commodity', 'forex'],
                'REGULATORY_BODY': ['SEC', 'FINRA', 'CFTC', 'Federal Reserve', 'ECB', 
                                  'Bank of England', 'FSA', 'ESMA'],
                'FINANCIAL_METRIC': ['P/E ratio', 'market cap', 'dividend yield', 'EPS', 
                                   'ROI', 'ROE', 'EBITDA', 'revenue']
            },
            'technology': {
                'TECHNOLOGY': ['AI', 'machine learning', 'blockchain', 'IoT', 'cloud computing',
                             '5G', 'quantum computing', 'edge computing'],
                'PROGRAMMING_LANGUAGE': ['Python', 'Java', 'JavaScript', 'C++', 'Go', 
                                       'Rust', 'Swift', 'Kotlin'],
                'FRAMEWORK': ['React', 'Angular', 'Vue.js', 'Django', 'Flask', 'Spring',
                            'TensorFlow', 'PyTorch']
            },
            'healthcare': {
                'MEDICAL_CONDITION': ['diabetes', 'cancer', 'hypertension', 'COVID-19',
                                    'Alzheimer\'s', 'heart disease', 'stroke'],
                'TREATMENT': ['chemotherapy', 'radiation therapy', 'immunotherapy',
                            'surgery', 'physical therapy', 'dialysis'],
                'DRUG': ['aspirin', 'insulin', 'metformin', 'antibiotics', 'vaccine',
                       'ibuprofen', 'penicillin']
            }
        }
        
        patterns = default_patterns.get(industry, {})
        
        # Generate example sentences
        templates = [
            "The company announced a new {entity} product.",
            "Experts predict {entity} will revolutionize the industry.",
            "The latest {entity} technology shows promising results.",
            "{entity} is being investigated by regulators.",
            "Patients treated with {entity} showed improvement.",
            "The {entity} market grew by 20% last quarter."
        ]
        
        # Create training examples
        for entity_type, examples in patterns.items():
            for entity in examples:
                for template in templates:
                    text = template.format(entity=entity)
                    
                    # Find entity position
                    start = text.find(entity)
                    end = start + len(entity)
                    
                    if start >= 0:
                        self.add_training_example(
                            text,
                            [(start, end, entity_type)],
                            industry
                        )
        
        logger.info(f"Generated training data for {industry}")
    
    def active_learning_loop(self, unlabeled_texts: List[str], 
                           model_path: str, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Active learning loop to identify uncertain predictions for human review.
        
        Args:
            unlabeled_texts: Texts to process
            model_path: Path to current model
            confidence_threshold: Threshold for uncertain predictions
            
        Returns:
            List of uncertain examples for human annotation
        """
        nlp = self.load_custom_model(model_path)
        if not nlp:
            return []
        
        uncertain_examples = []
        
        for text in unlabeled_texts:
            doc = nlp(text)
            
            # For demonstration, we'll consider all entities as uncertain
            # In practice, you'd have a confidence score from the model
            if doc.ents:
                uncertain_examples.append({
                    'text': text,
                    'predicted_entities': [
                        {
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char
                        }
                        for ent in doc.ents
                    ],
                    'needs_review': True
                })
        
        return uncertain_examples
    
    def export_training_data(self, industry: str, output_file: str) -> None:
        """Export training data in various formats.
        
        Args:
            industry: Industry to export data for
            output_file: Output file path
            
        Raises:
            InvalidInputError: If format is not supported
        """
        data = self.training_data.get(industry, [])
        
        try:
            if output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
            elif output_file.endswith('.jsonl'):
                with open(output_file, 'w') as f:
                    for example in data:
                        f.write(json.dumps(example) + '\n')
            else:
                raise InvalidInputError('output_file', 'JSON or JSONL file', output_file)
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            raise
    
    def import_training_data(self, industry: str, input_file: str) -> None:
        """Import training data from external sources.
        
        Args:
            industry: Industry to import data for
            input_file: Input file path
            
        Raises:
            ResourceNotFoundError: If input file doesn't exist
            InvalidInputError: If format is not supported
        """
        # Validate file exists
        if not os.path.exists(input_file):
            raise ResourceNotFoundError('file', input_file)
        
        try:
            if input_file.endswith('.json'):
                with open(input_file, 'r') as f:
                    data = json.load(f)
            elif input_file.endswith('.jsonl'):
                data = []
                with open(input_file, 'r') as f:
                    for line in f:
                        data.append(json.loads(line))
            else:
                raise InvalidInputError('input_file', 'JSON or JSONL file', input_file)
            
            # Add to training data
            for example in data:
                if 'text' in example and 'entities' in example:
                    self.add_training_example(
                        example['text'],
                        example['entities'],
                        industry
                    )
            
            logger.info(f"Imported {len(data)} examples for {industry}")
            
        except Exception as e:
            logger.error(f"Failed to import training data: {e}")
            raise


class IndustryEntityRecognizer:
    """Use custom trained models for industry-specific entity recognition."""
    
    def __init__(self):
        """Initialize the recognizer with available models."""
        self.models = {}
        self.model_dir = Path("models/custom_ner")
        self._load_available_models()
    
    def _load_available_models(self) -> None:
        """Load all available custom models."""
        if not self.model_dir.exists():
            logger.warning("No custom models directory found")
            return
        
        for model_path in self.model_dir.iterdir():
            if model_path.is_dir() and 'meta.json' in [f.name for f in model_path.iterdir()]:
                industry = model_path.name.split('_')[0]
                try:
                    self.models[industry] = spacy.load(str(model_path))
                    logger.info(f"Loaded custom model for {industry}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_path}: {e}")
    
    def extract_entities(self, text: str, industry: str) -> List[Dict[str, Any]]:
        """Extract entities using industry-specific model.
        
        Args:
            text: Text to extract entities from
            industry: Industry model to use
            
        Returns:
            List of extracted entities
            
        Raises:
            ModelNotAvailableError: If no model exists for the industry
        """
        if industry not in self.models:
            raise ModelNotAvailableError(f'custom_{industry}_ner', industry)
        
        nlp = self.models[industry]
        doc = nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'industry': industry
            })
        
        return entities


def create_sample_training_data():
    """Create sample training data for demonstration."""
    trainer = CustomEntityTrainer()
    
    # Finance examples
    finance_examples = [
        ("The SEC announced new regulations for cryptocurrency exchanges.",
         [(4, 7, "REGULATORY_BODY")]),
        ("Apple's P/E ratio increased to 28.5 after strong earnings.",
         [(8, 17, "FINANCIAL_METRIC")]),
        ("Investors are bullish on ETF products linked to renewable energy.",
         [(26, 29, "FINANCIAL_INSTRUMENT")]),
        ("The Federal Reserve raised interest rates by 25 basis points.",
         [(4, 19, "REGULATORY_BODY")]),
        ("Trading in futures contracts reached record volumes.",
         [(11, 18, "FINANCIAL_INSTRUMENT")])
    ]
    
    for text, entities in finance_examples:
        trainer.add_training_example(text, entities, 'finance')
    
    # Technology examples
    tech_examples = [
        ("The company uses TensorFlow for its machine learning models.",
         [(17, 27, "FRAMEWORK"), (36, 52, "TECHNOLOGY")]),
        ("Python remains the most popular programming language for AI.",
         [(0, 6, "PROGRAMMING_LANGUAGE"), (57, 59, "TECHNOLOGY")]),
        ("Quantum computing breakthrough announced by IBM researchers.",
         [(0, 17, "TECHNOLOGY")]),
        ("React and Vue.js dominate the frontend framework market.",
         [(0, 5, "FRAMEWORK"), (10, 16, "FRAMEWORK")])
    ]
    
    for text, entities in tech_examples:
        trainer.add_training_example(text, entities, 'technology')
    
    # Healthcare examples
    healthcare_examples = [
        ("New immunotherapy treatment shows promise for lung cancer patients.",
         [(4, 17, "TREATMENT"), (46, 57, "MEDICAL_CONDITION")]),
        ("Insulin prices continue to rise affecting diabetes patients.",
         [(0, 7, "DRUG"), (42, 50, "MEDICAL_CONDITION")]),
        ("The vaccine proved effective against COVID-19 variants.",
         [(4, 11, "DRUG"), (37, 45, "MEDICAL_CONDITION")]),
        ("Chemotherapy combined with radiation therapy improved outcomes.",
         [(0, 12, "TREATMENT"), (27, 44, "TREATMENT")])
    ]
    
    for text, entities in healthcare_examples:
        trainer.add_training_example(text, entities, 'healthcare')
    
    logger.info("Sample training data created")
    return trainer


if __name__ == "__main__":
    # Test the custom entity trainer
    logging.basicConfig(level=logging.INFO)
    
    # Create sample training data
    trainer = create_sample_training_data()
    
    # Generate more training data from patterns
    trainer.generate_training_data_from_patterns('finance')
    trainer.generate_training_data_from_patterns('technology')
    trainer.generate_training_data_from_patterns('healthcare')
    
    # Train a model for finance (if enough data)
    logger.info("\nTraining custom model for finance industry...")
    model_path = trainer.train_custom_model('finance', n_iter=50)
    
    if model_path:
        # Test the trained model
        test_text = "The SEC is investigating hedge fund trading in ETF securities."
        entities = trainer.extract_with_custom_model(test_text, model_path)
        
        logger.info(f"\nTest extraction: {test_text}")
        for entity in entities:
            logger.info(f"  - {entity['text']} ({entity['label']})")
    
    # Export training data
    trainer.export_training_data('finance', 'finance_training_data.json')