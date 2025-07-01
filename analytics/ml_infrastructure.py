"""ML Infrastructure for Mimir Analytics.

This module provides MLflow integration, model registry, A/B testing framework,
and feature store capabilities for managing machine learning models and experiments.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
import joblib  # Safer alternative to pickle
import hmac

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, r2_score
)

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient
    from mlflow.models.signature import ModelSignature, infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    
from .data_warehouse import AnalyticsDataWarehouse

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model registry for managing ML models with versioning and metadata."""
    
    def __init__(self, registry_path: str = "analytics/model_registry"):
        """Initialize the model registry.
        
        Args:
            registry_path: Path to store registered models
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.registry_path / "registry_metadata.json"
        self.metadata = self._load_metadata()
        
        self.logger = logging.getLogger(__name__)
    
    def _generate_model_hash(self, model_path: Path) -> str:
        """Generate SHA-256 hash for model integrity checking."""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _verify_model_integrity(self, model_path: Path, expected_hash: str) -> bool:
        """Verify model file integrity using hash."""
        actual_hash = self._generate_model_hash(model_path)
        return hmac.compare_digest(expected_hash, actual_hash)
    
    def register_model(self, model: Any, model_name: str, version: str = None,
                      model_type: str = "sklearn", metadata: Dict[str, Any] = None,
                      metrics: Dict[str, float] = None) -> str:
        """Register a new model version.
        
        Args:
            model: The model object to register
            model_name: Name of the model
            version: Version string (auto-generated if not provided)
            model_type: Type of model (sklearn, pytorch, tensorflow, custom)
            metadata: Additional metadata to store
            metrics: Model performance metrics
            
        Returns:
            Model ID
        """
        try:
            # Generate version if not provided
            if not version:
                version = self._generate_version(model_name)
            
            # Create model ID
            model_id = f"{model_name}_v{version}"
            
            # Create model directory
            model_dir = self.models_dir / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Save model based on type (using secure serialization)
            model_path = model_dir / "model.joblib"
            
            if model_type == "sklearn":
                # Use joblib for sklearn models (safer than pickle)
                joblib.dump(model, model_path, compress=3)
                
                # Generate integrity hash
                model_hash = self._generate_model_hash(model_path)
            elif model_type == "pytorch":
                import torch
                torch.save(model.state_dict(), model_path.with_suffix('.pt'))
            elif model_type == "tensorflow":
                model.save(str(model_dir / "tf_model"))
            else:
                # Custom serialization
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save model metadata
            model_metadata = {
                'model_id': model_id,
                'model_name': model_name,
                'version': version,
                'model_type': model_type,
                'registered_at': datetime.utcnow().isoformat(),
                'model_path': str(model_path),
                'metrics': metrics or {},
                'metadata': metadata or {},
                'status': 'registered'
            }
            
            # Save metadata
            with open(model_dir / 'metadata.json', 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            # Update registry
            if model_name not in self.metadata:
                self.metadata[model_name] = {}
            
            self.metadata[model_name][version] = model_metadata
            self._save_metadata()
            
            self.logger.info(f"Registered model: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            raise
    
    def load_model(self, model_name: str, version: str = 'latest') -> Tuple[Any, Dict[str, Any]]:
        """Load a registered model.
        
        Args:
            model_name: Name of the model
            version: Version to load ('latest' for most recent)
            
        Returns:
            Tuple of (model object, metadata)
        """
        try:
            # Get model metadata
            if model_name not in self.metadata:
                raise ValueError(f"Model {model_name} not found in registry")
            
            if version == 'latest':
                # Get latest version
                versions = sorted(self.metadata[model_name].keys(), 
                                key=lambda v: self.metadata[model_name][v]['registered_at'],
                                reverse=True)
                if not versions:
                    raise ValueError(f"No versions found for model {model_name}")
                version = versions[0]
            
            if version not in self.metadata[model_name]:
                raise ValueError(f"Version {version} not found for model {model_name}")
            
            model_metadata = self.metadata[model_name][version]
            model_type = model_metadata['model_type']
            
            # Load model based on type with integrity verification
            model_path = Path(model_metadata['model_path'])
            expected_hash = model_metadata.get('model_hash')
            
            if expected_hash and not self._verify_model_integrity(model_path, expected_hash):
                raise ValueError(f"Model integrity check failed for {model_name} v{version}")
            
            if model_type == "sklearn":
                model = joblib.load(model_path)
            elif model_type == "pytorch":
                import torch
                model = torch.load(model_metadata['model_path'])
            elif model_type == "tensorflow":
                import tensorflow as tf
                model = tf.keras.models.load_model(
                    str(Path(model_metadata['model_path']).parent / "tf_model")
                )
            else:
                # Custom deserialization with joblib
                model = joblib.load(model_path)
            
            self.logger.info(f"Loaded model: {model_name} v{version}")
            return model, model_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def list_models(self, model_name: str = None) -> List[Dict[str, Any]]:
        """List all registered models or versions of a specific model.
        
        Args:
            model_name: Optional model name to filter by
            
        Returns:
            List of model metadata
        """
        models = []
        
        if model_name:
            if model_name in self.metadata:
                for version, metadata in self.metadata[model_name].items():
                    models.append(metadata)
        else:
            for name, versions in self.metadata.items():
                for version, metadata in versions.items():
                    models.append(metadata)
        
        return sorted(models, key=lambda m: m['registered_at'], reverse=True)
    
    def promote_model(self, model_name: str, version: str, stage: str = 'production'):
        """Promote a model version to a specific stage.
        
        Args:
            model_name: Name of the model
            version: Version to promote
            stage: Target stage (staging, production, archived)
        """
        if model_name not in self.metadata or version not in self.metadata[model_name]:
            raise ValueError(f"Model {model_name} v{version} not found")
        
        # Update all versions' stages
        for v in self.metadata[model_name]:
            if self.metadata[model_name][v].get('stage') == stage:
                self.metadata[model_name][v]['stage'] = 'archived'
        
        # Set new stage
        self.metadata[model_name][version]['stage'] = stage
        self.metadata[model_name][version]['promoted_at'] = datetime.utcnow().isoformat()
        
        self._save_metadata()
        self.logger.info(f"Promoted {model_name} v{version} to {stage}")
    
    def _generate_version(self, model_name: str) -> str:
        """Generate a new version number for a model."""
        if model_name not in self.metadata:
            return "1.0.0"
        
        # Get latest version
        versions = list(self.metadata[model_name].keys())
        if not versions:
            return "1.0.0"
        
        # Parse versions and increment
        latest = max(versions, key=lambda v: [int(x) for x in v.split('.')])
        parts = latest.split('.')
        parts[-1] = str(int(parts[-1]) + 1)
        
        return '.'.join(parts)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load registry metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save registry metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)


class ExperimentTracker:
    """MLflow experiment tracker for managing ML experiments."""
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = "mimir_analytics"):
        """Initialize the experiment tracker.
        
        Args:
            tracking_uri: MLflow tracking URI (defaults to local)
            experiment_name: Default experiment name
        """
        if not MLFLOW_AVAILABLE:
            self.logger = logging.getLogger(__name__)
            self.logger.warning("MLflow not available. Experiment tracking disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.tracking_uri = tracking_uri or "file:./mlruns"
        mlflow.set_tracking_uri(self.tracking_uri)
        
        self.experiment_name = experiment_name
        self.experiment = mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
        self.logger = logging.getLogger(__name__)
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags to add to the run
            
        Returns:
            Run ID
        """
        if not self.enabled:
            return str(uuid4())
        
        run = mlflow.start_run(run_name=run_name)
        
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        return run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters for the current run."""
        if not self.enabled:
            return
        
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics for the current run."""
        if not self.enabled:
            return
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model: Any, artifact_path: str, model_type: str = "sklearn",
                  signature: ModelSignature = None, input_example: Any = None):
        """Log a model to MLflow."""
        if not self.enabled:
            return
        
        if model_type == "sklearn":
            mlflow.sklearn.log_model(
                model, artifact_path, signature=signature, 
                input_example=input_example
            )
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(
                model, artifact_path, signature=signature,
                input_example=input_example
            )
        elif model_type == "tensorflow":
            mlflow.tensorflow.log_model(
                model, artifact_path, signature=signature,
                input_example=input_example
            )
        else:
            # Log as generic Python model
            mlflow.pyfunc.log_model(artifact_path, python_model=model)
    
    def end_run(self):
        """End the current MLflow run."""
        if not self.enabled:
            return
        
        mlflow.end_run()
    
    def get_best_run(self, metric_name: str, maximize: bool = True) -> Dict[str, Any]:
        """Get the best run based on a metric.
        
        Args:
            metric_name: Metric to optimize
            maximize: Whether to maximize (True) or minimize (False)
            
        Returns:
            Best run information
        """
        if not self.enabled:
            return {}
        
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"],
            max_results=1
        )
        
        if runs:
            run = runs[0]
            return {
                'run_id': run.info.run_id,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'tags': run.data.tags
            }
        
        return {}


class ABTestingFramework:
    """A/B testing framework for comparing model performance."""
    
    def __init__(self, analytics_warehouse: AnalyticsDataWarehouse):
        """Initialize the A/B testing framework.
        
        Args:
            analytics_warehouse: Analytics data warehouse
        """
        self.warehouse = analytics_warehouse
        self.active_tests = {}
        self.test_results_dir = Path("analytics/ab_tests")
        self.test_results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def create_test(self, test_name: str, model_a: Any, model_b: Any,
                   test_config: Dict[str, Any]) -> str:
        """Create a new A/B test.
        
        Args:
            test_name: Name of the test
            model_a: Control model
            model_b: Treatment model
            test_config: Test configuration including:
                - traffic_split: Percentage of traffic for model B (0-100)
                - duration_hours: Test duration
                - metrics: List of metrics to track
                - minimum_samples: Minimum samples needed
                
        Returns:
            Test ID
        """
        test_id = f"{test_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        test_data = {
            'test_id': test_id,
            'test_name': test_name,
            'created_at': datetime.utcnow().isoformat(),
            'status': 'active',
            'config': test_config,
            'model_a_stats': {'predictions': 0, 'errors': 0},
            'model_b_stats': {'predictions': 0, 'errors': 0},
            'results': []
        }
        
        # Save models
        model_dir = self.test_results_dir / test_id
        model_dir.mkdir(exist_ok=True)
        
        with open(model_dir / 'model_a.pkl', 'wb') as f:
            pickle.dump(model_a, f)
        
        with open(model_dir / 'model_b.pkl', 'wb') as f:
            pickle.dump(model_b, f)
        
        # Save test metadata
        with open(model_dir / 'test_metadata.json', 'w') as f:
            json.dump(test_data, f, indent=2)
        
        self.active_tests[test_id] = {
            'model_a': model_a,
            'model_b': model_b,
            'data': test_data
        }
        
        self.logger.info(f"Created A/B test: {test_id}")
        return test_id
    
    def get_model_for_request(self, test_id: str, request_id: str = None) -> Tuple[Any, str]:
        """Get model to use for a request based on traffic split.
        
        Args:
            test_id: Test ID
            request_id: Optional request ID for consistent assignment
            
        Returns:
            Tuple of (model, variant)
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        traffic_split = test['data']['config']['traffic_split']
        
        # Use request ID for consistent assignment if provided
        if request_id:
            hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
            use_model_b = (hash_value % 100) < traffic_split
        else:
            use_model_b = np.random.random() * 100 < traffic_split
        
        if use_model_b:
            return test['model_b'], 'model_b'
        else:
            return test['model_a'], 'model_a'
    
    def record_prediction(self, test_id: str, variant: str, prediction: Any,
                         actual: Any = None, metadata: Dict[str, Any] = None):
        """Record a prediction made during A/B test.
        
        Args:
            test_id: Test ID
            variant: Model variant used (model_a or model_b)
            prediction: Model prediction
            actual: Actual value (if available)
            metadata: Additional metadata
        """
        if test_id not in self.active_tests:
            return
        
        test = self.active_tests[test_id]
        
        # Update stats
        test['data'][f'{variant}_stats']['predictions'] += 1
        
        # Record result
        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'variant': variant,
            'prediction': prediction,
            'actual': actual,
            'metadata': metadata or {}
        }
        
        test['data']['results'].append(result)
        
        # Save periodically
        if len(test['data']['results']) % 100 == 0:
            self._save_test_results(test_id)
    
    def evaluate_test(self, test_id: str) -> Dict[str, Any]:
        """Evaluate A/B test results.
        
        Args:
            test_id: Test ID
            
        Returns:
            Test evaluation results
        """
        if test_id not in self.active_tests:
            # Load from disk
            test_dir = self.test_results_dir / test_id
            if not test_dir.exists():
                raise ValueError(f"Test {test_id} not found")
            
            with open(test_dir / 'test_metadata.json', 'r') as f:
                test_data = json.load(f)
        else:
            test_data = self.active_tests[test_id]['data']
        
        results = test_data['results']
        
        if not results:
            return {'error': 'No results to evaluate'}
        
        # Separate results by variant
        model_a_results = [r for r in results if r['variant'] == 'model_a']
        model_b_results = [r for r in results if r['variant'] == 'model_b']
        
        # Calculate metrics for each model
        evaluation = {
            'test_id': test_id,
            'test_name': test_data['test_name'],
            'total_predictions': len(results),
            'model_a': {
                'predictions': len(model_a_results),
                'percentage': len(model_a_results) / len(results) * 100 if results else 0
            },
            'model_b': {
                'predictions': len(model_b_results),
                'percentage': len(model_b_results) / len(results) * 100 if results else 0
            }
        }
        
        # Calculate performance metrics if actual values are available
        model_a_with_actual = [r for r in model_a_results if r['actual'] is not None]
        model_b_with_actual = [r for r in model_b_results if r['actual'] is not None]
        
        if model_a_with_actual and model_b_with_actual:
            # Regression metrics
            try:
                a_predictions = [float(r['prediction']) for r in model_a_with_actual]
                a_actuals = [float(r['actual']) for r in model_a_with_actual]
                b_predictions = [float(r['prediction']) for r in model_b_with_actual]
                b_actuals = [float(r['actual']) for r in model_b_with_actual]
                
                evaluation['model_a']['metrics'] = {
                    'mae': mean_absolute_error(a_actuals, a_predictions),
                    'mse': mean_squared_error(a_actuals, a_predictions),
                    'r2': r2_score(a_actuals, a_predictions)
                }
                
                evaluation['model_b']['metrics'] = {
                    'mae': mean_absolute_error(b_actuals, b_predictions),
                    'mse': mean_squared_error(b_actuals, b_predictions),
                    'r2': r2_score(b_actuals, b_predictions)
                }
                
                # Statistical significance test
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(
                    np.array(a_predictions) - np.array(a_actuals),
                    np.array(b_predictions) - np.array(b_actuals)
                )
                
                evaluation['statistical_test'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
                
            except (ValueError, TypeError):
                # Try classification metrics
                try:
                    a_predictions = [r['prediction'] for r in model_a_with_actual]
                    a_actuals = [r['actual'] for r in model_a_with_actual]
                    b_predictions = [r['prediction'] for r in model_b_with_actual]
                    b_actuals = [r['actual'] for r in model_b_with_actual]
                    
                    evaluation['model_a']['metrics'] = {
                        'accuracy': accuracy_score(a_actuals, a_predictions),
                        'precision': precision_score(a_actuals, a_predictions, average='weighted'),
                        'recall': recall_score(a_actuals, a_predictions, average='weighted'),
                        'f1': f1_score(a_actuals, a_predictions, average='weighted')
                    }
                    
                    evaluation['model_b']['metrics'] = {
                        'accuracy': accuracy_score(b_actuals, b_predictions),
                        'precision': precision_score(b_actuals, b_predictions, average='weighted'),
                        'recall': recall_score(b_actuals, b_predictions, average='weighted'),
                        'f1': f1_score(b_actuals, b_predictions, average='weighted')
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Could not calculate metrics: {e}")
        
        # Determine winner
        if 'metrics' in evaluation['model_a'] and 'metrics' in evaluation['model_b']:
            # Compare primary metric (first one)
            primary_metric = list(evaluation['model_a']['metrics'].keys())[0]
            
            if primary_metric in ['mae', 'mse']:  # Lower is better
                winner = 'model_a' if evaluation['model_a']['metrics'][primary_metric] < \
                        evaluation['model_b']['metrics'][primary_metric] else 'model_b'
            else:  # Higher is better
                winner = 'model_a' if evaluation['model_a']['metrics'][primary_metric] > \
                        evaluation['model_b']['metrics'][primary_metric] else 'model_b'
            
            evaluation['winner'] = winner
            evaluation['improvement'] = abs(
                evaluation['model_b']['metrics'][primary_metric] - 
                evaluation['model_a']['metrics'][primary_metric]
            )
        
        return evaluation
    
    def end_test(self, test_id: str) -> Dict[str, Any]:
        """End an A/B test and return final results.
        
        Args:
            test_id: Test ID
            
        Returns:
            Final test results
        """
        if test_id not in self.active_tests:
            return {'error': 'Test not found'}
        
        # Get final evaluation
        evaluation = self.evaluate_test(test_id)
        
        # Update test status
        test = self.active_tests[test_id]
        test['data']['status'] = 'completed'
        test['data']['completed_at'] = datetime.utcnow().isoformat()
        test['data']['final_evaluation'] = evaluation
        
        # Save final results
        self._save_test_results(test_id)
        
        # Remove from active tests
        del self.active_tests[test_id]
        
        self.logger.info(f"Ended A/B test: {test_id}")
        return evaluation
    
    def _save_test_results(self, test_id: str):
        """Save test results to disk."""
        test = self.active_tests[test_id]
        test_dir = self.test_results_dir / test_id
        
        with open(test_dir / 'test_metadata.json', 'w') as f:
            json.dump(test['data'], f, indent=2)


class FeatureStore:
    """Feature store for managing ML features with versioning and lineage."""
    
    def __init__(self, analytics_warehouse: AnalyticsDataWarehouse,
                 storage_path: str = "analytics/feature_store"):
        """Initialize the feature store.
        
        Args:
            analytics_warehouse: Analytics data warehouse
            storage_path: Path to store feature data
        """
        self.warehouse = analytics_warehouse
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.features_dir = self.storage_path / "features"
        self.features_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.storage_path / "feature_metadata.json"
        self.metadata = self._load_metadata()
        
        self.logger = logging.getLogger(__name__)
    
    def register_feature(self, feature_name: str, feature_type: str,
                        description: str = None, schema: Dict[str, str] = None,
                        tags: List[str] = None) -> str:
        """Register a new feature definition.
        
        Args:
            feature_name: Name of the feature
            feature_type: Type (numeric, categorical, embedding, etc.)
            description: Feature description
            schema: Feature schema definition
            tags: Optional tags for categorization
            
        Returns:
            Feature ID
        """
        feature_id = f"feature_{feature_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        feature_metadata = {
            'feature_id': feature_id,
            'feature_name': feature_name,
            'feature_type': feature_type,
            'description': description,
            'schema': schema or {},
            'tags': tags or [],
            'created_at': datetime.utcnow().isoformat(),
            'versions': []
        }
        
        if feature_name not in self.metadata:
            self.metadata[feature_name] = []
        
        self.metadata[feature_name].append(feature_metadata)
        self._save_metadata()
        
        self.logger.info(f"Registered feature: {feature_id}")
        return feature_id
    
    def compute_features(self, feature_name: str, start_time: datetime,
                        end_time: datetime, computation_function: callable,
                        **kwargs) -> pd.DataFrame:
        """Compute features for a given time range.
        
        Args:
            feature_name: Name of the feature
            start_time: Start time for computation
            end_time: End time for computation
            computation_function: Function to compute features
            **kwargs: Additional arguments for computation
            
        Returns:
            Computed features DataFrame
        """
        try:
            self.logger.info(f"Computing features: {feature_name}")
            
            # Get data from warehouse
            with self.warehouse.get_connection() as conn:
                # Example: Get article metrics for feature computation
                query = """
                    SELECT * FROM analytics.article_metrics
                    WHERE time >= %s AND time <= %s
                    ORDER BY time
                """
                df = pd.read_sql_query(query, conn, params=[start_time, end_time])
            
            # Compute features
            features_df = computation_function(df, **kwargs)
            
            # Add metadata columns
            features_df['feature_name'] = feature_name
            features_df['computed_at'] = datetime.utcnow()
            
            # Store features
            version = self._save_features(feature_name, features_df)
            
            self.logger.info(f"Computed {len(features_df)} feature records")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Failed to compute features: {e}")
            raise
    
    def get_features(self, feature_names: List[str], entity_ids: List[str] = None,
                    start_time: datetime = None, end_time: datetime = None,
                    version: str = 'latest') -> pd.DataFrame:
        """Retrieve features from the store.
        
        Args:
            feature_names: List of feature names to retrieve
            entity_ids: Optional list of entity IDs to filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            version: Feature version to retrieve
            
        Returns:
            Features DataFrame
        """
        all_features = []
        
        for feature_name in feature_names:
            try:
                # Load feature data
                feature_df = self._load_features(feature_name, version)
                
                # Apply filters
                if entity_ids:
                    feature_df = feature_df[feature_df['entity_id'].isin(entity_ids)]
                
                if start_time:
                    feature_df = feature_df[feature_df['timestamp'] >= start_time]
                
                if end_time:
                    feature_df = feature_df[feature_df['timestamp'] <= end_time]
                
                all_features.append(feature_df)
                
            except Exception as e:
                self.logger.warning(f"Failed to load feature {feature_name}: {e}")
        
        if all_features:
            # Merge all features
            result = all_features[0]
            for df in all_features[1:]:
                result = pd.merge(result, df, on=['entity_id', 'timestamp'], how='outer')
            
            return result
        
        return pd.DataFrame()
    
    def create_training_dataset(self, feature_names: List[str], 
                               label_column: str = None,
                               start_time: datetime = None,
                               end_time: datetime = None,
                               test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create a training dataset from features.
        
        Args:
            feature_names: List of features to include
            label_column: Name of the label column
            start_time: Start time for data
            end_time: End time for data
            test_size: Fraction of data for test set
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Get features
        features_df = self.get_features(
            feature_names, 
            start_time=start_time,
            end_time=end_time
        )
        
        if features_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        
        if label_column and label_column in features_df.columns:
            X = features_df.drop(columns=[label_column])
            y = features_df[label_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
        else:
            train_df, test_df = train_test_split(
                features_df, test_size=test_size, random_state=42
            )
        
        return train_df, test_df
    
    def _save_features(self, feature_name: str, features_df: pd.DataFrame) -> str:
        """Save features to storage."""
        version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        feature_dir = self.features_dir / feature_name
        feature_dir.mkdir(exist_ok=True)
        
        # Save as parquet for efficiency
        file_path = feature_dir / f"v{version}.parquet"
        features_df.to_parquet(file_path, index=False)
        
        # Update metadata
        for feature_meta in self.metadata.get(feature_name, []):
            if feature_meta['feature_name'] == feature_name:
                feature_meta['versions'].append({
                    'version': version,
                    'created_at': datetime.utcnow().isoformat(),
                    'file_path': str(file_path),
                    'record_count': len(features_df),
                    'columns': list(features_df.columns)
                })
                break
        
        self._save_metadata()
        return version
    
    def _load_features(self, feature_name: str, version: str = 'latest') -> pd.DataFrame:
        """Load features from storage."""
        if feature_name not in self.metadata:
            raise ValueError(f"Feature {feature_name} not found")
        
        feature_meta = self.metadata[feature_name][0]  # Get first definition
        
        if not feature_meta['versions']:
            raise ValueError(f"No versions found for feature {feature_name}")
        
        if version == 'latest':
            version_info = feature_meta['versions'][-1]
        else:
            version_info = next(
                (v for v in feature_meta['versions'] if v['version'] == version),
                None
            )
            
        if not version_info:
            raise ValueError(f"Version {version} not found for feature {feature_name}")
        
        return pd.read_parquet(version_info['file_path'])
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load feature metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save feature metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)


class MLInfrastructureManager:
    """Main manager for ML infrastructure components."""
    
    def __init__(self, analytics_warehouse: AnalyticsDataWarehouse):
        """Initialize the ML infrastructure manager.
        
        Args:
            analytics_warehouse: Analytics data warehouse
        """
        self.warehouse = analytics_warehouse
        
        # Initialize components
        self.model_registry = ModelRegistry()
        self.experiment_tracker = ExperimentTracker()
        self.ab_testing = ABTestingFramework(analytics_warehouse)
        self.feature_store = FeatureStore(analytics_warehouse)
        
        self.logger = logging.getLogger(__name__)
    
    def train_and_register_model(self, model_class: type, model_name: str,
                                training_data: pd.DataFrame, 
                                target_column: str,
                                model_params: Dict[str, Any] = None,
                                feature_columns: List[str] = None) -> str:
        """Train a model and register it with tracking.
        
        Args:
            model_class: Model class to instantiate
            model_name: Name for the model
            training_data: Training DataFrame
            target_column: Target column name
            model_params: Model hyperparameters
            feature_columns: Feature columns to use
            
        Returns:
            Model ID
        """
        try:
            # Start MLflow run
            run_id = self.experiment_tracker.start_run(
                run_name=f"{model_name}_training",
                tags={'model_name': model_name}
            )
            
            # Log parameters
            self.experiment_tracker.log_params(model_params or {})
            
            # Prepare data
            if feature_columns:
                X = training_data[feature_columns]
            else:
                X = training_data.drop(columns=[target_column])
            
            y = training_data[target_column]
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = model_class(**(model_params or {}))
            model.fit(X_train, y_train)
            
            # Evaluate model
            predictions = model.predict(X_test)
            
            # Calculate metrics based on problem type
            try:
                # Regression metrics
                metrics = {
                    'mae': mean_absolute_error(y_test, predictions),
                    'mse': mean_squared_error(y_test, predictions),
                    'r2': r2_score(y_test, predictions)
                }
            except:
                # Classification metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, predictions),
                    'precision': precision_score(y_test, predictions, average='weighted'),
                    'recall': recall_score(y_test, predictions, average='weighted'),
                    'f1': f1_score(y_test, predictions, average='weighted')
                }
            
            # Log metrics
            self.experiment_tracker.log_metrics(metrics)
            
            # Log model to MLflow
            signature = infer_signature(X_train, predictions)
            self.experiment_tracker.log_model(
                model, 
                artifact_path=model_name,
                signature=signature,
                input_example=X_train.iloc[:5]
            )
            
            # Register model
            model_id = self.model_registry.register_model(
                model=model,
                model_name=model_name,
                model_type='sklearn',
                metadata={
                    'mlflow_run_id': run_id,
                    'feature_columns': list(X.columns),
                    'target_column': target_column,
                    'training_samples': len(X_train)
                },
                metrics=metrics
            )
            
            # End MLflow run
            self.experiment_tracker.end_run()
            
            self.logger.info(f"Trained and registered model: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Failed to train and register model: {e}")
            self.experiment_tracker.end_run()
            raise
    
    def run_model_experiment(self, experiment_name: str,
                           model_configs: List[Dict[str, Any]],
                           training_data: pd.DataFrame,
                           target_column: str) -> Dict[str, Any]:
        """Run experiments with multiple model configurations.
        
        Args:
            experiment_name: Name of the experiment
            model_configs: List of model configurations to test
            training_data: Training data
            target_column: Target column
            
        Returns:
            Experiment results
        """
        self.logger.info(f"Running experiment: {experiment_name}")
        
        experiment_results = {
            'experiment_name': experiment_name,
            'started_at': datetime.utcnow().isoformat(),
            'model_results': []
        }
        
        best_score = None
        best_model_id = None
        
        for config in model_configs:
            try:
                model_id = self.train_and_register_model(
                    model_class=config['model_class'],
                    model_name=config['model_name'],
                    training_data=training_data,
                    target_column=target_column,
                    model_params=config.get('params', {}),
                    feature_columns=config.get('features')
                )
                
                # Get model metrics
                _, metadata = self.model_registry.load_model(
                    config['model_name'], 
                    version='latest'
                )
                
                metrics = metadata.get('metrics', {})
                primary_metric = config.get('optimize_metric', 'r2')
                score = metrics.get(primary_metric, 0)
                
                experiment_results['model_results'].append({
                    'model_id': model_id,
                    'model_name': config['model_name'],
                    'metrics': metrics,
                    'score': score
                })
                
                # Track best model
                if best_score is None or score > best_score:
                    best_score = score
                    best_model_id = model_id
                    
            except Exception as e:
                self.logger.error(f"Failed to train {config['model_name']}: {e}")
                experiment_results['model_results'].append({
                    'model_name': config['model_name'],
                    'error': str(e)
                })
        
        experiment_results['completed_at'] = datetime.utcnow().isoformat()
        experiment_results['best_model_id'] = best_model_id
        experiment_results['best_score'] = best_score
        
        return experiment_results