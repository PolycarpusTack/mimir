"""
Sentence Transformer Setup and Benchmarking
EPIC 3: Semantic Search & Similarity Engine

This script downloads, sets up, and benchmarks sentence transformer models
for optimal embedding generation performance.

Author: Claude Code
"""

import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceTransformerSetup:
    """
    Setup and benchmark sentence transformer models for semantic search.
    """

    def __init__(self, cache_dir: str = "./models"):
        """
        Initialize the setup manager.

        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Available models for benchmarking
        self.available_models = {
            # Multi-language models
            "all-MiniLM-L6-v2": {
                "description": "Multilingual, 384-dim, fast and good quality",
                "dimensions": 384,
                "languages": ["en", "de", "fr", "es", "it", "pt", "ru", "zh"],
                "speed_rating": 5,
                "quality_rating": 4,
                "recommended": True,
            },
            "all-MiniLM-L12-v2": {
                "description": "Multilingual, 384-dim, slower but better quality",
                "dimensions": 384,
                "languages": ["en", "de", "fr", "es", "it", "pt", "ru", "zh"],
                "speed_rating": 3,
                "quality_rating": 5,
                "recommended": True,
            },
            "all-mpnet-base-v2": {
                "description": "English only, 768-dim, high quality",
                "dimensions": 768,
                "languages": ["en"],
                "speed_rating": 3,
                "quality_rating": 5,
                "recommended": True,
            },
            "paraphrase-multilingual-MiniLM-L12-v2": {
                "description": "50+ languages, 384-dim, good for paraphrases",
                "dimensions": 384,
                "languages": ["multi"],
                "speed_rating": 3,
                "quality_rating": 4,
                "recommended": False,
            },
            "distiluse-base-multilingual-cased": {
                "description": "15+ languages, 512-dim, good general purpose",
                "dimensions": 512,
                "languages": ["multi"],
                "speed_rating": 4,
                "quality_rating": 4,
                "recommended": False,
            },
        }

        # Test sentences for benchmarking
        self.test_sentences = [
            "Artificial intelligence revolutionizes modern business operations.",
            "Tesla reports record quarterly earnings despite supply chain challenges.",
            "Climate change impacts global agricultural productivity significantly.",
            "Cryptocurrency markets experience volatile trading patterns today.",
            "Space exploration missions advance scientific understanding rapidly.",
            "Remote work policies reshape corporate culture permanently.",
            "Renewable energy adoption accelerates across emerging markets.",
            "Social media platforms face increased regulatory scrutiny.",
            "Healthcare technology improves patient outcomes worldwide.",
            "Economic uncertainty affects consumer spending behavior.",
        ]

        # Benchmark results
        self.benchmark_results = {}

    def download_model(self, model_name: str, force_redownload: bool = False) -> bool:
        """
        Download and cache a sentence transformer model.

        Args:
            model_name: Name of the model to download
            force_redownload: Whether to force redownload even if cached

        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.available_models:
            logger.error(f"Unknown model: {model_name}")
            return False

        try:
            logger.info(f"Downloading model: {model_name}")

            # Set cache directory
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(self.cache_dir)

            # Download model
            model = SentenceTransformer(model_name, cache_folder=str(self.cache_dir))

            # Verify model works
            test_embedding = model.encode("Test sentence")
            expected_dim = self.available_models[model_name]["dimensions"]

            if len(test_embedding) != expected_dim:
                logger.warning(f"Model dimension mismatch: expected {expected_dim}, got {len(test_embedding)}")

            logger.info(f"Successfully downloaded and verified model: {model_name}")
            logger.info(f"Model dimension: {len(test_embedding)}")

            return True

        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            return False

    def benchmark_model(
        self, model_name: str, num_iterations: int = 3, batch_sizes: List[int] = [1, 8, 16, 32]
    ) -> Dict:
        """
        Benchmark a sentence transformer model for speed and memory usage.

        Args:
            model_name: Name of the model to benchmark
            num_iterations: Number of benchmark iterations
            batch_sizes: List of batch sizes to test

        Returns:
            Dictionary with benchmark results
        """
        if model_name not in self.available_models:
            logger.error(f"Unknown model: {model_name}")
            return {}

        logger.info(f"Benchmarking model: {model_name}")

        try:
            # Load model
            model = SentenceTransformer(model_name, cache_folder=str(self.cache_dir))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            results = {
                "model_name": model_name,
                "device": device,
                "model_info": self.available_models[model_name],
                "batch_benchmarks": {},
                "memory_usage": {},
                "quality_metrics": {},
            }

            # Benchmark different batch sizes
            for batch_size in batch_sizes:
                logger.info(f"Benchmarking batch size: {batch_size}")

                # Prepare test data
                test_data = (self.test_sentences * (batch_size // len(self.test_sentences) + 1))[:batch_size]

                # Warm up
                _ = model.encode(test_data[: min(4, len(test_data))])

                # Benchmark encoding speed
                times = []
                memory_usage = []

                for iteration in range(num_iterations):
                    # Measure memory before
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    else:
                        memory_before = psutil.Process().memory_info().rss

                    # Time encoding
                    start_time = time.time()
                    embeddings = model.encode(test_data, batch_size=batch_size, show_progress_bar=False)
                    end_time = time.time()

                    # Measure memory after
                    if device == "cuda":
                        memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                        memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB
                    else:
                        memory_after = psutil.Process().memory_info().rss
                        memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB

                    times.append(end_time - start_time)
                    memory_usage.append(memory_used)

                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)
                avg_memory = np.mean(memory_usage)
                sentences_per_second = batch_size / avg_time

                results["batch_benchmarks"][batch_size] = {
                    "avg_time_seconds": avg_time,
                    "std_time_seconds": std_time,
                    "sentences_per_second": sentences_per_second,
                    "avg_memory_mb": avg_memory,
                    "embedding_dimension": len(embeddings[0]),
                }

                logger.info(f"Batch {batch_size}: {sentences_per_second:.1f} sentences/sec, {avg_memory:.1f} MB")

            # Calculate quality metrics using cosine similarity
            logger.info("Calculating quality metrics...")
            quality_embeddings = model.encode(self.test_sentences)

            # Calculate average inter-sentence similarity
            similarities = []
            for i, _ in enumerate(quality_embeddings):
                for j in range(i + 1, len(quality_embeddings)):
                    sim = util.pytorch_cos_sim(quality_embeddings[i], quality_embeddings[j]).item()
                    similarities.append(sim)

            results["quality_metrics"] = {
                "avg_inter_similarity": np.mean(similarities),
                "std_inter_similarity": np.std(similarities),
                "min_similarity": np.min(similarities),
                "max_similarity": np.max(similarities),
            }

            # Store results
            self.benchmark_results[model_name] = results

            logger.info(f"Benchmark completed for {model_name}")
            return results

        except Exception as e:
            logger.error(f"Failed to benchmark model {model_name}: {e}")
            return {}

    def benchmark_all_recommended(self) -> Dict:
        """
        Benchmark all recommended models.

        Returns:
            Dictionary with all benchmark results
        """
        logger.info("Benchmarking all recommended models...")

        all_results = {}

        for model_name, model_info in self.available_models.items():
            if model_info.get("recommended", False):
                logger.info(f"\n{'='*50}")
                logger.info(f"Benchmarking: {model_name}")
                logger.info(f"Description: {model_info['description']}")
                logger.info(f"{'='*50}")

                # Download model if needed
                if not self.download_model(model_name):
                    logger.error(f"Failed to download {model_name}, skipping benchmark")
                    continue

                # Benchmark model
                results = self.benchmark_model(model_name)
                if results:
                    all_results[model_name] = results

                # Clean up memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return all_results

    def generate_recommendation(self) -> Dict:
        """
        Generate model recommendations based on benchmark results.

        Returns:
            Dictionary with recommendations
        """
        if not self.benchmark_results:
            logger.warning("No benchmark results available")
            return {}

        recommendations = {
            "fastest_model": None,
            "best_quality": None,
            "balanced": None,
            "memory_efficient": None,
            "multilingual": None,
        }

        fastest_speed = 0
        best_quality_score = 0
        best_balanced_score = 0
        lowest_memory = float("inf")

        for model_name, results in self.benchmark_results.items():
            model_info = self.available_models[model_name]

            # Get performance metrics (use batch size 16 as reference)
            batch_16 = results.get("batch_benchmarks", {}).get(16, {})
            if not batch_16:
                continue

            speed = batch_16.get("sentences_per_second", 0)
            memory = batch_16.get("avg_memory_mb", float("inf"))
            quality = results.get("quality_metrics", {}).get("avg_inter_similarity", 0)

            # Fastest model
            if speed > fastest_speed:
                fastest_speed = speed
                recommendations["fastest_model"] = {
                    "model": model_name,
                    "speed": speed,
                    "description": model_info["description"],
                }

            # Best quality (lower inter-similarity often means better distinction)
            quality_score = (
                model_info["quality_rating"] * 2 - quality
            )  # Prefer models with good ratings and lower inter-similarity
            if quality_score > best_quality_score:
                best_quality_score = quality_score
                recommendations["best_quality"] = {
                    "model": model_name,
                    "quality_score": quality_score,
                    "description": model_info["description"],
                }

            # Balanced (speed * quality)
            balanced_score = speed * model_info["quality_rating"]
            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
                recommendations["balanced"] = {
                    "model": model_name,
                    "balanced_score": balanced_score,
                    "description": model_info["description"],
                }

            # Memory efficient
            if memory < lowest_memory:
                lowest_memory = memory
                recommendations["memory_efficient"] = {
                    "model": model_name,
                    "memory_mb": memory,
                    "description": model_info["description"],
                }

            # Multilingual
            if "multi" in model_info["languages"] or len(model_info["languages"]) > 5:
                if not recommendations["multilingual"]:
                    recommendations["multilingual"] = {
                        "model": model_name,
                        "languages": model_info["languages"],
                        "description": model_info["description"],
                    }

        return recommendations

    def save_results(self, filename: str = "sentence_transformer_benchmark.json"):
        """Save benchmark results to file."""
        output_file = Path(filename)

        output_data = {
            "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "python_version": torch.__version__,
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            },
            "available_models": self.available_models,
            "benchmark_results": self.benchmark_results,
            "recommendations": self.generate_recommendation(),
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"Benchmark results saved to {output_file}")

    def print_summary(self):
        """Print a summary of benchmark results."""
        if not self.benchmark_results:
            print("No benchmark results available.")
            return

        print("\n" + "=" * 80)
        print("SENTENCE TRANSFORMER BENCHMARK SUMMARY")
        print("=" * 80)

        # Print model comparison table
        print(f"\n{'Model':<35} {'Speed (sent/s)':<15} {'Memory (MB)':<12} {'Dimension':<10}")
        print("-" * 75)

        for model_name, results in self.benchmark_results.items():
            batch_16 = results.get("batch_benchmarks", {}).get(16, {})
            if batch_16:
                speed = batch_16.get("sentences_per_second", 0)
                memory = batch_16.get("avg_memory_mb", 0)
                dim = batch_16.get("embedding_dimension", 0)
                print(f"{model_name:<35} {speed:<15.1f} {memory:<12.1f} {dim:<10}")

        # Print recommendations
        print("\n" + "=" * 50)
        print("RECOMMENDATIONS")
        print("=" * 50)

        recommendations = self.generate_recommendation()
        for category, rec in recommendations.items():
            if rec:
                print(f"\n{category.replace('_', ' ').title()}: {rec['model']}")
                print(f"  {rec['description']}")


def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Sentence Transformer Setup and Benchmarking")
    parser.add_argument("command", choices=["download", "benchmark", "all", "recommend"])
    parser.add_argument("--model", help="Specific model to download/benchmark")
    parser.add_argument("--cache-dir", default="./models", help="Directory to cache models")
    parser.add_argument("--iterations", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument("--save", help="Save results to file")

    args = parser.parse_args()

    # Initialize setup
    setup = SentenceTransformerSetup(cache_dir=args.cache_dir)

    if args.command == "download":
        if not args.model:
            print("Error: --model required for download command")
            return

        success = setup.download_model(args.model)
        print(f"Download {'successful' if success else 'failed'} for {args.model}")

    elif args.command == "benchmark":
        if not args.model:
            print("Error: --model required for benchmark command")
            return

        # Download model first if needed
        setup.download_model(args.model)

        # Run benchmark
        results = setup.benchmark_model(args.model, num_iterations=args.iterations)
        if results:
            print(f"Benchmark completed for {args.model}")
            setup.print_summary()

    elif args.command == "all":
        # Benchmark all recommended models
        results = setup.benchmark_all_recommended()
        setup.print_summary()

        if args.save:
            setup.save_results(args.save)

    elif args.command == "recommend":
        # Load existing results or run benchmark
        if not setup.benchmark_results:
            print("No existing results found. Running benchmark...")
            setup.benchmark_all_recommended()

        recommendations = setup.generate_recommendation()

        print("\n" + "=" * 50)
        print("MODEL RECOMMENDATIONS")
        print("=" * 50)

        for category, rec in recommendations.items():
            if rec:
                print(f"\n{category.replace('_', ' ').title()}: {rec['model']}")
                print(f"  {rec['description']}")


if __name__ == "__main__":
    main()
