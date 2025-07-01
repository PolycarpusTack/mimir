"""Code Complexity Analyzer for Mimir Analytics.

This module provides tools for analyzing code complexity, identifying technical debt,
and generating actionable reports for code improvement.
"""

import ast
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from radon.complexity import cc_visit
from radon.metrics import mi_visit, h_visit
from radon.raw import analyze

from .exceptions import DataValidationException, AnalyticsBaseException
from .utils import format_bytes, performance_timer


logger = logging.getLogger(__name__)


class CodeComplexityAnalyzer:
    """Analyzer for measuring and tracking code complexity metrics."""
    
    def __init__(self, project_root: Path = None):
        """Initialize the complexity analyzer.
        
        Args:
            project_root: Root directory of the project to analyze
        """
        self.project_root = project_root or Path(".")
        self.logger = logging.getLogger(__name__)
        
        # Complexity thresholds
        self.thresholds = {
            'cyclomatic_complexity': {
                'low': 5,
                'medium': 10,
                'high': 15,
                'very_high': 20
            },
            'maintainability_index': {
                'excellent': 85,
                'good': 70,
                'fair': 50,
                'poor': 25
            },
            'lines_of_code': {
                'small': 50,
                'medium': 100,
                'large': 200,
                'very_large': 500
            },
            'halstead_difficulty': {
                'low': 10,
                'medium': 20,
                'high': 30,
                'very_high': 40
            }
        }
    
    @performance_timer
    def analyze_project(self, include_patterns: List[str] = None,
                       exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """Analyze complexity for entire project.
        
        Args:
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            
        Returns:
            Comprehensive project complexity analysis
        """
        if include_patterns is None:
            include_patterns = ['*.py']
        
        if exclude_patterns is None:
            exclude_patterns = [
                '**/test_*.py',
                '**/tests/**',
                '**/__pycache__/**',
                '**/venv/**',
                '**/env/**'
            ]
        
        self.logger.info("Starting project complexity analysis")
        
        # Find Python files
        python_files = self._find_python_files(include_patterns, exclude_patterns)
        
        if not python_files:
            raise DataValidationException("No Python files found for analysis")
        
        self.logger.info(f"Analyzing {len(python_files)} Python files")
        
        # Analyze each file
        file_analyses = {}
        project_stats = {
            'total_files': len(python_files),
            'total_loc': 0,
            'total_sloc': 0,
            'total_functions': 0,
            'total_classes': 0,
            'complexity_distribution': defaultdict(int),
            'maintainability_distribution': defaultdict(int)
        }
        
        for file_path in python_files:
            try:
                analysis = self.analyze_file(file_path)
                file_analyses[str(file_path)] = analysis
                
                # Update project stats
                project_stats['total_loc'] += analysis['raw_metrics']['loc']
                project_stats['total_sloc'] += analysis['raw_metrics']['sloc']
                project_stats['total_functions'] += len(analysis['functions'])
                project_stats['total_classes'] += len(analysis['classes'])
                
                # Update distributions
                for func in analysis['functions']:
                    complexity_level = self._categorize_complexity(
                        func['cyclomatic_complexity']
                    )
                    project_stats['complexity_distribution'][complexity_level] += 1
                
                mi_level = self._categorize_maintainability(
                    analysis['maintainability_index']
                )
                project_stats['maintainability_distribution'][mi_level] += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path}: {e}")
                file_analyses[str(file_path)] = {'error': str(e)}
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(file_analyses, project_stats)
        
        # Identify hotspots
        hotspots = self._identify_complexity_hotspots(file_analyses)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_metrics, hotspots, project_stats
        )
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'project_root': str(self.project_root),
            'file_count': len(python_files),
            'overall_metrics': overall_metrics,
            'project_stats': dict(project_stats),
            'file_analyses': file_analyses,
            'complexity_hotspots': hotspots,
            'recommendations': recommendations
        }
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze complexity metrics for a single file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            File complexity analysis
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise DataValidationException(f"Failed to read file {file_path}: {e}")
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {
                'error': f'Syntax error: {e}',
                'raw_metrics': {},
                'functions': [],
                'classes': [],
                'maintainability_index': 0,
                'halstead_metrics': {}
            }
        
        # Raw metrics (lines of code, etc.)
        raw_metrics = analyze(content)
        
        # Cyclomatic complexity
        complexity_results = cc_visit(content)
        
        # Maintainability index
        mi_results = mi_visit(content, multi=True)
        
        # Halstead metrics
        halstead_results = h_visit(content)
        
        # Extract function and class information
        functions = self._extract_function_info(complexity_results)
        classes = self._extract_class_info(complexity_results)
        
        # Calculate file-level maintainability
        maintainability_index = mi_results.mi if hasattr(mi_results, 'mi') else 0
        
        # Process Halstead metrics
        halstead_metrics = self._process_halstead_metrics(halstead_results)
        
        return {
            'file_path': str(file_path),
            'raw_metrics': {
                'loc': raw_metrics.loc,
                'lloc': raw_metrics.lloc,
                'sloc': raw_metrics.sloc,
                'comments': raw_metrics.comments,
                'multi': raw_metrics.multi,
                'blank': raw_metrics.blank
            },
            'functions': functions,
            'classes': classes,
            'maintainability_index': maintainability_index,
            'halstead_metrics': halstead_metrics,
            'complexity_score': self._calculate_file_complexity_score(
                functions, classes, maintainability_index
            )
        }
    
    def analyze_function(self, source_code: str, function_name: str = None) -> Dict[str, Any]:
        """Analyze complexity of a specific function.
        
        Args:
            source_code: Source code containing the function
            function_name: Name of function to analyze (analyzes all if None)
            
        Returns:
            Function complexity analysis
        """
        try:
            complexity_results = cc_visit(source_code)
            functions = self._extract_function_info(complexity_results)
            
            if function_name:
                # Find specific function
                target_func = next(
                    (f for f in functions if f['name'] == function_name),
                    None
                )
                if not target_func:
                    raise DataValidationException(f"Function '{function_name}' not found")
                return target_func
            else:
                return {'functions': functions}
                
        except Exception as e:
            raise AnalyticsBaseException(f"Function analysis failed: {e}")
    
    def track_complexity_trends(self, history_file: Path = None) -> Dict[str, Any]:
        """Track complexity trends over time.
        
        Args:
            history_file: Path to complexity history file
            
        Returns:
            Complexity trend analysis
        """
        if history_file is None:
            history_file = self.project_root / "complexity_history.json"
        
        # Perform current analysis
        current_analysis = self.analyze_project()
        
        # Load historical data
        history = self._load_complexity_history(history_file)
        
        # Add current analysis to history
        history.append({
            'timestamp': current_analysis['timestamp'],
            'overall_metrics': current_analysis['overall_metrics'],
            'project_stats': current_analysis['project_stats']
        })
        
        # Save updated history
        self._save_complexity_history(history, history_file)
        
        # Calculate trends
        trends = self._calculate_complexity_trends(history)
        
        return {
            'current_analysis': current_analysis,
            'trends': trends,
            'history_count': len(history)
        }
    
    def generate_complexity_report(self, output_format: str = 'markdown',
                                 output_file: Path = None) -> str:
        """Generate a comprehensive complexity report.
        
        Args:
            output_format: Report format ('markdown', 'html', 'json')
            output_file: Output file path (optional)
            
        Returns:
            Report content or file path
        """
        analysis = self.analyze_project()
        
        if output_format == 'markdown':
            report_content = self._generate_markdown_report(analysis)
        elif output_format == 'html':
            report_content = self._generate_html_report(analysis)
        elif output_format == 'json':
            import json
            report_content = json.dumps(analysis, indent=2)
        else:
            raise DataValidationException(f"Unsupported output format: {output_format}")
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            return str(output_file)
        else:
            return report_content
    
    def identify_refactoring_candidates(self, threshold_multiplier: float = 1.5) -> List[Dict[str, Any]]:
        """Identify functions and classes that need refactoring.
        
        Args:
            threshold_multiplier: Multiplier for complexity thresholds
            
        Returns:
            List of refactoring candidates with priorities
        """
        analysis = self.analyze_project()
        candidates = []
        
        for file_path, file_analysis in analysis['file_analyses'].items():
            if 'error' in file_analysis:
                continue
            
            # Check functions
            for func in file_analysis['functions']:
                complexity = func['cyclomatic_complexity']
                threshold = self.thresholds['cyclomatic_complexity']['medium'] * threshold_multiplier
                
                if complexity > threshold:
                    priority = self._calculate_refactoring_priority(
                        complexity, func['lines_of_code']
                    )
                    
                    candidates.append({
                        'type': 'function',
                        'name': func['name'],
                        'file_path': file_path,
                        'cyclomatic_complexity': complexity,
                        'lines_of_code': func['lines_of_code'],
                        'priority': priority,
                        'reason': f'High cyclomatic complexity ({complexity})'
                    })
            
            # Check classes
            for cls in file_analysis['classes']:
                complexity = cls['cyclomatic_complexity']
                threshold = self.thresholds['cyclomatic_complexity']['high'] * threshold_multiplier
                
                if complexity > threshold:
                    priority = self._calculate_refactoring_priority(
                        complexity, cls['lines_of_code']
                    )
                    
                    candidates.append({
                        'type': 'class',
                        'name': cls['name'],
                        'file_path': file_path,
                        'cyclomatic_complexity': complexity,
                        'lines_of_code': cls['lines_of_code'],
                        'priority': priority,
                        'reason': f'High class complexity ({complexity})'
                    })
        
        # Sort by priority (highest first)
        candidates.sort(key=lambda x: x['priority'], reverse=True)
        
        return candidates
    
    # Helper methods
    
    def _find_python_files(self, include_patterns: List[str],
                          exclude_patterns: List[str]) -> List[Path]:
        """Find Python files matching patterns."""
        files = []
        
        for pattern in include_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    # Check if file should be excluded
                    should_exclude = False
                    for exclude_pattern in exclude_patterns:
                        if file_path.match(exclude_pattern):
                            should_exclude = True
                            break
                    
                    if not should_exclude:
                        files.append(file_path)
        
        return files
    
    def _extract_function_info(self, complexity_results) -> List[Dict[str, Any]]:
        """Extract function information from complexity analysis."""
        functions = []
        
        for result in complexity_results:
            if result.type == 'function':
                functions.append({
                    'name': result.name,
                    'lineno': result.lineno,
                    'cyclomatic_complexity': result.complexity,
                    'lines_of_code': result.endline - result.lineno + 1,
                    'complexity_level': self._categorize_complexity(result.complexity)
                })
        
        return functions
    
    def _extract_class_info(self, complexity_results) -> List[Dict[str, Any]]:
        """Extract class information from complexity analysis."""
        classes = []
        
        for result in complexity_results:
            if result.type == 'class':
                classes.append({
                    'name': result.name,
                    'lineno': result.lineno,
                    'cyclomatic_complexity': result.complexity,
                    'lines_of_code': result.endline - result.lineno + 1,
                    'complexity_level': self._categorize_complexity(result.complexity)
                })
        
        return classes
    
    def _process_halstead_metrics(self, halstead_results) -> Dict[str, float]:
        """Process Halstead complexity metrics."""
        if not halstead_results:
            return {}
        
        try:
            return {
                'vocabulary': halstead_results.vocabulary,
                'length': halstead_results.length,
                'calculated_length': halstead_results.calculated_length,
                'volume': halstead_results.volume,
                'difficulty': halstead_results.difficulty,
                'effort': halstead_results.effort,
                'time': halstead_results.time,
                'bugs': halstead_results.bugs
            }
        except AttributeError:
            return {}
    
    def _categorize_complexity(self, complexity: int) -> str:
        """Categorize complexity level."""
        thresholds = self.thresholds['cyclomatic_complexity']
        
        if complexity <= thresholds['low']:
            return 'low'
        elif complexity <= thresholds['medium']:
            return 'medium'
        elif complexity <= thresholds['high']:
            return 'high'
        else:
            return 'very_high'
    
    def _categorize_maintainability(self, mi_score: float) -> str:
        """Categorize maintainability level."""
        thresholds = self.thresholds['maintainability_index']
        
        if mi_score >= thresholds['excellent']:
            return 'excellent'
        elif mi_score >= thresholds['good']:
            return 'good'
        elif mi_score >= thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_file_complexity_score(self, functions: List[Dict],
                                       classes: List[Dict],
                                       maintainability_index: float) -> float:
        """Calculate overall complexity score for a file."""
        if not functions and not classes:
            return 0.0
        
        # Average cyclomatic complexity
        all_items = functions + classes
        avg_complexity = sum(item['cyclomatic_complexity'] for item in all_items) / len(all_items)
        
        # Normalize maintainability index (0-100 to 0-1, inverted)
        mi_factor = max(0, (100 - maintainability_index) / 100)
        
        # Combine factors
        complexity_score = (avg_complexity / 10) * 0.7 + mi_factor * 0.3
        
        return min(complexity_score, 10.0)  # Cap at 10
    
    def _calculate_overall_metrics(self, file_analyses: Dict,
                                 project_stats: Dict) -> Dict[str, Any]:
        """Calculate project-wide complexity metrics."""
        valid_analyses = [
            analysis for analysis in file_analyses.values()
            if 'error' not in analysis
        ]
        
        if not valid_analyses:
            return {}
        
        # Average metrics
        avg_maintainability = sum(
            analysis['maintainability_index']
            for analysis in valid_analyses
        ) / len(valid_analyses)
        
        avg_complexity_score = sum(
            analysis['complexity_score']
            for analysis in valid_analyses
        ) / len(valid_analyses)
        
        # Technical debt estimation
        technical_debt_score = self._calculate_technical_debt_score(
            project_stats, avg_complexity_score, avg_maintainability
        )
        
        return {
            'average_maintainability_index': avg_maintainability,
            'average_complexity_score': avg_complexity_score,
            'technical_debt_score': technical_debt_score,
            'complexity_distribution': dict(project_stats['complexity_distribution']),
            'maintainability_distribution': dict(project_stats['maintainability_distribution'])
        }
    
    def _calculate_technical_debt_score(self, project_stats: Dict,
                                      avg_complexity: float,
                                      avg_maintainability: float) -> float:
        """Calculate technical debt score (0-10, lower is better)."""
        # Complexity factor (0-1)
        complexity_factor = min(avg_complexity / 5.0, 1.0)
        
        # Maintainability factor (0-1, inverted)
        maintainability_factor = max(0, (100 - avg_maintainability) / 100)
        
        # Size factor (larger projects have more debt potential)
        size_factor = min(project_stats['total_loc'] / 10000, 1.0)
        
        # Distribution factor (many high-complexity functions = more debt)
        high_complexity_ratio = (
            project_stats['complexity_distribution']['high'] +
            project_stats['complexity_distribution']['very_high']
        ) / max(project_stats['total_functions'], 1)
        
        # Combine factors
        debt_score = (
            complexity_factor * 0.3 +
            maintainability_factor * 0.3 +
            size_factor * 0.2 +
            high_complexity_ratio * 0.2
        ) * 10
        
        return min(debt_score, 10.0)
    
    def _identify_complexity_hotspots(self, file_analyses: Dict) -> List[Dict[str, Any]]:
        """Identify the most complex functions and classes."""
        hotspots = []
        
        for file_path, analysis in file_analyses.items():
            if 'error' in analysis:
                continue
            
            # Add high-complexity functions
            for func in analysis['functions']:
                if func['complexity_level'] in ['high', 'very_high']:
                    hotspots.append({
                        'type': 'function',
                        'name': func['name'],
                        'file_path': file_path,
                        'cyclomatic_complexity': func['cyclomatic_complexity'],
                        'lines_of_code': func['lines_of_code'],
                        'severity': func['complexity_level']
                    })
            
            # Add high-complexity classes
            for cls in analysis['classes']:
                if cls['complexity_level'] in ['high', 'very_high']:
                    hotspots.append({
                        'type': 'class',
                        'name': cls['name'],
                        'file_path': file_path,
                        'cyclomatic_complexity': cls['cyclomatic_complexity'],
                        'lines_of_code': cls['lines_of_code'],
                        'severity': cls['complexity_level']
                    })
        
        # Sort by complexity (highest first)
        hotspots.sort(key=lambda x: x['cyclomatic_complexity'], reverse=True)
        
        return hotspots[:20]  # Return top 20 hotspots
    
    def _generate_recommendations(self, overall_metrics: Dict,
                                hotspots: List[Dict], project_stats: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Technical debt recommendations
        debt_score = overall_metrics.get('technical_debt_score', 0)
        if debt_score > 7:
            recommendations.append(
                "HIGH PRIORITY: Technical debt score is very high. "
                "Focus on refactoring the most complex functions first."
            )
        elif debt_score > 5:
            recommendations.append(
                "MEDIUM PRIORITY: Technical debt is accumulating. "
                "Consider scheduled refactoring sessions."
            )
        
        # Complexity recommendations
        high_complexity_count = len([h for h in hotspots if h['severity'] == 'very_high'])
        if high_complexity_count > 5:
            recommendations.append(
                f"Found {high_complexity_count} functions/classes with very high complexity. "
                "Break these down into smaller, focused units."
            )
        
        # Maintainability recommendations
        avg_maintainability = overall_metrics.get('average_maintainability_index', 100)
        if avg_maintainability < 50:
            recommendations.append(
                "Low maintainability index detected. Add documentation, "
                "improve naming, and reduce complexity."
            )
        
        # Size recommendations
        if project_stats['total_loc'] > 50000:
            recommendations.append(
                "Large codebase detected. Consider modularization and "
                "architecture review to improve maintainability."
            )
        
        # Function size recommendations
        large_functions = [
            h for h in hotspots
            if h['type'] == 'function' and h['lines_of_code'] > 100
        ]
        if large_functions:
            recommendations.append(
                f"Found {len(large_functions)} functions over 100 lines. "
                "Consider breaking these into smaller functions."
            )
        
        if not recommendations:
            recommendations.append("Code complexity is within acceptable limits. Keep up the good work!")
        
        return recommendations
    
    def _calculate_refactoring_priority(self, complexity: int, loc: int) -> float:
        """Calculate refactoring priority score."""
        complexity_score = min(complexity / 20, 1.0)  # Normalize to 0-1
        size_score = min(loc / 200, 1.0)  # Normalize to 0-1
        
        # Priority is weighted average
        priority = complexity_score * 0.7 + size_score * 0.3
        
        return priority * 100  # Scale to 0-100
    
    def _load_complexity_history(self, history_file: Path) -> List[Dict]:
        """Load complexity history from file."""
        if not history_file.exists():
            return []
        
        try:
            import json
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load complexity history: {e}")
            return []
    
    def _save_complexity_history(self, history: List[Dict], history_file: Path):
        """Save complexity history to file."""
        try:
            import json
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save complexity history: {e}")
    
    def _calculate_complexity_trends(self, history: List[Dict]) -> Dict[str, Any]:
        """Calculate complexity trends from history."""
        if len(history) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Get latest and previous measurements
        latest = history[-1]
        previous = history[-2]
        
        # Calculate changes
        debt_change = (
            latest['overall_metrics']['technical_debt_score'] -
            previous['overall_metrics']['technical_debt_score']
        )
        
        maintainability_change = (
            latest['overall_metrics']['average_maintainability_index'] -
            previous['overall_metrics']['average_maintainability_index']
        )
        
        loc_change = (
            latest['project_stats']['total_loc'] -
            previous['project_stats']['total_loc']
        )
        
        return {
            'technical_debt_trend': 'improving' if debt_change < 0 else 'degrading',
            'technical_debt_change': debt_change,
            'maintainability_trend': 'improving' if maintainability_change > 0 else 'degrading',
            'maintainability_change': maintainability_change,
            'size_change': loc_change,
            'analysis_count': len(history)
        }
    
    def _generate_markdown_report(self, analysis: Dict) -> str:
        """Generate a markdown complexity report."""
        report = []
        
        # Header
        report.append("# Code Complexity Analysis Report")
        report.append(f"**Generated:** {analysis['timestamp']}")
        report.append(f"**Project:** {analysis['project_root']}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        overall = analysis['overall_metrics']
        debt_score = overall.get('technical_debt_score', 0)
        
        report.append(f"- **Technical Debt Score:** {debt_score:.1f}/10")
        report.append(f"- **Average Maintainability:** {overall.get('average_maintainability_index', 0):.1f}")
        report.append(f"- **Files Analyzed:** {analysis['file_count']}")
        report.append(f"- **Total Lines of Code:** {analysis['project_stats']['total_loc']:,}")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        for rec in analysis['recommendations']:
            report.append(f"- {rec}")
        report.append("")
        
        # Complexity Hotspots
        report.append("## Complexity Hotspots")
        hotspots = analysis['complexity_hotspots'][:10]  # Top 10
        
        if hotspots:
            report.append("| Type | Name | File | Complexity | Lines |")
            report.append("|------|------|------|------------|-------|")
            
            for hotspot in hotspots:
                report.append(
                    f"| {hotspot['type']} | {hotspot['name']} | "
                    f"{Path(hotspot['file_path']).name} | "
                    f"{hotspot['cyclomatic_complexity']} | "
                    f"{hotspot['lines_of_code']} |"
                )
        else:
            report.append("No significant complexity hotspots found.")
        
        report.append("")
        
        # Complexity Distribution
        report.append("## Complexity Distribution")
        dist = overall.get('complexity_distribution', {})
        total_funcs = sum(dist.values())
        
        if total_funcs > 0:
            for level in ['low', 'medium', 'high', 'very_high']:
                count = dist.get(level, 0)
                percentage = (count / total_funcs) * 100
                report.append(f"- **{level.title()}:** {count} ({percentage:.1f}%)")
        
        return "\n".join(report)
    
    def _generate_html_report(self, analysis: Dict) -> str:
        """Generate an HTML complexity report."""
        # Simplified HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Complexity Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; }}
                .high {{ color: red; }}
                .medium {{ color: orange; }}
                .low {{ color: green; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            </style>
        </head>
        <body>
            <h1>Code Complexity Analysis Report</h1>
            <p><strong>Generated:</strong> {analysis['timestamp']}</p>
            <p><strong>Project:</strong> {analysis['project_root']}</p>
            
            <h2>Executive Summary</h2>
            <div class="metric">
                <strong>Technical Debt Score:</strong> 
                {analysis['overall_metrics'].get('technical_debt_score', 0):.1f}/10
            </div>
            
            <h2>Recommendations</h2>
            <ul>
        """
        
        for rec in analysis['recommendations']:
            html += f"<li>{rec}</li>"
        
        html += """
            </ul>
            
            <h2>Top Complexity Hotspots</h2>
            <table>
                <tr>
                    <th>Type</th>
                    <th>Name</th>
                    <th>File</th>
                    <th>Complexity</th>
                    <th>Lines</th>
                </tr>
        """
        
        for hotspot in analysis['complexity_hotspots'][:10]:
            html += f"""
                <tr>
                    <td>{hotspot['type']}</td>
                    <td>{hotspot['name']}</td>
                    <td>{Path(hotspot['file_path']).name}</td>
                    <td class="{hotspot['severity']}">{hotspot['cyclomatic_complexity']}</td>
                    <td>{hotspot['lines_of_code']}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html


# CLI interface for complexity analysis
def main():
    """Command-line interface for complexity analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze code complexity")
    parser.add_argument("--project-root", type=Path, default=".",
                       help="Project root directory")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--format", choices=['markdown', 'html', 'json'],
                       default='markdown', help="Output format")
    parser.add_argument("--hotspots-only", action='store_true',
                       help="Show only complexity hotspots")
    
    args = parser.parse_args()
    
    analyzer = CodeComplexityAnalyzer(args.project_root)
    
    if args.hotspots_only:
        candidates = analyzer.identify_refactoring_candidates()
        print("Refactoring Candidates:")
        print("=" * 50)
        for candidate in candidates[:10]:  # Top 10
            print(f"{candidate['type'].title()}: {candidate['name']}")
            print(f"  File: {candidate['file_path']}")
            print(f"  Complexity: {candidate['cyclomatic_complexity']}")
            print(f"  Priority: {candidate['priority']:.1f}")
            print(f"  Reason: {candidate['reason']}")
            print()
    else:
        report = analyzer.generate_complexity_report(
            output_format=args.format,
            output_file=args.output
        )
        
        if args.output:
            print(f"Report saved to: {args.output}")
        else:
            print(report)


if __name__ == "__main__":
    main()