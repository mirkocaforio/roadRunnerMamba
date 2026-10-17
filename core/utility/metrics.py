import os
import glob
import re
from typing import Dict, List, Tuple, Any

import numpy as np
from numpy import floating
from scipy import stats



class MetricsAnalyzer:
    """
    A reusable and modular class for analyzing metrics from log files.
    Supports computing statistical measures like mean and confidence intervals.
    """
    
    def __init__(self, confidence_level: float = 0.95, metric_pattern: str | None = None):
        """
        Initialize the MetricsAnalyzer.
        
        Args:
            confidence_level: Confidence level for interval computation (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.metric_pattern = re.compile(metric_pattern) if metric_pattern \
            else re.compile(r'\s*-\s*([\w_]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
    
    def parse_metrics_from_log(self, log_path: str) -> Dict[str, float]:
        """
        Extract metrics from the last lines of a .log file.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            Dictionary mapping metric_name -> value (float)
        """
        metrics = {}
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = self.metric_pattern.match(line)
                    if match:
                        name, val = match.group(1), float(match.group(2))
                        if name.startswith('test_'):
                            name = name[5:]  # remove 'test_' prefix
                            metrics[name] = val
        except (FileNotFoundError, IOError) as e:
            print(f"Warning: Could not read {log_path}: {e}")
        return metrics
    
    def compute_mean_ci(self, data: List[float]) -> tuple[float, float, float] | tuple[floating, floating, floating] | \
                                                    tuple[floating, float | Any, float | Any]:
        """
        Calculate mean and confidence interval at the specified level
        using Student's t-distribution.
        
        Args:
            data: List of numerical values
            
        Returns:
            Tuple of (mean, ci_lower, ci_upper)
        """
        if not data:
            return 0.0, 0.0, 0.0
            
        array = np.array(data)
        n = len(array)
        mean = np.mean(array)
        
        if n < 2:
            # Cannot calculate CI with a single sample
            return mean, mean, mean
            
        sem = stats.sem(array)  # standard error of the mean
        df = n - 1
        t_crit = stats.t.ppf((1 + self.confidence_level) / 2., df)
        margin = t_crit * sem
        return mean, mean - margin, mean + margin
    
    def process_subfolder(self, subfolder: str, 
                         log_pattern: str = 'training_*.log') -> Dict[str, Tuple[float, float, float, int]]:
        """
        Process all log files matching the pattern in the subfolder,
        collect values per metric and calculate mean+CI.
        
        Args:
            subfolder: Path to the subfolder containing log files
            log_pattern: Glob pattern for log files (default: 'training_*.log')
            
        Returns:
            Dictionary mapping metric -> (mean, ci_lower, ci_upper, sample_count)
        """
        log_files = glob.glob(os.path.join(subfolder, log_pattern))
        all_metrics = {}
        
        for log_file in log_files:
            metrics = self.parse_metrics_from_log(log_file)
            for name, val in metrics.items():
                all_metrics.setdefault(name, []).append(val)
        
        results = {}
        for name, values in all_metrics.items():
            mean, lo, hi = self.compute_mean_ci(values)
            results[name] = (mean, lo, hi, len(values))
        
        return results
        
    def analyze_parent_folder(self, parent_folder: str, 
                            log_pattern: str = 'training_*.log',
                            print_results: bool = True,
                            max_depth: int = 1,
                            current_depth: int = 0) -> Dict[str, Dict[str, Tuple[float, float, float, int]]]:
        """
        Analyze all subfolders in the parent folder recursively.
        
        Args:
            parent_folder: Path to parent folder containing subfolders with log files
            log_pattern: Glob pattern for log files (default: 'training_*.log')
            print_results: Whether to print results to console (default: True)
            max_depth: Maximum recursion depth (None for unlimited)
            current_depth: Current recursion depth (internal parameter)
            
        Returns:
            Dictionary mapping relative_path -> metrics_results
        """
        all_results = {}
        
        # Check if we've reached max depth
        if max_depth is not None and current_depth >= max_depth:
            return all_results
        
        try:
            # Iterate over all entries in the current directory
            for entry in sorted(os.listdir(parent_folder)):
                entry_path = os.path.join(parent_folder, entry)
                
                if os.path.isdir(entry_path):
                    # Process current subfolder for log files
                    results = self.process_subfolder(entry_path, log_pattern)
                    
                    # If current folder has results, add them
                    if results:
                        # Create relative path for better organization
                        if current_depth == 0:
                            folder_key = entry
                        else:
                            # Get relative path from original parent folder
                            rel_path = os.path.relpath(entry_path, parent_folder if current_depth == 0 else 
                                                    os.path.dirname(parent_folder))
                            folder_key = rel_path.replace(os.sep, '/')  # Normalize path separators
                        
                        all_results[folder_key] = results
                        
                        if print_results:
                            self._print_results(folder_key, results)
                    
                    # Recursively process subdirectories
                    nested_results = self.analyze_parent_folder(
                        entry_path, 
                        log_pattern=log_pattern,
                        print_results=print_results,
                        max_depth=max_depth,
                        current_depth=current_depth + 1
                    )
                    
                    # Merge nested results with current results
                    for nested_key, nested_values in nested_results.items():
                        # Create hierarchical key
                        if current_depth == 0:
                            hierarchical_key = f"{entry}/{nested_key}"
                        else:
                            hierarchical_key = nested_key
                        
                        all_results[hierarchical_key] = nested_values
                        
        except (OSError, PermissionError) as e:
            if print_results:
                print(f"Warning: Could not access {parent_folder}: {e}")
        
        return all_results

    def _print_results(self, folder_name: str, 
                      results: Dict[str, Tuple[float, float, float, int]]) -> None:
        """
        Print results in a formatted way.
        
        Args:
            folder_name: Name of the folder
            results: Results dictionary from process_subfolder
        """
        print(f"\n=== {folder_name} ===")
        for metric, (mean, lo, hi, n) in results.items():
            deviation = (hi - lo) / 2 * 100
            ci_percent = int(self.confidence_level * 100)
            print(f"{metric:20s}  n={n:2d}  mean={mean:.6f}  CI{ci_percent}%=({lo:.6f}, {hi:.6f})  ±{deviation:.2f}%")
    
    def set_confidence_level(self, confidence_level: float) -> None:
        """
        Update the confidence level for interval computation.
        
        Args:
            confidence_level: New confidence level (between 0 and 1)
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        self.confidence_level = confidence_level


# Example usage:
# # Utilizzo base
# analyzer = MetricsAnalyzer(confidence_level=0.95)
# results = analyzer.analyze_parent_folder('/path/to/experiments')

# # Utilizzo avanzato
# analyzer = MetricsAnalyzer(confidence_level=0.99)
# results = analyzer.analyze_parent_folder(
#     '/path/to/experiments', 
#     log_pattern='*.log', 
#     print_results=False
# )

# # Processare una singola cartella
# single_results = analyzer.process_subfolder('/path/to/single/experiment')
