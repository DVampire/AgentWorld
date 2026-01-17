#!/usr/bin/env python3
"""
Extract incorrect samples from experiment results JSON file.
"""

import json
import sys
from pathlib import Path
from src.logger import logger
from typing import List, Dict, Any
from src.benchmark import benchmark_manager

def extract_incorrect_samples_file(input_file: str, output_file: str = None) -> None:
    """
    Extract samples where 'correct' field is False from the experiment results.

    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output JSON file (optional)
    """
    # Load the experiment results
    print(f"Loading results from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract experiment metadata
    experiment_meta = data.get('experiment_meta', {})
    all_results = data.get('results', [])

    # Filter incorrect samples
    incorrect_samples = [sample for sample in all_results if not sample.get('correct', True)]

    # Calculate statistics
    total_samples = len(all_results)
    incorrect_count = len(incorrect_samples)
    correct_count = total_samples - incorrect_count
    accuracy = correct_count / total_samples * 100 if total_samples > 0 else 0

    print("\nStatistics:")
    print(f"   Total samples: {total_samples}")
    print(f"   Correct samples: {correct_count}")
    print(f"   Incorrect samples: {incorrect_count}")
    print(f"   Accuracy: {accuracy:.1f}%")
    # Create output data structure
    output_data = {
        "experiment_meta": experiment_meta,
        "extraction_info": {
            "total_samples": total_samples,
            "correct_samples": correct_count,
            "incorrect_samples": incorrect_count,
            "accuracy": accuracy,

        },
        "incorrect_samples": incorrect_samples
    }

    # Save to output file
    print(f"\nSaving {incorrect_count} incorrect samples to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("Extraction completed!")



def main():
    """Main function to handle command line arguments."""
    input_file = r'D:\86134\Documents\GitHub\AgentWorld\examples\workdir\results\reflection_gpqa_2.json'
    output_file = r'D:\86134\Documents\GitHub\AgentWorld\examples\workdir\results\reflection_gpqa_2_incorrect_samples.json'

    try:
        extract_incorrect_samples_file(input_file, output_file)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    main()
