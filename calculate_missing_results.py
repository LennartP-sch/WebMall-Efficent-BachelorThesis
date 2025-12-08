r"""
Simple script to calculate results for specific study runs.

This script processes the following study directories:
1. C:\Users\lenna\agentlab_results\2025-11-28_22-28-05_genericagent-gpt-4-1-AP(SAX-M)-2025-04-14-on-webmall-short-basic-ax
2. C:\Users\lenna\agentlab_results\2025-11-28_22-28-03_genericagent-gpt-4-1-AP(SAX-M)2025-04-14-on-webmall-short-advanced-ax

It uses the existing functions from the repository:
- summarize_all_tasks_in_subdirs: Creates task_summary.json for each task and study_summary.json
- process_study_directory: Creates action_statistics.csv

Usage:
    python calculate_missing_results.py
"""

import os
import sys
import gzip
import pickle
from glob import glob

# Add the repository root to the path so imports work correctly
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

# Import the analysis functions
from analyze_agentlab_results.summarize_study import summarize_all_tasks_in_subdirs
from analyze_agentlab_results.aggregate_log_statistics import process_study_directory


def inspect_pickle_files(study_dir: str):
    """
    Inspect pickle files in the study directory to understand their structure.
    """
    print(f"\n{'='*80}")
    print(f"INSPECTING PICKLE FILES IN: {os.path.basename(study_dir)}")
    print(f"{'='*80}")
    
    if not os.path.isdir(study_dir):
        print(f"  Directory does not exist!")
        return
    
    # Find all subdirectories (task directories)
    task_dirs = [d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))]
    print(f"\n  Found {len(task_dirs)} task directories")
    
    if not task_dirs:
        return
    
    # Check the first task directory
    first_task_dir = os.path.join(study_dir, task_dirs[0])
    print(f"\n  Checking first task: {task_dirs[0]}")
    
    # List all files in the task directory
    all_files = os.listdir(first_task_dir)
    print(f"  Files in task directory: {all_files}")
    
    # Look for pickle files (any format)
    pkl_files = [f for f in all_files if '.pkl' in f]
    print(f"  Pickle files found: {pkl_files}")
    
    # Try to read the first pickle file
    if pkl_files:
        first_pkl = os.path.join(first_task_dir, pkl_files[0])
        print(f"\n  Reading: {pkl_files[0]}")
        try:
            if first_pkl.endswith('.gz'):
                with gzip.open(first_pkl, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(first_pkl, 'rb') as f:
                    data = pickle.load(f)
            
            print(f"  Data type: {type(data)}")
            print(f"  Data attributes: {dir(data)}")
            
            # Try to print some useful info
            if hasattr(data, '__dict__'):
                print(f"  Data __dict__ keys: {list(data.__dict__.keys())}")
            if hasattr(data, 'step'):
                print(f"  Step number: {data.step}")
            if hasattr(data, 'action'):
                print(f"  Action: {data.action}")
            if hasattr(data, 'reward'):
                print(f"  Reward: {data.reward}")
            if hasattr(data, 'terminated'):
                print(f"  Terminated: {data.terminated}")
            if hasattr(data, 'truncated'):
                print(f"  Truncated: {data.truncated}")
                
        except Exception as e:
            print(f"  Error reading pickle: {e}")
            import traceback
            traceback.print_exc()


def calculate_results_for_study(study_dir: str) -> bool:
    """
    Calculate all results for a single study directory.
    
    Args:
        study_dir: Path to the study directory
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.isdir(study_dir):
        print(f" Error: Directory does not exist: {study_dir}")
        return False
    
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(study_dir)}")
    print(f"Full path: {study_dir}")
    print(f"{'='*80}\n")
    
    try:
        # Step 1: Summarize all tasks (creates task_summary.json for each task and study_summary.json)
        print(" Step 1: Running summarize_all_tasks_in_subdirs...")
        summarize_all_tasks_in_subdirs(study_dir)
        print(" Task summaries completed\n")
        
        # Step 2: Process study directory (creates action_statistics.csv)
        print(" Step 2: Running process_study_directory...")
        success = process_study_directory(study_dir)
        if success:
            print(" Action statistics completed\n")
        else:
            print(" No action statistics generated (possibly no valid log data)\n")
        
        return True
        
    except Exception as e:
        print(f" Error processing {study_dir}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process the specified study directories."""
    
    # Define the study directories to process
    # UPDATED: Corrected folder names with AP(SAX-M) in the path
    study_dirs = [
        r"C:\Users\lenna\agentlab_results\2025-11-28_22-28-05_genericagent-gpt-4-1-APSAX-2025-04-14-on-webmall-short-basic-ax"
    ]
    
    print("="*80)
    print("CALCULATING MISSING RESULTS")
    print("="*80)
    print(f"\nStudy directories to process: {len(study_dirs)}")
    for i, d in enumerate(study_dirs, 1):
        print(f"  {i}. {os.path.basename(d)}")
    
    # First, inspect pickle files to understand the data
    for study_dir in study_dirs:
        inspect_pickle_files(study_dir)
    
    # Process each study directory
    results = []
    for study_dir in study_dirs:
        success = calculate_results_for_study(study_dir)
        results.append((study_dir, success))
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = sum(1 for _, s in results if s)
    print(f"\nProcessed {successful}/{len(study_dirs)} study directories successfully\n")
    
    for study_dir, success in results:
        status = " Success" if success else " Failed"
        print(f"  {status}: {os.path.basename(study_dir)}")
    
    print("\n" + "="*80)
    print("OUTPUT FILES CREATED (per study):")
    print("="*80)
    print("  - task_summary.json (in each task subdirectory)")
    print("  - study_summary.json (in study root)")
    print("  - action_statistics.csv (in study root)")
    print("="*80)
    
    return 0 if all(s for _, s in results) else 1


if __name__ == "__main__":
    exit(main())
