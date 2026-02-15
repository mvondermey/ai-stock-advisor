#!/usr/bin/env python3
"""
Rolling Windows Test Runner

Run this script to verify all strategies use rolling windows correctly.
This can be run manually or integrated into CI/CD pipelines.

Usage:
    python run_rolling_windows_tests.py
    python run_rolling_windows_tests.py --verbose
    python run_rolling_windows_tests.py --strategy 3m_1y_ratio
"""

import sys
import os
import argparse
import subprocess
from datetime import datetime

def run_tests(strategy_filter=None, verbose=False):
    """Run the rolling windows compliance tests."""
    
    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(project_root)
    sys.path.insert(0, project_root)
    
    # Prepare pytest command
    pytest_args = [
        sys.executable, "-m", "pytest",
        "tests/test_rolling_windows.py",
        "-v" if verbose else "-q",
        "--tb=short",  # Short traceback format
        "--color=yes",  # Colored output
    ]
    
    # Add strategy filter if specified
    if strategy_filter:
        pytest_args.extend(["-k", strategy_filter])
    
    # Add custom markers for different test categories
    pytest_args.extend([
        "-m", "not slow",  # Skip slow tests by default
    ])
    
    print("Running Rolling Windows Compliance Tests")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Strategy filter: {strategy_filter or 'All strategies'}")
    print(f"Verbose mode: {verbose}")
    print("=" * 60)
    
    try:
        # Run pytest
        result = subprocess.run(pytest_args, cwd=project_root, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Print summary
        print("=" * 60)
        if result.returncode == 0:
            print("ALL ROLLING WINDOWS TESTS PASSED!")
            print("All strategies are properly using rolling windows.")
        else:
            print("SOME TESTS FAILED!")
            print("Issues detected with rolling windows implementation.")
            print("Check the output above for details.")
        
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run rolling windows compliance tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--strategy", "-s", help="Filter by strategy name (e.g., 3m_1y_ratio, turnaround)")
    parser.add_argument("--list", "-l", action="store_true", help="List available test categories")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available test categories:")
        print("  • 3m_1y_ratio - Test 3M/1Y Ratio strategy")
        print("  • 1y_3m_ratio - Test 1Y/3M Ratio strategy") 
        print("  • turnaround - Test Turnaround strategy")
        print("  • current_date - Test current_date parameter compliance")
        print("  • static_detection - Test for static behavior patterns")
        print("  • performance_filters - Test performance filter rolling windows")
        print("  • future_dates - Test for future date usage (NEW!)")
        print("  • data_boundaries - Test data boundary compliance")
        print("  • historical_only - Test historical data only usage")
        print("\nUsage examples:")
        print("  python run_rolling_windows_tests.py --strategy 3m_1y_ratio")
        print("  python run_rolling_windows_tests.py --strategy current_date")
        print("  python run_rolling_windows_tests.py --strategy future_dates")
        print("  python run_rolling_windows_tests.py --verbose")
        return
    
    success = run_tests(strategy_filter=args.strategy, verbose=args.verbose)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
