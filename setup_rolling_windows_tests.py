#!/usr/bin/env python3
"""
Setup Rolling Windows Testing Infrastructure

This script sets up the testing infrastructure for rolling windows compliance.
Run this once to configure everything.

Usage:
    python setup_rolling_windows_tests.py
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    dependencies = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0", 
        "pytest-xdist>=3.0.0",
        "pandas>=1.5.0",
        "numpy>=1.20.0"
    ]
    
    for dep in dependencies:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
            print(f"   ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {dep}")
            return False
    
    return True

def setup_pre_commit():
    """Setup pre-commit hooks."""
    print("🔧 Setting up pre-commit hooks...")
    
    try:
        # Install pre-commit
        subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"], 
                     check=True, capture_output=True)
        
        # Install hooks
        subprocess.run(["pre-commit", "install"], check=True, capture_output=True)
        
        print("   ✅ Pre-commit hooks installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to setup pre-commit: {e}")
        return False

def create_test_directories():
    """Create necessary directories."""
    print("📁 Creating test directories...")
    
    directories = [
        "tests",
        ".github/workflows",
        ".github"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    return True

def verify_files():
    """Verify all test files exist."""
    print("🔍 Verifying test files...")
    
    required_files = [
        "tests/test_rolling_windows.py",
        "tests/run_rolling_windows_tests.py", 
        ".github/workflows/rolling_windows_tests.yml",
        ".pre-commit-config.yaml"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
        return False
    
    return True

def run_initial_tests():
    """Run initial tests to verify setup."""
    print("🧪 Running initial tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "tests/run_rolling_windows_tests.py", 
            "--strategy", "current_date"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("   ✅ Initial tests passed")
            return True
        else:
            print(f"   ❌ Initial tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⏰ Tests timed out")
        return False
    except Exception as e:
        print(f"   ❌ Error running tests: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Rolling Windows Testing Infrastructure")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    steps = [
        ("Creating directories", create_test_directories),
        ("Installing dependencies", install_dependencies),
        ("Setting up pre-commit hooks", setup_pre_commit),
        ("Verifying files", verify_files),
        ("Running initial tests", run_initial_tests),
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        if not step_func():
            print(f"❌ Failed at {step_name}")
            return False
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📖 Next steps:")
    print("   1. Run tests: python tests/run_rolling_windows_tests.py")
    print("   2. Run specific tests: python tests/run_rolling_windows_tests.py --strategy 3m_1y_ratio")
    print("   3. Pre-commit hooks will run automatically before commits")
    print("   4. CI/CD will run tests on push and pull requests")
    print("\n🔍 Test categories:")
    print("   • 3m_1y_ratio - Test 3M/1Y Ratio strategy")
    print("   • 1y_3m_ratio - Test 1Y/3M Ratio strategy")
    print("   • turnaround - Test Turnaround strategy")
    print("   • current_date - Test parameter compliance")
    print("   • static_detection - Test for static behavior")
    print("   • performance_filters - Test filter rolling windows")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
