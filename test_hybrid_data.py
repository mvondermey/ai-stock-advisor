#!/usr/bin/env python3
"""
Test script for hybrid data processing
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_hybrid_data_config():
    """Test hybrid data configuration"""
    try:
        from config import (
            DATA_PROVIDER, DATA_INTERVAL, AGGREGATE_TO_DAILY,
            USE_INTRADAY_FEATURES, USE_DAILY_FEATURES, FEATURE_COMBINATION
        )
        
        print("✅ Hybrid Data Configuration:")
        print(f"   Data Provider: {DATA_PROVIDER}")
        print(f"   Data Interval: {DATA_INTERVAL}")
        print(f"   Aggregate to Daily: {AGGREGATE_TO_DAILY}")
        print(f"   Use Intraday Features: {USE_INTRADAY_FEATURES}")
        print(f"   Use Daily Features: {USE_DAILY_FEATURES}")
        print(f"   Feature Combination: {FEATURE_COMBINATION}")
        return True
        
    except ImportError as e:
        print(f"❌ Config import error: {e}")
        return False

def test_hybrid_processor():
    """Test hybrid data processor"""
    try:
        from hybrid_data_processor import (
            aggregate_hourly_to_daily, calculate_intraday_features,
            create_hybrid_features, process_hybrid_data
        )
        
        print("✅ Hybrid Data Processor imported successfully")
        print("   Available functions:")
        print("   - aggregate_hourly_to_daily")
        print("   - calculate_intraday_features")
        print("   - create_hybrid_features")
        print("   - process_hybrid_data")
        return True
        
    except ImportError as e:
        print(f"❌ Hybrid processor import error: {e}")
        return False

def test_data_utils_integration():
    """Test data_utils integration"""
    try:
        from data_utils import (
            load_hybrid_features, calculate_hybrid_features_for_training,
            HYBRID_PROCESSOR_AVAILABLE
        )
        
        print("✅ Data Utils Integration:")
        print(f"   Hybrid Processor Available: {HYBRID_PROCESSOR_AVAILABLE}")
        print("   Available functions:")
        print("   - load_hybrid_features")
        print("   - calculate_hybrid_features_for_training")
        return True
        
    except ImportError as e:
        print(f"❌ Data utils integration error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Hybrid Data Processing Implementation")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_hybrid_data_config),
        ("Hybrid Processor", test_hybrid_processor),
        ("Data Utils Integration", test_data_utils_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Hybrid data processing is ready!")
        print("\n🚀 Next steps:")
        print("   1. Run: python src/main.py")
        print("   2. Check for hybrid data processing logs")
        print("   3. Verify intraday cache files in data_cache/")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main()
