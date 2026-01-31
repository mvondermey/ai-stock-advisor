#!/usr/bin/env python3
"""
Quick Multi-Task Learning Test
Tests if the multi-task learning strategy is properly integrated.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_multitask_integration():
    """Test if multi-task learning is properly integrated."""
    
    print("🧠 Testing Multi-Task Learning Integration")
    print("=" * 50)
    
    try:
        # Test config import
        from config import ENABLE_MULTITASK_LEARNING, ENABLE_AI_STRATEGY
        print(f"✅ Config loaded:")
        print(f"   ENABLE_MULTITASK_LEARNING = {ENABLE_MULTITASK_LEARNING}")
        print(f"   ENABLE_AI_STRATEGY = {ENABLE_AI_STRATEGY}")
        
        # Test shared strategies import
        from shared_strategies import select_multitask_learning_stocks, MULTITASK_AVAILABLE
        print(f"✅ Shared strategies loaded:")
        print(f"   MULTITASK_AVAILABLE = {MULTITASK_AVAILABLE}")
        
        # Test multitask strategy import
        from multitask_strategy import MultiTaskStrategy, select_multitask_stocks
        print(f"✅ Multi-task strategy loaded:")
        print(f"   MultiTaskStrategy class available")
        print(f"   select_multitask_stocks function available")
        
        # Test basic strategy initialization
        strategy = MultiTaskStrategy()
        print(f"✅ MultiTaskStrategy initialized successfully")
        
        print(f"\n🎉 Multi-Task Learning Integration Test PASSED!")
        print(f"   Strategy is ready for backtesting")
        print(f"   Run 'python src/main.py' to see it in action")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multitask_integration()
    
    if success:
        print(f"\n📋 Next Steps:")
        print(f"1. Run 'python src/main.py' to test in full system")
        print(f"2. Look for '🧠 Multi-Task Learning' messages in output")
        print(f"3. Check final summary for multi-task results")
    else:
        print(f"\n💥 Fix integration issues before running main system")
