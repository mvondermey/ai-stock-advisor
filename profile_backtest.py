"""
Profile backtesting performance to identify bottlenecks.
Run: python profile_backtest.py
"""

import cProfile
import pstats
import io
from pstats import SortKey

def profile_backtest():
    """Run backtesting with profiling enabled."""
    
    # Import main after setting up profiler
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from main import main
    
    # Create profiler
    profiler = cProfile.Profile()
    
    print("🔍 Starting profiled backtest run...")
    print("=" * 80)
    
    # Run with profiling
    profiler.enable()
    try:
        main()
    finally:
        profiler.disable()
    
    print("\n" + "=" * 80)
    print("📊 PROFILING RESULTS")
    print("=" * 80)
    
    # Create stats object
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    
    # Sort by cumulative time and print top 30 functions
    print("\n🔥 TOP 30 FUNCTIONS BY CUMULATIVE TIME:")
    print("-" * 80)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(30)
    print(s.getvalue())
    
    # Sort by total time in function (excluding subcalls)
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    print("\n⏱️  TOP 30 FUNCTIONS BY TOTAL TIME (excluding subcalls):")
    print("-" * 80)
    stats.sort_stats(SortKey.TIME)
    stats.print_stats(30)
    print(s.getvalue())
    
    # Save detailed stats to file
    stats = pstats.Stats(profiler)
    stats.dump_stats('backtest_profile.prof')
    print("\n💾 Detailed profile saved to: backtest_profile.prof")
    print("   View with: python -m pstats backtest_profile.prof")
    print("   Or visualize with snakeviz: pip install snakeviz && snakeviz backtest_profile.prof")

if __name__ == "__main__":
    profile_backtest()
