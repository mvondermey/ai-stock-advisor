"""
Line-by-line profiling for backtesting.
Install: pip install line_profiler
Run: python line_profile_backtest.py
"""

def profile_with_line_profiler():
    """Profile specific functions line-by-line."""
    try:
        from line_profiler import LineProfiler
    except ImportError:
        print("❌ line_profiler not installed")
        print("   Install with: pip install line_profiler")
        return
    
    # Import functions to profile
    from backtesting import _run_portfolio_backtest_walk_forward, _smart_rebalance_portfolio
    from main import main
    
    # Create profiler
    profiler = LineProfiler()
    
    # Add functions to profile (add the ones you suspect are slow)
    profiler.add_function(_run_portfolio_backtest_walk_forward)
    profiler.add_function(_smart_rebalance_portfolio)
    
    print("🔍 Starting line-by-line profiling...")
    print("=" * 80)
    
    # Wrap and run main
    profiler_wrapper = profiler(main)
    profiler_wrapper()
    
    print("\n" + "=" * 80)
    print("📊 LINE-BY-LINE PROFILING RESULTS")
    print("=" * 80)
    
    # Print results
    profiler.print_stats()

if __name__ == "__main__":
    profile_with_line_profiler()
