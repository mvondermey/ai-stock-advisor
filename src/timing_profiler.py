"""
Simple timing profiler for strategy execution.
Add timing decorators to measure strategy performance.
"""

import time
from functools import wraps
from collections import defaultdict

# Global timing storage
timing_stats = defaultdict(lambda: {'total': 0.0, 'count': 0, 'times': []})

def time_strategy(strategy_name):
    """Decorator to time strategy execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            
            timing_stats[strategy_name]['total'] += elapsed
            timing_stats[strategy_name]['count'] += 1
            timing_stats[strategy_name]['times'].append(elapsed)
            
            return result
        return wrapper
    return decorator

def print_timing_report():
    """Print timing statistics for all strategies."""
    print("\n" + "=" * 80)
    print("⏱️  STRATEGY TIMING REPORT")
    print("=" * 80)
    print(f"{'Strategy':<30} {'Total Time':<12} {'Avg Time':<12} {'Count':<8} {'% Total':<8}")
    print("-" * 80)
    
    total_time = sum(stats['total'] for stats in timing_stats.values())
    
    # Sort by total time
    sorted_stats = sorted(timing_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    
    for strategy_name, stats in sorted_stats:
        avg_time = stats['total'] / stats['count'] if stats['count'] > 0 else 0
        pct = (stats['total'] / total_time * 100) if total_time > 0 else 0
        
        print(f"{strategy_name:<30} {stats['total']:>10.2f}s {avg_time:>10.4f}s {stats['count']:>6} {pct:>6.1f}%")
    
    print("-" * 80)
    print(f"{'TOTAL':<30} {total_time:>10.2f}s")
    print("=" * 80)
    
    # Identify bottlenecks
    print("\n🔥 TOP 5 BOTTLENECKS:")
    for i, (strategy_name, stats) in enumerate(sorted_stats[:5], 1):
        pct = (stats['total'] / total_time * 100) if total_time > 0 else 0
        print(f"  {i}. {strategy_name}: {stats['total']:.2f}s ({pct:.1f}% of total)")

def reset_timing_stats():
    """Reset all timing statistics."""
    timing_stats.clear()
