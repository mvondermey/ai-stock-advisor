"""
Daily Trading Scheduler
Runs live trading at market open every weekday
"""

import schedule
import time
import subprocess
import sys
from datetime import datetime
import pytz

# Configuration
TRADING_TIME = "09:35"  # 9:35 AM ET (5 min after market open)
TIMEZONE = pytz.timezone('America/New_York')


def is_weekday():
    """Check if today is a weekday (Monday-Friday)"""
    now = datetime.now(TIMEZONE)
    return now.weekday() < 5  # Monday=0, Friday=4


def run_trading():
    """Execute the live trading script"""
    if not is_weekday():
        now = datetime.now(TIMEZONE)
        print(f"⏭️  Skipping: Today is {now.strftime('%A')} (weekend)")
        return
    
    now = datetime.now(TIMEZONE)
    print(f"\n{'='*80}")
    print(f"🤖 Starting Live Trading")
    print(f"📅 {now.strftime('%A, %B %d, %Y at %I:%M %p ET')}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "src/live_trading.py"],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ Trading completed at {datetime.now(TIMEZONE).strftime('%I:%M %p ET')}")
        else:
            print(f"\n❌ Trading failed with exit code {result.returncode}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print(f"{'='*80}\n")


def main():
    """Main scheduler loop"""
    print("=" * 80)
    print("🤖 AI STOCK ADVISOR - DAILY TRADING SCHEDULER")
    print("=" * 80)
    print(f"⏰ Scheduled to run at {TRADING_TIME} ET on weekdays")
    print(f"📍 Current time: {datetime.now(TIMEZONE).strftime('%I:%M %p ET')}")
    print("=" * 80)
    print("\n✅ Scheduler is running. Press Ctrl+C to stop.\n")
    
    # Schedule for each weekday
    schedule.every().monday.at(TRADING_TIME).do(run_trading)
    schedule.every().tuesday.at(TRADING_TIME).do(run_trading)
    schedule.every().wednesday.at(TRADING_TIME).do(run_trading)
    schedule.every().thursday.at(TRADING_TIME).do(run_trading)
    schedule.every().friday.at(TRADING_TIME).do(run_trading)
    
    # Keep running
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\n\n⛔ Scheduler stopped")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            time.sleep(300)  # Wait 5 min before retrying


if __name__ == "__main__":
    main()










