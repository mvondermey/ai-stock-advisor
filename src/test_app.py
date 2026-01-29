from ticker_selection import _get_current_1y_return_from_cache
from datetime import datetime, timezone

today = datetime.now(timezone.utc)
print("Testing APP performance...")

perf_3m = _get_current_1y_return_from_cache('APP', {}, today, 90)
perf_1y = _get_current_1y_return_from_cache('APP', {}, today, 365)

print(f"APP 3M: {perf_3m}%")
print(f"APP 1Y: {perf_1y}%")
