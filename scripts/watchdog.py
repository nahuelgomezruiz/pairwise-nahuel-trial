#!/usr/bin/env python3
import subprocess
import time
import os
import glob
from datetime import datetime

CLUSTERS = [
    'car_free_cities',
    'driverless_cars_policy',
    'electoral_college_debate',
    'emotion_recognition_schools',
    'face_on_mars_evidence',
    'seagoing_cowboys_program',
    'venus_exploration_worthiness',
]

RESTART_CMD = "source venv/bin/activate && python scripts/restart_optimizations.py 2>&1 | tee -a watchdog_restart.log &"


def is_process_running(match: str) -> bool:
    try:
        res = subprocess.run(
            f"ps aux | grep -E '{match}' | grep -v grep",
            shell=True,
            capture_output=True,
            text=True,
        )
        return bool(res.stdout.strip())
    except Exception:
        return False


def optimized_count() -> int:
    files = glob.glob('src/data/cluster_samples/*_optimized.csv')
    return len(files)


def main():
    print("=" * 80)
    print("WATCHDOG STARTED", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 80)

    target = len(CLUSTERS)

    while True:
        try:
            done = optimized_count()
            all_running = is_process_running("python.*(restart_optimizations|optimize_grading|pairwise)")

            if done >= target:
                print(f"✅ All clusters optimized ({done}/{target}). Exiting watchdog.")
                break

            if not all_running:
                print("⚠️ Pipeline not detected. Restarting...")
                subprocess.run(RESTART_CMD, shell=True)
                time.sleep(5)
            else:
                print(f"OK: running. Optimized so far: {done}/{target}")

            time.sleep(60)
        except KeyboardInterrupt:
            print("Stopping watchdog.")
            break
        except Exception as e:
            print(f"Watchdog error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()