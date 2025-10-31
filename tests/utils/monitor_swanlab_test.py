"""
Simple smoke test for SwanlabMonitor.

Run:
	python cradle.py

What it does:
- Ensures SWANLAB_API_KEY is read from environment (sets a dummy if missing).
- Initializes SwanlabMonitor with minimal args.
- Logs a small metric and closes the run.

Notes:
- If `swanlab` is not installed, this script will print a helpful message and exit.
- The dummy API key is used only to exercise the login path; real authentication isn't required for this smoke test.
"""

import os
import sys


def main() -> int:
	# Defer imports to keep error handling simple
	try:
		from trinity.utils.monitor import SwanlabMonitor
	except Exception as e:
		print("[cradle] Failed to import SwanlabMonitor:", e)
		return 1

	# Ensure an env-based key path is exercised (uses dummy if not provided)
	env_keys = ["SWANLAB_API_KEY", "SWANLAB_APIKEY", "SWANLAB_KEY", "SWANLAB_TOKEN"]
	if not any(os.getenv(k) for k in env_keys):
		os.environ["SWANLAB_API_KEY"] = "dummy_key_for_smoke_test"
		print("[cradle] Set SWANLAB_API_KEY to a dummy value to test env-based login path.")

	# Try creating the monitor; if swanlab isn't installed, __init__ will assert
	try:
		mon = SwanlabMonitor(
			project="trinity-smoke",
			group="cradle",
			name="swanlab-env",
			role="tester",
			config=None,
		)
	except AssertionError as e:
		print("[cradle] SwanLab not available or not installed:", e)
		print("[cradle] Install swanlab to run this smoke test: pip install swanlab")
		return 0
	except Exception as e:
		print("[cradle] Unexpected error constructing SwanlabMonitor:", e)
		return 1

	# Log a minimal metric to verify basic flow
	try:
		mon.log({"smoke/metric": 1.0}, step=1)
		print("[cradle] Logged a test metric via SwanlabMonitor.")
	except Exception as e:
		print("[cradle] Error during logging:", e)
		try:
			mon.close()
		except Exception:
			pass
		return 1

	# Close cleanly
	try:
		mon.close()
		print("[cradle] SwanlabMonitor closed successfully.")
	except Exception as e:
		print("[cradle] Error closing monitor:", e)
		return 1

	print("[cradle] Smoke test completed.")
	return 0


if __name__ == "__main__":
	sys.exit(main())

