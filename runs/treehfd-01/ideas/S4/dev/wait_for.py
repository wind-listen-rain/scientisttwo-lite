"""Block until a file contains at least N occurrences of a marker (or a Traceback), then print it.

usage: wait_for.py <file> <marker> <N>
"""
import sys
import time
from pathlib import Path

path, marker, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
while True:
    text = Path(path).read_text() if Path(path).exists() else ""
    if text.count(marker) >= count or "Traceback" in text:
        print(text)
        break
    time.sleep(5)
