"""Run a dev script and copy its stdout to a log file.

usage: run_log.py <log file> <script.py> [args...]
"""
import runpy
import sys


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


log, script, *args = sys.argv[1:]
with open(log, "w") as fh:
    sys.stdout = Tee(sys.__stdout__, fh)
    sys.argv = [script, *args]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.stdout = sys.__stdout__
