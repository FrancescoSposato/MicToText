"""Allows running the app with `python -m mictotext`."""

from mictotext.cli import main

# The guard is mandatory on Windows: the transcriber uses multiprocessing "spawn",
# which re-imports this module in the child process.
if __name__ == "__main__":
    raise SystemExit(main())
