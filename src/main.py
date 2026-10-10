"""Entry point: ``python -m src.main <command>``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
