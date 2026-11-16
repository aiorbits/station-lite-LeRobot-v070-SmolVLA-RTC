#!/usr/bin/env python3
"""Discover and run all multi-object closed-loop v070_demo unit tests offline."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_tests(loader: unittest.TestLoader, standard_tests: unittest.TestSuite, pattern: str) -> unittest.TestSuite:
    del loader, pattern
    demo_tests_dir = SRC_DIR / "v070_demo" / "tests"
    fresh_loader = unittest.TestLoader()
    discovered = fresh_loader.discover(
        start_dir=str(demo_tests_dir),
        pattern="test_*.py",
        top_level_dir=str(SRC_DIR),
    )
    standard_tests.addTests(discovered)
    return standard_tests


if __name__ == "__main__":
    unittest.main()
