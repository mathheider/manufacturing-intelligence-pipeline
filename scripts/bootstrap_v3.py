# -*- coding: utf-8 -*-
"""Superseded by summary_percent_v3.py (plain percentages + 95% ranges). Kept as an alias.
    python scripts/bootstrap_v3.py
"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("summary_percent_v3.py")), run_name="__main__")
