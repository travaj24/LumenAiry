"""Keep the standalone, lumenairy-free oracle SCRIPTS in this directory out of
pytest collection.  They are CLI tools (``python debye_oracle_v3.py job.json``)
and adversarial cross-checks, not test modules -- importing them at collection
time would try to read ``sys.argv`` jobs.  The unit tests that EXERCISE the
oracles live under ``tests/unit/`` (e.g. ``test_niche_p2_*.py``)."""

collect_ignore_glob = ["*.py"]
