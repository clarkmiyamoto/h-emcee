# tests/run_tests.py
#!/usr/bin/env python
"""
Repo-wide pytest runner.

- Run from anywhere: `python tests/run_tests.py`
- Pass through any pytest args: `python tests/run_tests.py -k affine -m "not slow" -q`
- In CI (CI=1), it will fail fast and be a bit stricter.
"""
from pathlib import Path
import os
import sys

def main() -> int:
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("pytest is not installed. Try: pip install -U pytest", file=sys.stderr)
        return 1

    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # repo/tests/run_tests.py -> repo/
    os.chdir(repo_root)          # run pytest from repo root

    # Ensure `src/` is importable (so tests can do `import yourpackage`)
    src_path = repo_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    # Default args if none are provided
    default_args = ["-vv", "-ra", "--color=yes", "tests"]
    args = sys.argv[1:] or default_args

    # Be a bit stricter on CI
    if os.getenv("CI"):
        # Fail on first failure unless overridden
        if "-x" not in args and not any(a.startswith("--maxfail") for a in args):
            args = ["-x", "--maxfail=1", *args]
        # Ensure we see durations for slow tests
        if not any(a.startswith("--durations") for a in args):
            args = ["--durations=10", *args]

    # Optional: if you often run with coverage, just pass it in:
    #   python tests/run_tests.py --cov=src --cov-report=term-missing
    import pytest
    return pytest.main(args)

if __name__ == "__main__":
    raise SystemExit(main())
