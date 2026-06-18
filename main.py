"""Project entry point."""

from pathlib import Path
import sys


def _ensure_local_src_path() -> None:
    src_path = Path(__file__).resolve().parent / "src"
    src_path_value = str(src_path)
    if src_path.is_dir() and src_path_value not in sys.path:
        sys.path.insert(0, src_path_value)


def main() -> None:
    """Run the project command-line entry point."""
    _ensure_local_src_path()
    from techchallenge_fase2.cli import main as run_cli

    run_cli()


if __name__ == "__main__":
    main()
