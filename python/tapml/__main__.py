"""Entrypoint of all CLI commands from TapML"""
import sys

from tapml.support import logging
from tapml.support.argparse import ArgumentParser

logging.enable_logging()


def main():
    """Entrypoint of all CLI commands from TapML"""
    parser = ArgumentParser("TapML Command Line Interface.")
    parser.add_argument(
        "subcommand",
        type=str,
        choices=["compile", "convert_weight", "gen_config", "chat", "bench"],
        help="Subcommand to to run. (choices: %(choices)s)",
    )
    parsed = parser.parse_args(sys.argv[1:2])
    # pylint: disable=import-outside-toplevel
    if parsed.subcommand == "compile":
        from tapml.cli import compile as cli

        cli.main(sys.argv[2:])
    elif parsed.subcommand == "convert_weight":
        from tapml.cli import convert_weight as cli

        cli.main(sys.argv[2:])
    elif parsed.subcommand == "gen_config":
        from tapml.cli import gen_config as cli

        cli.main(sys.argv[2:])
    elif parsed.subcommand == "chat":
        from tapml.cli import chat as cli

        cli.main(sys.argv[2:])
    elif parsed.subcommand == "bench":
        from tapml.cli import bench as cli

        cli.main(sys.argv[2:])
    else:
        raise ValueError(f"Unknown subcommand {parsed.subcommand}")
    # pylint: enable=import-outside-toplevel


if __name__ == "__main__":
    main()
