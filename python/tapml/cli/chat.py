"""Command line entrypoint of chat."""
from tapml.help import HELP
from tapml.interface.chat import ChatConfigOverride, chat
from tapml.support.argparse import ArgumentParser


def main(argv):
    """Parse command line arguments and call `tapml.interface.chat`."""
    parser = ArgumentParser("TapML Chat CLI")

    parser.add_argument(
        "model",
        type=str,
        help=HELP["model"] + " (required)",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="O2",
        help=HELP["opt"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=HELP["device_deploy"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--overrides",
        type=ChatConfigOverride.from_str,
        default="",
        help=HELP["chatconfig_overrides"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--model-lib-path",
        type=str,
        default=None,
        help=HELP["model_lib_path"] + ' (default: "%(default)s")',
    )
    parsed = parser.parse_args(argv)
    chat(
        model=parsed.model,
        device=parsed.device,
        opt=parsed.opt,
        overrides=parsed.overrides,
        model_lib_path=parsed.model_lib_path,
    )
