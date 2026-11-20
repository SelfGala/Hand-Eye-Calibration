import sys
from handeye_core.cli import main as cli_main


def main():
    raise SystemExit(cli_main(["gui"] + sys.argv[1:]))

