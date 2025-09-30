from argparse import ArgumentParser
from dataclasses import dataclass
import os
import pathlib
from time import time


@dataclass
class Args:
    gui: bool
    player: int
    import_cake: str | None
    seed: int
    children: int
    export_cake: str | None
    debug: bool
    sandbox: bool


def get_cake_dir():
    return pathlib.Path(os.path.curdir + "/cakes/").resolve()


def sanitize_import_cake(org_import_cake: str | None) -> str | None:
    if org_import_cake is None:
        return org_import_cake

    cake_path = pathlib.Path(org_import_cake).resolve()

    cake_dir = get_cake_dir()

    if not cake_path.is_file():
        raise Exception(f'file with path "{cake_path}" not found')

    try:
        cake_path.relative_to(cake_dir)
    except ValueError:
        raise Exception('provided cake path file must be inside "cakes/" directory')

    return str(cake_path)


def sanitize_export_cake(org_export_cake: str | None) -> str | None:
    if org_export_cake is None:
        return org_export_cake

    cake_path = pathlib.Path(org_export_cake).resolve()

    cake_dir = get_cake_dir()

    if cake_path.exists():
        raise Exception(
            f"Can't export cake to '{org_export_cake}', path already in use."
        )

    try:
        cake_path.relative_to(cake_dir)
    except ValueError:
        raise Exception('provided cake path file must be inside "cakes/" directory')

    return str(cake_path)


def sanitize_seed(org_seed: None | str) -> int:
    if org_seed is None:
        seed = int(time() * 100_000) % 1_000_000
        print(f"Generated seed: {seed}")
        return seed

    return int(org_seed)


def sanitize_player(org_player: str) -> int:
    if org_player.isdigit() and 1 <= int(org_player) <= 10:
        return int(org_player)

    elif org_player == "r":
        return 0

    raise Exception(
        f'unknown `--player` value provided: "{org_player}". Expected digit 1<=10 or "r"'
    )


def get_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument("--gui", "-g", action="store_true", help="render GUI")
    parser.add_argument(
        "--seed", "-s", type=int, help="Seed used by random number generator"
    )
    parser.add_argument(
        "--player",
        "-p",
        default="r",
        help="Specify which player to run",
    )
    parser.add_argument(
        "--children",
        "-n",
        type=int,
        help="Number of children to serve cake to",
        default=10,
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Display debug info")
    parser.add_argument(
        "--sandbox",
        "-x",
        action="store_true",
        help="Load cakes in sandbox environment. Implies `--gui` flag.",
    )

    # users cannot import and export a cake simultaneously
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--import-cake", "-i", help="path to a cake file within cakes/")
    group.add_argument("--export-cake", "-e", help="path to save generated cake to")

    namespace = parser.parse_args()

    seed = sanitize_seed(namespace.seed)
    player = sanitize_player(namespace.player)
    import_cake = sanitize_import_cake(namespace.import_cake)
    export_cake = sanitize_export_cake(namespace.export_cake)

    args = Args(
        gui=namespace.gui or namespace.sandbox,
        player=player,
        import_cake=import_cake,
        seed=seed,
        children=namespace.children,
        export_cake=export_cake,
        debug=namespace.debug,
        sandbox=namespace.sandbox,
    )

    return args
