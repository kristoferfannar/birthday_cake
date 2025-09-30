# Project 2, Birthday Cake

This is the simulator for [COMS 4444, F25 project 2](https://www.cs.columbia.edu/~kar/4444f25/node19.html).

## Setup

Start with installing uv, uv is a modern python package manager.

- [UV Install instructions](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

Using brew:

```bash
brew install uv
```

### macOS (homebrew)

Python from homebrew doesn't include necessary graphical libraries `tkinter`.
If you're using a Python interpreter from homebrew, you'll need the following library:

```bash
brew install python-tk@3.13
```

## Running the simulator

```bash
uv run main.py <CLI_ARGS>
```

---

### CLI Arguments

The simulation can be configured using a variety of command-line arguments. If no arguments are provided, the simulation will run with a default set of parameters.

#### General Options

| Argument        | Default    | Description                                                                                                                                                                      |
| :-------------- | :--------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--gui`         | `False`    | Launches the graphical user interface to visualize the simulation. If omitted, the simulation runs in the command line and outputs text.                                         |
| `--player`      | `r`        | Sets the ID of the player to run. By default, runs the random player.                                                                                                            |
| `--children`    | `10`       | Sets the total number of unique subjects in the simulation.                                                                                                                      |
| `--debug`       | `False`    | Determines whether debug information is provided.                                                                                                                                |
| `--seed`        | `<random>` | Provides a seed for the random number generator to ensure reproducible simulations.                                                                                              |
| `--import-cake` | `N/A`      | Provide a file path to load the cake from. Must be inside the `cakes/` directory. If omitted, a cake is generated from the random seed. Mutually exclusive with `--export-cake`. |
| `--export-cake` | `N/A`      | Provide a file path to save the generated cake to. Must be inside the `cakes/` directory. Mutually exclusive with `--import-cake`.                                               |
| `--sandbox`     | `False`    | Omits special cake restrictions set by the project. Used for admiring cakes without cutting them.                                                                                |

---

### Code Quality and Formatting

The repository uses Ruff for both formatting and linting, if your PR does not pass the CI checks it won't be merged.

VSCode has a Ruff extension that can run on save. [Editor Setup](https://docs.astral.sh/ruff/editors/setup/).

To run formatting check:

```bash
uv run ruff format --check
```

To run formatting:

```bash
uv run ruff format
```

To run linting:

```bash
uv run ruff check
```

To run linting with auto-fix:

```bash
uv run ruff check --fix
```

---

### Usage Examples

Here are some common examples of how to run the simulator with different configurations.

##### Example 0: Run with default params

This is how to run the simulator with the default configuration.
This will select a random, undeterministic seed and will generate a cake based off that seed.

```bash
uv run main.py
```

This is equivalent to

```bash
uv run main.py --player r --children 10
```

##### Example 1: Run with the GUI

To run the simulator with the GUI, use the `--gui` flag.

```bash
uv run main.py --gui
```

##### Example 2: Run a Simulator with existing cake.

To run the simulator with a specific cake object, use the `--import-cake` flag.

```bash
uv run main.py --import-cake cakes/cake.csv
```

##### Example 3: Run a Simulator and save the cake.

To run the simulator with a generated cake, save the cake to a file path using `--export-cake`.

```bash
uv run main.py --export-cake cakes/new_cake.csv
```

##### Example 4: Run a Simulator with known seed.

To run the simulator with a known, consistent seed, specify it with a `--seed` value.

```bash
uv run main.py --seed 123
```
