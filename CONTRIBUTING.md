# Contributing to pygifi

Thanks for your interest in improving `pygifi`.

## Getting Started

1. Fork the repository and create a feature branch.
2. Create a virtual environment.
3. Install the project in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

## Development Workflow

1. Make focused changes with clear commit messages.
2. Add or update tests for any behavior you change.
3. Run the test suite before opening a pull request:

```bash
pytest tests/ --ignore=tests/test_parity.py -v
```

If you have R and the `Gifi` package installed, you can also run the parity checks described in `README.md`.

## Reporting Bugs

Please open a GitHub issue at:

`https://github.com/vam-math/pyGifi/issues`

When possible, include:

- Your Python version
- Your operating system
- A small reproducible example
- The full traceback or error output

## Proposing Changes

Pull requests are welcome for bug fixes, tests, documentation improvements, and new examples.

Please keep pull requests scoped and explain:

- What changed
- Why it changed
- How you tested it

## Support

For questions about usage, installation, or expected behavior, please open a GitHub issue with a short description of what you are trying to do.

## Code Style

- Follow the existing project structure and naming style.
- Prefer small, readable functions.
- Update documentation when public behavior changes.
