# Contributing

Thank you for your interest in contributing to AtomVoxelizer!

## Reporting Issues

If you encounter a bug or unexpected behavior, please open a GitHub Issue and include:

- Operating system
- Python version
- AtomVoxelizer version
- Minimal reproducible example
- Complete error traceback (if applicable)

## Development Setup

Clone the repository and install in editable mode:

```bash
git clone https://github.com/<username>/atomvoxelizer.git
cd atomvoxelizer
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Coding Style

Please follow the existing code style.

- Format code with `black`
- Sort imports with `isort`
- Keep functions focused and well documented
- Add type hints where practical

## Testing

Before submitting a pull request, run the test suite:

```bash
pytest
```

If you add new functionality, please include corresponding tests.

## Pull Requests

Please:

- Keep pull requests focused on a single feature or bug fix.
- Write clear commit messages.
- Update documentation when behavior changes.
- Ensure all tests pass.

## Documentation

Documentation is built with Sphinx.

To build locally:

```bash
cd docs
make html
```

## Questions

If you're unsure about a feature or implementation, feel free to open an Issue to discuss it before starting work.

Thank you for helping improve AtomVoxelizer!