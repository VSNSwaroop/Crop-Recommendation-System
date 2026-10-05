# Contributing

Thank you for considering a contribution to the Crop Recommendation System.

## Before you start

Please search existing issues before opening a new one. For substantial changes, open an issue first so the approach can be discussed before implementation.

## Development setup

1. Fork and clone the repository.
2. Create a virtual environment.
3. Install runtime and development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
4. Copy `.env.example` to `.env` and use local-only values.
5. Run:
   ```bash
   pytest -q
   ruff check .
   ```

## Pull requests

Keep pull requests focused. Explain:

- what problem is being solved
- what changed
- how the change was tested
- whether model files, datasets, or database behavior changed

Do not commit secrets, personal data, local databases, virtual environments, or generated caches.

## Machine-learning contributions

If a contribution changes model behavior, include enough information to reproduce and evaluate it where practical:

- dataset source and license
- preprocessing
- train/test split
- model configuration
- evaluation metrics
- known limitations

Avoid reporting accuracy without describing the evaluation data.

## Security

Do not open a public issue for an undisclosed security vulnerability. Follow SECURITY.md instead.
