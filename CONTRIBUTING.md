# Contributing to Nexus Equity Terminal

First off, thank you for considering contributing to the Nexus Equity Terminal! It's people like you that make the open-source community around AI-driven financial analytics such a great place to learn, inspire, and create.

## How Can I Contribute?

### Reporting Bugs
If you find a bug regarding the LLM output parsing, the FAISS vector indexing, or a UI component failing to render, please submit an Issue on GitHub. Provide your error logs and ideally the specific SEC 10-K PDF you were trying to ingest when the error occurred.

### Proposing Enhancements
We are always looking to expand the quantitative capabilities of the terminal. If you have a suggestion:
- Expand the `simulate_dcf()` function to include WACC calculation from raw Beta, Risk-Free Rate, and Equity Risk Premium.
- Expand the `macro_universe.json` baseline to include more than 30 S&P 500 companies.
- Introduce additional Large Language Models via HuggingFace's Inference API.

### Pull Requests
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. Keep your PRs highly focused. Do not mix unrelated changes in the same Pull Request.
4. Ensure the UI aesthetic (dark-mode, dense metric cards) is maintained.
5. Issue that pull request!

## Academic Contributions
If you are extending this platform for an academic thesis at a business school (like Warwick), feel free to open a Discussion regarding your methodology prior to submitting code.

Thank you!
