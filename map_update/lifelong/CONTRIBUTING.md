# Contributing

1. Work on a feature branch.
2. Keep the current GLUEMAP reconstruction immutable in all tests and experiments.
3. Add a regression test before changing any gate, metric, coordinate convention, or map format.
4. Do not promote historical-only geometry into the production map.
5. Run `pytest` and `ruff check .` before opening a pull request.
6. Document experiment data splits by session or flight; random adjacent-frame splits are not accepted.
