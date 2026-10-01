# Contributing

Thanks for helping out. The project is deliberately small, so the easiest contributions are focused ones.

## Setup

```bash
git clone <your fork>
cd localseek
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

The tests use a built-in hashing embedder, so they run without downloading a model.

## Good first contributions

- **A new file format.** Write `extract_xxx(path) -> list[Section]` in `src/localseek/extractors/`, register the
  suffix in `extractors/__init__.py`, and add a test in `tests/test_extractors.py`. Raise `ExtractionError` with a
  clear message when a file cannot be read.
- **More evaluation data.** Add queries and documents under `eval/` and `examples/`. Queries list `relevant` documents as paths relative to the corpus folder; see the README.
- **Docs and error messages.** If something confused you, fix the wording.

## Ground rules

- **Nothing leaves the machine.** Do not add code that sends document text, queries, or file names over the network.
  The only allowed network use is the one-time model download in `embedder.py`.
- Keep dependencies small. Prefer the standard library when it is good enough.
- Add or update tests for behaviour changes. `ruff check src tests eval` should pass.
- Changing chunking or the default model can change retrieval quality, so run `eval/run_eval.py` before and after and
  include the numbers in your pull request.
