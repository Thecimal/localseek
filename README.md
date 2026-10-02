# localseek

<!-- Tests Badge -->
[![Tests](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Thecimal/54b6d06aa7fa3f825c99d9727994276a/raw/localseek-junit-tests.json)](https://github.com/Thecimal/localseek/actions)

<!-- Coverage Badge -->
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Thecimal/54b6d06aa7fa3f825c99d9727994276a/raw/localseek-cobertura-coverage.json)](https://github.com/Thecimal/localseek/actions)

<!-- Flat Square Style -->
[![Tests](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Thecimal/54b6d06aa7fa3f825c99d9727994276a/raw/localseek-junit-tests.json&style=flat-square)](https://github.com/Thecimal/localseek/actions)

Private, on-device semantic search for your own documents.

Point it at a few folders and search them by meaning, not just exact words:
"the contract where the landlord agreed to fix the heating" finds the right file even if it never uses those words.
Everything runs on your computer with small CPU-friendly models. There is no account, no server, and no cloud.

```console
$ localseek index ~/Documents ~/Notes
Scanned 1,284 files: 1,284 added, 0 updated, 0 unchanged, 0 removed (9,731 new chunks).

$ localseek search who promised to fix the radiators
1. ~/Documents/flat/lease-notes.md  (1.00)
   ...The landlord agreed in writing to repair the heating system before the first cold spell and to replace
   the two broken radiators in the bedroom...
```

## Privacy

- Your documents, queries, and file names are never sent anywhere. The test suite fails if indexing or search open a
  network connection.
- The embedding model (about 100 MB) is downloaded **once**, on first use, and cached on disk. After that, localseek
  works fully offline. To be strict about it, download the model once, then set `HF_HUB_OFFLINE=1`.
- The index is a single SQLite file on your machine. It contains chunks of your document text, so treat it with the
  same care as the documents themselves.

## Install

Requires Python 3.11 or newer.

```bash
pipx install .          # from a clone of this repository
# or: pip install .
```

## Quick start

```bash
localseek index ~/Documents          # first run downloads the model, then indexes
localseek search tax deadline for freelancers
localseek update                     # re-scan every folder you have indexed (only changes are processed)
localseek serve                      # local search page at http://127.0.0.1:8765
```

## Commands

| Command | What it does |
|---|---|
| `index PATH...` | Index folders or files, and remember the folders. |
| `update` | Re-scan all remembered folders. Adds new files, re-indexes changed ones, removes deleted ones. |
| `search QUERY...` | Search. Options: `-n 10`, `--mode hybrid\|vector\|keyword`, `--type pdf`, `--since 2026-01-31`, `--path contracts`, `--json`. |
| `status` | Show the model, index size, and remembered folders. |
| `forget PATH` | Stop tracking a folder and remove its files from the index. |
| `watch` | Keep the index fresh by re-scanning every 30 seconds (`--interval N`). |
| `serve` | Start a small search page bound to `127.0.0.1` only. |

Global options: `--db PATH`, `--config PATH`, `--model NAME`.

## Supported files

Text and Markdown, source code and config files, PDF (text-based), Word `.docx`, HTML, and EPUB.
The full suffix list is in `src/localseek/extractors/text.py`. Adding a format takes one small function; see
[CONTRIBUTING.md](CONTRIBUTING.md).

Ignored by default: `.git`, `node_modules`, virtual environments, caches. Add a `.localseekignore` file (one glob per
line) to any indexed folder for more, or set `ignore` in the config file.

## Configuration

Optional. Copy [`examples/config.toml`](examples/config.toml) to `~/.config/localseek/config.toml`
(Windows: `%APPDATA%\localseek\config.toml`). Settings: `model`, `chunk_words`, `overlap_words`, `max_file_mb`,
`ignore`, `db_path`.

**Choosing a model.** The default, `BAAI/bge-small-en-v1.5`, is small and good for English. For other languages use a
multilingual model, for example `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Pick the model
*before* your first big index: changing it later means re-embedding everything with
`localseek update --rebuild`. Any model supported by [fastembed](https://github.com/qdrant/fastembed) works; check its
supported-model list for exact names.

## How it works

```
files -> scanner -> extractor -> chunker -> embedder -> SQLite (vectors + FTS5 + metadata)

query -> embedder -> vector search ─┐
query -> FTS5 / BM25 keyword search ┴> Reciprocal Rank Fusion -> best chunk per file -> results
```

- **Chunking:** about 220 words per chunk with 30 words of overlap, preferring paragraph boundaries. Markdown headings
  and PDF page numbers are kept as metadata.
- **Incremental updates:** files are skipped when size and modified time match. If only the modified time changed, a
  content hash decides. Chunks with identical text reuse their stored embedding, so nothing is embedded twice.
- **Hybrid ranking:** meaning-based matches and exact-word matches are merged, so both "that thing about heating" and
  a precise phrase or error code work.
- **Storage:** vectors are stored as BLOBs and searched with a brute-force NumPy dot product. That is fast and simple
  up to a few hundred thousand chunks. Beyond that, an approximate index (for example `sqlite-vec`) would be the next
  step.

## Measuring quality

```bash
python eval/run_eval.py --corpus examples/corpus --queries eval/queries.json --model BAAI/bge-small-en-v1.5
```

The script indexes a folder and reports recall@1/5/10 and MRR for vector, keyword, and hybrid search against a list of
queries with known answers. Use `--model hash` for a quick, lexical-only baseline without any download. Replace the
sample corpus and queries with your own documents for numbers that mean something for your data.

Each query names its relevant documents by path **relative to the corpus folder**, so files with the same name in
different folders stay distinct:

```json
{"query": "bread fermentation", "relevant": ["recipes/sourdough.md", "recipes/fermentation.md"]}
```

Recall@k is the fraction of a query's relevant documents found in the top k results, and MRR uses the rank of the first
relevant document. The script validates the dataset before indexing (malformed JSON, empty or duplicate entries, paths
that are absolute, outside the corpus, missing, or not indexable) and exits with status 2 and a list of problems.

### Benchmark

`eval/benchmark/` holds a larger benchmark: 62 mixed-format documents and 97 queries across eleven categories
(paraphrases, error codes, near-duplicate documents, answers deep inside long files, and more), each judgment backed by
an exact quote. Run it with `--by-category` to see where retrieval is weak. See `eval/benchmark/README.md` for what it
covers, what it does not, and the rules for changing it. The small `examples/corpus` set remains a quick demo.

Every run begins by printing the model, chunk settings, library versions and content fingerprints of the corpus and
queries, so two runs can be compared. `--show-misses` lists the queries whose first relevant document is not the top
result, and `--json FILE` saves the whole run (configuration and all metrics) in machine-readable form.

### Quality gate

By default the script only reports. Pass `--thresholds FILE` to make it fail when quality drops. The repository ships
a gate for the benchmark that CI runs on every pull request:

```bash
python eval/run_eval.py --corpus eval/benchmark/corpus --queries eval/benchmark/queries.json \
    --model hash --thresholds eval/benchmark/thresholds-hash.json --by-category
```

A thresholds file sets minimum values, overall per mode and optionally per query category:

```json
{
  "hybrid": {"recall@1": 0.8, "mrr": 0.85},
  "by_category": {"hybrid": {"late_answer": {"recall@5": 0.9}}},
  "meta": {"dataset_sha256": "...", "corpus_sha256": "..."}
}
```

Valid metrics are `recall@1`, `recall@5`, `recall@10`, and `mrr`; valid modes are `hybrid`, `vector`, and `keyword`.
Anything not listed is reported but not checked. Unknown modes, metrics and categories are rejected, so a typo cannot
silently disable a check. Category floors catch a collapse in one kind of query that an overall average would hide.
`meta` records which version of the benchmark the floors were derived from, and the run refuses to gate a different
version; `eval/derive_thresholds.py` builds a file from a recorded baseline by a fixed rule (see
`eval/benchmark/README.md`).

| Exit status | Meaning |
|---|---|
| 0 | Success, and every threshold was met (or none were given) |
| 1 | At least one threshold was not met; each failure is listed on stderr, e.g. `hybrid recall@1: 0.714 < required 0.850` |
| 2 | Invalid corpus, dataset, or thresholds file |

## Limitations

- Scanned PDFs and images have no extractable text. OCR is not supported yet, and such files are reported and skipped.
- Chunk whitespace is normalised, so snippets from source code lose their line breaks.
- The local search page shows results but cannot open files. Use "Copy path".
- Results show the best chunk per file, not every matching passage.

## Roadmap

- OCR for scanned PDFs and images
- Optional cross-encoder reranker for better top results
- Approximate nearest-neighbour index for very large collections
- MCP server so AI assistants can search your local index
- Optional local-LLM answers on top of retrieved passages
- Editor plugins (Obsidian, VS Code)

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests eval
```

## License

MIT. See [LICENSE](LICENSE).
