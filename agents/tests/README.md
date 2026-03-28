# Tests Structure

`agents/tests` is standardized into:

- `unit/`: executable unit test cases (`test_*.py`)
- `utils/`: shared test helpers (path bootstrap, import stubs)

Run all tests:

```bash
python -m unittest discover -s agents/tests -p "test_*.py" -v
```

