# Contributing to DecodeForge

we welcome contributions from compiler engineers, systems hackers, and inference runtime builders.

---

### engineering guidelines

1. **reproduce before fixing**: write a deterministic test reproducing the broken grammar or token mismatch before writing code.
2. **no pr without benchmark or test**: any change to the trie or mask generator must include throughput and latency benchmarks.
3. **strict typing**: all python code must pass `mypy` and `ruff check`. c++ kernel code must follow modern c++20 standards.
4. **one concern per pr**: keep pull requests atomic and reviewable.

---

### development workflow

```bash
# clone and setup environment
git clone https://github.com/devtyagi3909/decodeforge.git
cd decodeforge
pip install -e ".[dev]"

# run tests
pytest tests/ -v

# run linter and formatter
ruff check .
ruff format .
```

---

### submitting changes

1. fork the repository and create a feature branch (`git checkout -b feat/trie-simd-mask`).
2. sign off your commits (`git commit -s`).
3. push to your fork and submit a pull request with before-and-after benchmarks.
