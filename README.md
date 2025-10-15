# llamacppR

Minimal R bindings for llama.cpp (CPU-only). This package vendors the upstream llama.cpp/ggml sources and exposes a single, simple R function for prompt completion using a local GGUF model.

## Features

- CPU-only inference via static linking (no GPU backends)
- Single-call API: `llama_simple(model, prompt, n_predict)`
- Ships llama.cpp as a vendored third-party dependency built at install time

## Requirements

- R (>= 3.6)
- CMake (>= 3.20)
- A C++17-capable toolchain
- A local GGUF model file (not provided by this package)

Notes:
- macOS links against Accelerate (BLAS) by default.
- Windows CI uses Rtools + MinGW via CMake.
- Linux builds may require minor linker flag adjustments depending on your BLAS setup.

## Install (from source)

From the repository root:

```sh
R CMD build .
R CMD INSTALL llamacppR_*.tar.gz
```

Alternatively, in an interactive R session:

```r
install.packages(".", repos = NULL, type = "source")
```

## Quick Example

This example loads a local GGUF model and generates a short continuation with greedy decoding.

```r
library(llamacppR)

# Path to a local GGUF model (example path — update for your system)
model <- "~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# Generate up to 48 new tokens
out <- llama_simple(
  model  = model,
  prompt = "Write a concise haiku about R programming:",
  n_predict = 48L
)

cat(out)
```

Expected behavior:
- Returns the prompt concatenated with the generated continuation as a single UTF-8 string.
- Uses greedy decoding (no temperature/top-p/top-k controls exposed).

## Models

- The function expects a path to a GGUF model file compatible with the vendored llama.cpp.
- This package does not download models for you. Obtain GGUF models separately and point `model` to the file.

## Limitations

- CPU-only; GPU backends are disabled.
- No streaming API; generation is returned after completion.
- Minimal sampling controls (greedy decoding only) in the R interface.

## How It Works

- `R/llamacpp.R` defines `llama_simple()` and calls a compiled routine via `.Call`.
- `src/llamacppR.cpp` bridges R to llama.cpp: loads the model, tokenizes the prompt, decodes greedily, and returns text.
- `src/Makevars` / `src/Makevars.win` invoke CMake to build static libraries from the vendored sources and link them into the R shared library.

## CI

- A Windows GitHub Actions workflow builds a source tarball and a binary zip artifact for convenience.

## License

MIT. See `LICENSE` for details. llama.cpp is included under its respective license in `src/third_party/llama.cpp`.

## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp) and the ggml maintainers for the core inference engine.
