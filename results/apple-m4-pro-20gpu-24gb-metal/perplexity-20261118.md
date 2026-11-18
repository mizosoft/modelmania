### 2026-11-18 on apple-m4-pro-20gpu-24gb-metal (llama.cpp a300a9c, Metal)

`llama-perplexity -m models/<file>.gguf -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 16`

| Qwen3.5-4B | PPL (wikitext-2, 16x512 tokens) | cloud VM (CPU), same file |
|---|---|---|
| Q4_K_M | 11.1007 +/- 0.48424 | 11.07 +/- 0.48 |
| Q4_0   | 11.1907 +/- 0.49151 | 11.17 +/- 0.49 |

2.9 s per pass of 4x512 tokens. Metal and CPU agree to within 0.03 (different
summation order), far inside the error bars.
