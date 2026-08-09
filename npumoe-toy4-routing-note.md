# NPUMoE toy 4: simulated routing

The capacity-tier toy now simulates 512 tokens per layer. Each token receives
popularity-biased expert logits with Gaussian noise, followed by softmax and
argmax routing. The per-expert loads are counts derived from these individual
routing decisions instead of values preset directly.

| Metric | Old | New |
| --- | ---: | ---: |
| Total real tokens | 2984 | 2941 |
| Total padding tokens | 1496 | 1187 |
| Total overflow-pruned tokens | 50 | 131 |
| Total computed slots | 4480 | 4128 |
| Zero-padding percentage | 33.39% | 28.75% |
