# answer-generation

Generates answers over retrieved pages with qwen3.5:35b via Ollama, scores them with llama3.1:8b (different model family to avoid self-evaluation bias), and produces the final pass@1 comparison table.

Requires Ollama running locally with both models pulled:

```bash
ollama pull qwen3.5:35b
ollama pull llama3.1:8b
```

## Run

```bash
cd answer_generation && uv sync

bash run_all.sh  --condition jina_nemo    # generate answers (all 3 subsets, resume-safe)
bash run_judge.sh --condition jina_nemo   # judge answers

uv run answer_generation/analyze.py      # final comparison table
```

Available conditions: `jina_nemo`, `jina_nemo_reranked`, `jina_deepseek`, `jina_deepseek_reranked`, `colembed`, `hybrid_nemo`, `hybrid_deepseek`.

Single subset dry run:

```bash
uv run answer_generation/predict.py --subset computer_science --lang english --condition jina_nemo
uv run answer_generation/judge.py   --subset computer_science --lang english --condition jina_nemo
```

## Results (pass@1)

| Condition | CS | Finance | Pharma | avg |
|---|---|---|---|---|
| jina_nemo | 91.6% | 79.6% | 82.4% | 84.5% |
| jina_nemo_reranked | 92.6% | 83.5% | 88.5% | **88.2%** |
| jina_deepseek | 90.7% | 73.8% | 82.4% | 82.3% |
| jina_deepseek_reranked | **95.3%** | 79.0% | 88.5% | 87.6% |
| colembed | 93.0% | 74.8% | 83.2% | 83.7% |
| hybrid_nemo | 93.0% | 79.9% | 87.9% | 86.9% |
| hybrid_deepseek | **95.3%** | **81.6%** | **89.3%** | **88.7%** |
