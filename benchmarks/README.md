# Benchmarks
* Download openai_humaneval dataset
```bash
huggingface-cli download --resume-download --repo-type dataset --local-dir <path/to/SpecOffload>/benchmarks/openai_humaneval/dataset --local-dir-use-symlinks False openai/openai_humaneval
```

* Download summeval dataset
```bash
huggingface-cli download --resume-download --repo-type dataset --local-dir <path/to/SpecOffload>/benchmarks/summeval_data/dataset --local-dir-use-symlinks False mteb/summeval
```

* Download samsum dataset
```bash
huggingface-cli download --resume-download --repo-type dataset --revision refs/convert/parquet --local-dir <path/to/SpecOffload>/benchmarks/samsum/dataset --local-dir-use-symlinks False Samsung/samsum
```

* Download ceval_exam dataset
```bash
huggingface-cli download --resume-download --repo-type dataset --local-dir <path/to/SpecOffload>/benchmarks/ceval-exam/dataset --local-dir-use-symlinks False ceval/ceval-exam
```

## Baseline evaluations
Following is the instructions for running evaluation for SpecOffload and 4 baselines: [Accelerate v1.5.2](https://github.com/dvmazur/mixtral-offloading),  [DeepSpeed Zero-Inference v0.16.1](https://github.com/deepspeedai/DeepSpeed), [FlexGen](https://github.com/FMInference/FlexLLMGen) and [Fiddler](https://github.com/efeslab/fiddler).

```bash
cd <path/to/specoffload>
# run benchmarks for Mixtral-8x7B
bash benchmarks/run-<method>.sh <DATASET> 8x7B
# run benchmarks for Mixtral-8x22B
bash benchmarks/run-<method>.sh <DATASET> 8x22B
```

## NOTE: The hyperparameters of SpecOffload are set for developer's machine. You may need to adjust them according to your machine's configuration.