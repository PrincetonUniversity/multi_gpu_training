# Training with model parallelism via FSDP
## Background
See `slides.pdf`.

## Experiment
We are finetuning a [ShearLlama-2.7b](https://huggingface.co/princeton-nlp/Sheared-LLaMA-2.7B) model on a [chess move dataset](https://huggingface.co/datasets/laion/strategic_game_chess).
1. Start by downloading the models: `python download_models.py`
2. Look at `download_models.py`
3. Execute `chess_finetune.sh` or submit to slurm