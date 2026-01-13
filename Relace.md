Timur Khayrullin for Relace 2026

Task: first train a 300M parameter model using OLMo-core, then integrate an MOE.

Idea: follow this guide: https://olmo-core.readthedocs.io/en/latest/guides/all_in_one_for_researchers.html 

downloading minimal shards of c4 datasets:

curl -O http://olmo-data.org/examples/c4-en/gpt2/c4-train.00000-00099.npy
curl -O http://olmo-data.org/examples/c4-en/gpt2/c4-validation.00000-00008.npy


custom run:

torchrun --nproc-per-node=1 src/scripts/relace/train_300M_1B.py \
  custom-run-01 \
  --save-folder=/tmp/custom-run-01 \
  --work-dir=/tmp/dataset-cache \
  --trainer.hard_stop='{value: 100, unit: steps}'

Observations:
- got above run to work, wandb connects and loss goes down
- 

command for running src/scripts/relace/train_llama2_1B_15B.py, changed the model to be llama2_1B and disabled lmeval, downstream eval and reduced sequence length to 4 sequences. Also enabled activation checkpoints:

torchrun --nproc-per-node=1 src/scripts/relace/train_llama2_1B_15B.py llama2_1B --save-folder=/tmp/llama2_1B --work-dir=/tmp/dataset-cache --trainer.hard_stop='{value: 100, unit: steps}' --train_module.ac_config='{mode: budget, activation_memory_budget: 0.9}'

observations:
- uses 28 out of 32 gb of VRAM

command for running src/scripts/relace/train_smallmoe.py. enabled wandb, removed pipeline parallelism:

torchrun --nproc-per-node=1 src/scripts/relace/train_smallmoe.py smallmoe --trainer.hard_stop='{value: 100, unit: steps}' --train_module.ac_config='{mode: budget, activation_memory_budget: 0.9}'

Observations:
- uses 20gb vram for a 400M param moe.