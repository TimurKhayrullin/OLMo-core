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
- omlo's smallmoe arch uses 488,163,072 total params.

day 2 idea:

need to get a few dense LLMs to saturate their loss curves, and then find corresponding MoEs to train.

the openai paper lists models with as little as <1M params, so this 271M llama2 should be good. 

torchrun --nproc-per-node=1 src/scripts/relace/saturate_271M_15B.py \
  saturate_271M_15B \
  --save-folder=/tmp/saturate_271M_15B \
  --work-dir=/tmp/dataset-cache \
  --trainer.hard_stop='{value: 15000, unit: steps}'

this saturated at 30 steps(!) I let it train for 500, then I stopped it. Now doubling d_model and seeing how many params that would be.

torchrun --nproc-per-node=1 src/scripts/relace/saturate_2x271M_15B.py \
  saturate_2x271M_15B \
  --save-folder=/tmp/saturate_2x271M_15B \
  --work-dir=/tmp/dataset-cache \
  --trainer.hard_stop='{value: 500, unit: steps}'

observations: 
- had to reduce micro batch size to 4 sequences instead of 8, params are:
- 1,028,196,352 total params                                                                                                                                                                               
- 925,173,760 non-embedding params                                                                                                                                                                         
- 1,028,196,352 trainable params

run for llama2_135M:

torchrun --nproc-per-node=1 src/scripts/relace/saturate_135M_15B.py \
  saturate_135M_15B.py \
  --save-folder=/tmp/saturate_135M_15B.py \
  --work-dir=/tmp/dataset-cache \
  --trainer.hard_stop='{value: 1000, unit: steps}'

run for llama2_1B:

torchrun --nproc-per-node=1 src/scripts/relace/saturate_1B_15B.py \
  saturate_1B_15B.py \
  --save-folder=/tmp/saturate_1B_15B \
  --work-dir=/tmp/dataset-cache \
  --trainer.hard_stop='{value: 2000, unit: steps}'


run for moe1Bolmo:

torchrun --nproc-per-node=1 src/scripts/relace/saturate_moe1Bolmo_15B.py \
  saturate_moe1Bolmo_15B \
  --save-folder=/tmp/saturate_moe1Bolmo_15B \
  --work-dir=/tmp/dataset-cache \
  --trainer.hard_stop='{value: 2000, unit: steps}'

run for llama2_60M:

torchrun --nproc-per-node=1 src/scripts/relace/saturate_60M_15B.py \
  saturate_60M_15B \
  --save-folder=/tmp/saturate_60M_15B \
  --work-dir=/tmp/dataset-cache \
  --trainer.hard_stop='{value: 3000, unit: steps}'


run for llama2_30M:

torchrun --nproc-per-node=1 src/scripts/relace/saturate_30M_15B.py \
  saturate_30M_15B \
  --save-folder=/tmp/saturate_30M_15B \
  --work-dir=/tmp/dataset-cache \
  --trainer.hard_stop='{value: 3000, unit: steps}'

run for llama2_1M:

torchrun --nproc-per-node=1 src/scripts/relace/saturate_1M_15B.py \
  saturate_1M_15B \
  --save-folder=/tmp/saturate_1M_15B \
  --work-dir=/tmp/dataset-cache \
  --trainer.hard_stop='{value: 3000, unit: steps}'

run for llama2_100K:

torchrun --nproc-per-node=1 src/scripts/relace/saturate_100K_15B.py \
  saturate_100K_15B \
  --save-folder=/tmp/saturate_100K_15B \
  --work-dir=/tmp/dataset-cache \
  --trainer.hard_stop='{value: 3000, unit: steps}'

run for llama2_10K:

torchrun --nproc-per-node=1 src/scripts/relace/saturate_10K_15B.py \
  saturate_10K_15B \
  --save-folder=/tmp/saturate_10K_15B \
  --work-dir=/tmp/dataset-cache \
  --trainer.hard_stop='{value: 3000, unit: steps}'

