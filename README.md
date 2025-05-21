ParetoQ Guide
===
This fork of ParetoQ is to test ParetoQ 2bit QAT then cook 2bpw quants with "better quality".

## Goal
Test this on `Llama-3.2-1B-Instruct` LLM to see if it works before
fussing with changing it for other architectures and larger models.


## Installation
```bash
# Clone
#git clone https://github.com/facebookresearch/ParetoQ.git
git clone https://github.com/ubergarm/ParetoQ.git

cd ParetoQ

# Install Python and Dependencies with uv for repeatability
# https://docs.astral.sh/uv/getting-started/installation/
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ./venv --python 3.11 --python-preference=only-managed
source ./venv/bin/activate
uv pip install -r requirements.txt
uv pip install huggingface-hub[hf-xet]
uv pip install tensorboard
```

## Prepare Base Model
Choose a base model on which you would like to perform ParetoQ Quantization Aware Training (QAT).
```bash
# I like to manually download models and use full paths locally
# If its not llama-3.2 or similar architecture you'll have to edit the train code.
# Also note it has to have been saved with `save_pretrained` apparently:
[rank0]: OSError: The safetensors archive passed at
    /mnt/astrodata/llm/models/theo77186/Llama-3.2-8B-Instruct/model-00001-of-00005.safetensors
    does not contain the valid metadata. Make sure you save your model with
    the `save_pretrained` method.

# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
cd /mnt/astrodata/llm/models/meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download --local-dir ./ meta-llama/Llama-3.2-1B-Instruct
```

## Prepare Training Data
```
# let it download a few hundred MiB or so and then hit <cntrl>+c to break
wget https://huggingface.co/datasets/AlppAI/SlimPajama-chunked/resolve/main/chunk1/chunk1.jsonl
# then remove the final line so it will have valid JSONL entries
vi chunk1.jsonl
# G   (capitol G to go to last line)
# dd  (delete last line as it probably got truncated and is now malformed JSON)
# :wq (write save and exit)
```

## Perform ParetoQ QAT
```
cd ParetoQ

./run_train.sh

tensorboard --logdir ./checkpoints/runs/current/
# Open Browser to:
# http://localhost:6006/?darkMode=true
```

## References
* [Original ParetoQ Discussion](https://github.com/facebookresearch/ParetoQ/issues/10#issuecomment-2898599130)
