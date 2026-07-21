# ternary_QAT

Lightweight ternary QAT for [Ternary-Bonsai](https://huggingface.co/models?search=unpacked%20ternary) unpacked text models.

Qwen 3.5 is untested right now!

This is designed from the start to be Unsloth compatible.

Weights are ternarized to `{-1, 0, 1}` per group of (128/64/user-defined) consecutive weights along the last dim, matching the Bonsai on-disk format (verified bit-exact). Embeddings + all `nn.Linear` modules (attn, MLP, lm_head) are ternarized; norms stay FP.

Based on Prism-ML's whitepaper: https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf

## Install

Have your preferred torch version installed first so that this doesn't install the CPU version (which you probably don't want)

```bash
pip install ternary_QAT                 # core (torch only)
pip install "ternary_QAT[peft,transformers]"  # + LoRA / model loading

if you use uv:
uv pip install ternary_QAT --torch-backend=auto
```

## Use

### Full-finetune

```python
from transformers import AutoModelForCausalLM
from ternary import swap_linear, TernaryConfig

model = AutoModelForCausalLM.from_pretrained("prism-ml/Ternary-Bonsai-1.7B-unpacked")
swap_linear(model, TernaryConfig(group_size=128))
# ... train normally; ternarize fires in every Linear.forward
```

### LoRA (ternary frozen base + FP adapters)

```python
from peft import LoraConfig, get_peft_model
from ternary import swap_linear, TernaryConfig, ternarize_lora_params, reternarize_merged_linears

model = ...  # load model
swap_linear(model, TernaryConfig(group_size=128))
model = get_peft_model(model, LoraConfig(r=128, lora_alpha=128, ...))

# ... train ...

# at save: ternarize adapter, merge, re-ternarize merged linears
ternarize_lora_params(model)
model = model.merge_and_unload()
reternarize_merged_linears(model)
model.save_pretrained("./out")
```

See [`examples/`](examples/) for LoRA, FFT, and Unsloth examples.

## Learning rate

Ternary QAT needs **10-50x higher LR** than standard fine tuning.

The lowest usable LR I've found so far is around 7e-4, so experiment in that range up to the e-3s, depending on rank and dataset size.