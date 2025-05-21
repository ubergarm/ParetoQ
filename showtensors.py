import torch
from models.modeling_llama_quant import (
      LlamaForCausalLM as LlamaForCausalLMQuant,
  )


#model_path = "/mnt/astrodata/llm/models/meta-llama/Llama-3.2-1B-Instruct/"
model_path = "/mnt/astrodata/llm/models/ubergarm/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-ParetoQ-2bpw/"
model = LlamaForCausalLMQuant.from_pretrained(
      pretrained_model_name_or_path=model_path,
      low_cpu_mem_usage=True,
      device_map='cpu',
      )

w_bits = 2
for name, param in model.named_parameters():
  if "weight_clip_val" in name:
      print(f"name: {name} param: {param}")
      weight_name = name.replace("weight_clip_val", "weight")
      weight_param = dict(model.named_parameters()).get(weight_name, None)

      if w_bits == 1:
          scale = torch.mean(weight_param.abs(), dim=-1, keepdim=True).detach()
      elif w_bits == 0 or w_bits == 2:
          scale, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
      elif w_bits == 3 or w_bits == 4:
          xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
          maxq = 2 ** (w_bits - 1) - 1
          scale = xmax / maxq
      else:
          raise NotImplementedError

      param.data.copy_(scale)


print(model)
