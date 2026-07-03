from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
import torch
from typing import Optional

MODEL_ID = "Qwen/Qwen3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

def qwen( prompt:str,
          system:Optional[str]=None,
          max_new_tokens:int=256,
          temperature:float=0.0,
          top_p:float=1.0,
          enable_thinking:bool=False,
          do_sample:bool=False,
          stream:bool=False) -> str:

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # ID de tokens (respuesta)
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
    # Traducimos los Ids como texto
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        _ = model.generate(**inputs, streamer=streamer, **gen_kwargs)
        return ""

    with torch.no_grad():
        # resp = [input, output]
        out = model.generate(**inputs, **gen_kwargs)
    # Sólo la parte nueva:
    gen_ids = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)
