from typing import *

from PIL import Image
import torch
from transformers import AutoProcessor, LlamaForCausalLM, Qwen2ForCausalLM, Qwen2VLForConditionalGeneration

from tllm.generate.token_utils import TokenizerUtils


def load_llama(model_path: str):
    return LlamaForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="cpu", torch_dtype=torch.bfloat16
    )


def load_qwen2(model_path: str):
    return Qwen2ForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="cpu", torch_dtype=torch.bfloat16
    )


def load_qwen2_vl(model_path: str):
    return Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, trust_remote_code=True, device_map="cpu", torch_dtype=torch.bfloat16
    )


def load_text_message():
    return [{"role": "user", "content": "Hello, how are you?"}]


def load_vl_message():
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]


if __name__ == "__main__":
    model_path = "/Users/jianghulu/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
    model_path = "/Users/jianghulu/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/aca78372505e6cb469c4fa6a35c60265b00ff5a4"
    tok = TokenizerUtils(model_path)
    model = load_qwen2_vl(model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    messages = load_vl_message()
    # input_id_list = tok.preprocess(messages=messages).input_ids
    # input_ids = torch.tensor(input_id_list).unsqueeze(0)
    # print("input_ids: ", input_ids)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[Image.open("asserts/image-1.png")], return_tensors="pt")
    print("text: ", text)

    model.eval()

    with torch.no_grad():
        # output = model.generate(**inputs, max_length=30)
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    # print("generate token: ", output[0][len(input_id_list):])
    # print("generate text:", "".join(tok.decode(output[0][len(input_id_list):])))

    print("generate token: ", output[0][len(inputs.input_ids[0]) :])
    print("generate text:", "".join(tok.decode(output[0][len(inputs.input_ids[0]) :])))
