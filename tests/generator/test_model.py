import mlx.core as mx

from tllm.commons.tp_communicator import BaseCommunicator
from tllm.commons.weight_manager import load_client_model, load_master_model
from tllm.models.mlx.qwen3 import MLXQwen3ForCausalLM, MLXQwen3Model
from tllm.schemas import SeqInput
from tllm.singleton_logger import SingletonLogger

logger = SingletonLogger.setup_master_logger()


def load_messages():
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "今天天气怎么样"},
    ]
    return messages


if __name__ == "__main__":
    model_path: str = "mlx-community/Qwen3-0.6B-4bit"
    comm = BaseCommunicator(logger)
    client_model = load_client_model(0, 999, comm, model_path)
    master_model = load_master_model(model_path)

    messages = load_messages()

    tok_result = master_model.tok.preprocess(messages=messages, force_prompt="<think>\n</think>\n\n")
    print(f"tok_result: {tok_result}")
    seq_input = SeqInput(uuid_list=["test"], input_ids_list=[tok_result.input_ids])
    input_embeddings = master_model.get_input_embeddings(tok_result.input_ids)
    hidden_states = client_model(input_embeddings, seq_input)
    logits = master_model.get_logits(hidden_states)

    each_token_list = mx.argmax(logits, axis=-1)
    last_token = each_token_list[-1].tolist()
    print("last_token: ", last_token)
    last_text = master_model.tok.decode([last_token], [None])
    print(f"last token text: {last_text}")
