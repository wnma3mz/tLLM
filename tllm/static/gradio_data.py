from dataclasses import dataclass


@dataclass
class GenerationConfig:
    chat_url: str = "http://localhost:8000/v1/chat/completions"
    system_prompt: str = "You are a helpful AI assistant."
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 1024


# 自定义CSS样式
custom_css = """
footer{display:none !important}

.button-primary {
    background: #2563eb !important;
    border: none !important;
    color: white !important;
    padding: 10px 20px !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.button-primary:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px);
}

.button-secondary {
    background: #f3f4f6 !important;
    border: 1px solid #e5e7eb !important;
    color: #374151 !important;
    padding: 10px 20px !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.button-secondary:hover {
    background: #e5e7eb !important;
}

.message-input {
    border: 1px solid #e5e7eb !important;
    border-radius: 6px !important;
    padding: 12px !important;
    font-size: 16px !important;
}
"""
