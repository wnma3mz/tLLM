export PYTHONPATH="./":$PYTHONPATH;

python3 tllm/web/app.py --port 7860 --chat_url "http://localhost:8022/v1/chat/completions"