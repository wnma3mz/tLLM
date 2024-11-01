export PYTHONPATH="./":$PYTHONPATH;

python3 tllm/web/app.py --port 7860 --chat_url "localhost:8022"