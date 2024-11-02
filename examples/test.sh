for i in {1..10}; do
  curl http://localhost:8022/v1/chat/completions -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer common" \
    -d '{
      "model": "tt",
      "stream": true,
      "messages": [
        {
          "role": "user",
          "content": "Hello, how are you?"
        }
      ]
    }'
  echo ""
done
