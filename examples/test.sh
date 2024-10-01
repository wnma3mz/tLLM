curl http://localhost:8000/v1/chat/completions -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer common" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ]
  }'
