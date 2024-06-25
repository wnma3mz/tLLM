ports=(8000 8001 8002 8003)

# 循环遍历端口数组
for port in "${ports[@]}"
do
    echo "Killing processes using port $port..."
    lsof -ti tcp:$port | xargs kill -9
done