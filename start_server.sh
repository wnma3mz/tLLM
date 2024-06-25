ports=(8000 8001 8002 8003)

for port in "${ports[@]}"
do
    echo "start processes using port $port..."
    python src/communication/client.py --port $port > "server-$port.log" 2>&1 &
done