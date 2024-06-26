ports=(8000 8001)
for port in "${ports[@]}"
do
    echo "Killing processes using port $port..."
    lsof -ti tcp:$port | xargs kill -9
    echo "start processes using port $port..."
    python src/communication/client.py --port $port > "server-$port.log" 2>&1 &
done

echo $PYTHONPATH
export PYTHONPATH="./src":$PYTHONPATH
python src/app.py --config-path examples/pp2_config.json