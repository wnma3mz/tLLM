ports=(50051 50052)
for port in "${ports[@]}"
do
    echo "Killing processes using port $port..."
    lsof -ti tcp:$port | xargs kill -9
    echo "start processes using port $port..."
    python src/rpc_comm/client.py --port $port > "rpc-server-$port.log" 2>&1 &
done

sleep 1
echo $PYTHONPATH
export PYTHONPATH="./src":$PYTHONPATH
python src/app.py --config-path examples/rpc_pp2_llama3_config.json --comm rpc
