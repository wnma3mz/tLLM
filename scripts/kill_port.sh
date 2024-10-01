#!/bin/bash
for port in $@
do
    kill $(lsof -t -i:$port) || kill -9 $(lsof -t -i:$port)
done