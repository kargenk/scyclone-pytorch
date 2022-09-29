# dockerイメージのビルド
docker build ./ --tag $(whoami)_svc:latest

# dockerコンテナの立ち上げ
docker run --name kg_svc \
    --gpus all \
    -p 6006:6006 \
    -p 8888:8888 \
    -v /home/$(whoami)/scyclone-pytorch:/workspace/scyclone-pytorch \
    -v /home/$(whoami)/datasets:/datasets \
    -it -d --shm-size=32gb $(whoami)_svc:latest /bin/bash
