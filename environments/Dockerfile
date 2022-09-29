# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-08.html#rel_22-08
FROM nvcr.io/nvidia/pytorch:22.08-py3

# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 必要なパッケージのインストール
RUN apt-get update
RUN apt-get install -y screen ffmpeg

# ライブラリのインストール
# RUN pip install torchaudio
RUN pip install git+https://github.com/openai/whisper.git