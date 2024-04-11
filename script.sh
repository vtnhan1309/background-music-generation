docker run --gpus all -d -it --name=audio-test \
    -v /home/nhanvt2/stable-diffusion/zaloai_submission:/zaloai_submission \
    audio:1.0

docker exec -it audio-test bash
