#!/bin/sh
# Box setup fix: PaddlePaddle pins older CUDA wheels (nccl, cudnn, cusparselt) that break torch.
# Keep Paddle in its own venv; ocr_engines.SubprocessEngine reaches it via REDROB_PADDLE_PY.
set -eux
T=$HOME/redrob-image-venv
P=$HOME/paddle-venv
[ -x "$P/bin/python" ] || python3 -m venv "$P"
"$P/bin/pip" install -q -U pip wheel
"$P/bin/pip" install -q paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
"$P/bin/pip" install -q paddleocr==3.7.0 pillow numpy
"$P/bin/python" -c "import paddle,paddleocr;print('paddle', paddle.device.is_compiled_with_cuda())"
"$T/bin/pip" uninstall -y -q paddlepaddle-gpu paddleocr paddlex || true
"$T/bin/pip" install -q nvidia-nccl-cu12==2.28.9 nvidia-cudnn-cu12==9.19.0.56 nvidia-cusparselt-cu12==0.7.1 fonttools==4.60.1
"$T/bin/python" -c "import torch;print('torch', torch.cuda.is_available(), torch.backends.cudnn.version())"
