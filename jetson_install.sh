pip install --no-cache-dir "onnx>=1.12.0,<=1.17.0" "onnxslim>=0.1.71"
wget https://nvidia.box.com/shared/static/mvdcltm9ewdy2d5nurkiqorofz1s53ww.whl -O onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl
pip install onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl

pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# Install dependencies
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev # ros-foxy-pcl-ros

# Clone torchvision v0.16.2 (compatible with PyTorch 2.1)
git clone --branch v0.16.2 https://github.com/pytorch/vision torchvision
cd torchvision

# Build and install
export BUILD_VERSION=0.16.2
export TORCH_CUDA_ARCH_LIST="7.2;8.7"
python3 setup.py install --user

# Move out of the build directory
cd ../