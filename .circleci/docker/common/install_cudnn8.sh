#!/bin/bash

# cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
mkdir tmp_cudnn && cd tmp_cudnn
wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz -O cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz
tar xf cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz
cp -a cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive/include/* /usr/local/cuda/include/
cp -a cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive/lib/* /usr/local/cuda/lib64/
cd ..
rm -rf tmp_cudnn
ldconfig
