import tensorflow as tf
import torch

print(tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
))

print(tf.test.is_built_with_cuda())

print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())