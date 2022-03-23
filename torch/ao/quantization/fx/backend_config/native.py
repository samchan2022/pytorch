import torch
from .observation_type import ObservationType
import torch.nn.qat as nnqat
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.qat as nniqat

from ...fuser_method_mappings import reverse_sequential_wrapper2

def get_native_backend_config_dict():
    """ Get backend for PyTorch Native backend_config_dict (fbgemm/qnnpack)
    """
    # dtype configs

    # weighted op int8 config
    # activation: quint8, weight: qint8, bias: float
    weighted_op_int8_dtype_config = {
        # optional, input activation dtype
        "input_dtype": torch.quint8,
        # optional, weight dtype
        "weight_dtype": torch.qint8,
        # optional, bias dtype
        "bias_dtype": torch.float,
        # optional, output activation dtype
        "output_dtype": torch.quint8
    }
    # operator (module/functional/torch ops) configs
    linear_module_config = {
        # Please see README under this folder for pattern format
        "pattern": torch.nn.Linear,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        # the root module for the pattern, used to query the reference quantized module
        # e.g. for a (torch.nn.ReLU, torch.nn.Linear) pattern, the root will be torch.nn.Linear
        "root_module": torch.nn.Linear,
        # the corresponding reference quantized module for the root module
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
        "qat_module": nnqat.Linear,
    }
    linear_qat_config = {
        "pattern": nnqat.Linear,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
    }
    linear_functional_config = {
        "pattern": torch.nn.functional.linear,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    linear_relu_fused_config = {
        "pattern": nni.LinearReLU,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
        "qat_module": nniqat.LinearReLU,
    }
    linear_relu_qat_config = {
        "pattern": nniqat.LinearReLU,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
    }
    linear_relu_mm_config = {
        "pattern": (torch.nn.ReLU, torch.nn.Linear),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "fuser_method": reverse_sequential_wrapper2(nni.LinearReLU),
    }
    linear_relu_mf_config = {
        "pattern": (torch.nn.functional.relu, torch.nn.Linear),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "fuser_method": reverse_sequential_wrapper2(nni.LinearReLU),
    }
    linear_relu_fm_config = {
        "pattern": (torch.nn.ReLU, torch.nn.functional.linear),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    linear_relu_ff_config = {
        "pattern": (torch.nn.functional.relu, torch.nn.functional.linear),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
    }
    linear_bn_fused_config = {
        "pattern": nni.LinearBn1d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
        "qat_module": nniqat.LinearBn1d,
    }
    linear_bn_qat_config = {
        "pattern": nniqat.LinearBn1d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
    }

    return {
        # optional
        "name": "native",
        "configs": [
            linear_module_config,
            linear_qat_config,
            linear_functional_config,
            linear_relu_fused_config,
            linear_relu_qat_config,
            linear_relu_mm_config,
            linear_relu_mf_config,
            #linear_relu_fm_config,
            #linear_relu_ff_config,
            linear_bn_fused_config,
            linear_bn_qat_config,
        ],
    }
