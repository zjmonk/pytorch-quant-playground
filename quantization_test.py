import copy
import torch
import torch.nn as nn
from torch.quantization import (
    QConfig,
    FakeQuantize,
    MovingAverageMinMaxObserver,
    get_default_qat_module_mappings,
    propagate_qconfig_,
    convert,
    prepare,
    default_per_channel_weight_fake_quant,
    default_weight_fake_quant,
)


my_simple_qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
        ),
        weight=default_weight_fake_quant,
    )


my_conv_qconfig = torch.quantization.get_default_qat_qconfig()


def prepare_qat(model, mapping=None, inplace=False):
    r"""
    Prepares a copy of the model for quantization calibration or
    quantization-aware training and converts it to quantized version.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    Args:
        model: input model to be modified in-place
        mapping: dictionary that maps float modules to quantized modules to be
                 replaced.
        inplace: carry out model transformations in-place, the original module
                 is mutated
    """
    torch._C._log_api_usage_once("quantization_api.quantize.prepare_qat")
    if mapping is None:
        mapping = get_default_qat_module_mappings()

    if not inplace:
        model = copy.deepcopy(model)
    qdict = {
        torch.nn.modules.conv.ConvTranspose2d: my_simple_qconfig
    }
    propagate_qconfig_(model, qconfig_dict=qdict)
    convert(model, mapping=mapping, inplace=True, remove_qconfig=False)
    prepare(model, observer_non_leaf_module_list=set(mapping.values()), inplace=True)
    return model


class simple_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1)
        self.deconv = nn.ConvTranspose2d(12, 3, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.conv(x)
        out = self.deconv(out)
        return out


def quantization_test(model: nn.Module, input_tensor: torch.Tensor):
    model.qconfig = my_conv_qconfig
    prepare_qat(model, inplace=True)
    print(model)
    model(input_tensor)
    model.eval()
    model_quant = torch.quantization.convert(model)
    print(model_quant)
    print(model)


if __name__ == "__main__":
    print("Begin playing")
    test_input = torch.rand((1, 3, 128, 128))
    my_model = simple_model()
    output = my_model(test_input)
    print(type(my_model.deconv))
    print(output.shape)
    quantization_test(my_model, test_input)
