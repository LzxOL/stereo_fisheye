"""按需记录实际 shape / dtype；只保存元信息，不持有训练图中的特征体。"""

import torch


def tensor_info(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                    bytes=value.numel() * value.element_size())
    if isinstance(value, (list, tuple)):
        return [tensor_info(v) for v in value]
    if isinstance(value, dict):
        return {k: tensor_info(v) for k, v in value.items()}
    return type(value).__name__


class ShapeDebug:
    def __init__(self, model):
        self.records = {}
        self.handles = []
        def capture(name):
            def hook(module, args, output):
                self.records.setdefault(name, []).append(dict(input=tensor_info(args), output=tensor_info(output)))
            return hook
        modules = dict(cnn_left=model.encoder_left, cnn_right=model.encoder_right,
                       gwc_and_attention=model.volume_gen.att_branch, compress_left=model.volume_gen.ref_compress,
                       compress_right=model.volume_gen.tgt_compress, acv_and_gev=model.volume_gen.cost_agg,
                       classifier=model.volume_gen.classifier, context=model.context_net,
                       lookup_and_motion=model.update_block.encoder, gru=model.update_block.gru,
                       delta=model.update_block.depth_head)
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(capture(name)))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def record_output(self, output):
        self.records["network"] = tensor_info(output)
