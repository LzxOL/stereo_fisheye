"""验证双目代价体、Context Network 和多轮 GRU 的完整前向/反向。"""

import unittest
import torch
import torch.nn as nn

from module_baseline import StereoBaseline
from module_baseline.mrt_layers import CNNEncoder
from module_baseline.context import ContextNetwork
from module_baseline.loss_functions import sequence_loss
from module.mrt_layers import CNNEncoder as OriginalCNNEncoder
from module.network import PDFOmni
from module.volume_generator import Generator as OriginalGenerator


def make_grids(batch=1, height=8, width=16, depth=16, device="cpu"):
    y, x = torch.meshgrid(torch.linspace(-.8, .8, height, device=device),
                          torch.linspace(-.8, .8, width, device=device), indexing="ij")
    left = torch.stack([x, y], dim=-1)[None].expand(batch, -1, -1, -1).clone()
    right = left.unsqueeze(3).expand(-1, -1, -1, depth, -1).clone()
    right[..., 0] += torch.linspace(-.05, .05, depth, device=device)
    return [left, right]


def small_model():
    # 保留改造前的 joint 回归用例；新 split / 正式尺寸验证按用户要求稍后补做。
    return StereoBaseline(num_depth=16, sweep_size=(8, 16), volume_update_mode="joint")


class SelectFirst(nn.Module):
    def forward(self, x):
        # 仅用于原版对照：让前/后融合精确选第一张，从原版入口得到同一对参考/目标体。
        return torch.ones_like(x[:, :1])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(7)

    def test_copied_cnn_matches_original_both_scales(self):
        original = OriginalCNNEncoder(output_dim=32)
        copied = CNNEncoder(output_dim=32)
        copied.load_state_dict(original.state_dict())
        image = torch.randn(2, 3, 31, 35)
        for actual, expected in zip(copied(image), original(image)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_sweeping_matches_original_for_each_sample(self):
        model = small_model()
        features = [torch.randn(2, 32, 5, 7), torch.randn(2, 32, 6, 8)]
        grids = make_grids(batch=2)
        grids[1][1, ..., 1] += .02
        volumes, valid = model.spherical_sweep(features, grids)
        self.assertTrue(valid.all())
        for i in range(2):
            original_grids = [grids[0][i].unsqueeze(2).expand(-1, -1, 16, -1), grids[1][i]]
            expected = PDFOmni.spherical_sweep(None, [f[i:i+1] for f in features], original_grids)
            for actual, reference in zip(volumes, expected[:2]):
                torch.testing.assert_close(actual[i:i+1], reference, rtol=0, atol=0)

    def test_sweeping_masks_invalid_candidates_but_keeps_left_context(self):
        model = small_model()
        features = [torch.ones(1, 32, 5, 7), torch.ones(1, 32, 6, 8)]
        grids = make_grids()
        grids[0][:, 0, 0, 0] = float("nan")
        grids[1][:, 1, 1, 0, 0] = float("inf")
        grids[1][:, 1, 1, 1, 0] = 1.01
        mask = torch.ones(1, 1, 16, 8, 16, dtype=torch.bool)
        mask[:, :, 2, 1, 1] = False
        volumes, valid = model.spherical_sweep(features, grids, mask)
        self.assertFalse(valid[:, :, :, 0, 0].any())
        self.assertFalse(valid[:, :, :3, 1, 1].any())
        self.assertEqual(volumes[0][:, :, 0, 0].count_nonzero(), 0)
        self.assertEqual(volumes[1][:, :, 1, 1, :2].count_nonzero(), 0)
        # 右侧候选 0 不可见，左上下文仍应保留；不能被联合匹配 mask 清空。
        torch.testing.assert_close(volumes[0][:, :, 1, 1, 0], torch.ones(1, 32))
        for volume in volumes:
            self.assertTrue(torch.isfinite(volume).all())

    def test_generator_matches_original_after_four_camera_fusion(self):
        model = small_model().eval()
        original = OriginalGenerator(model.opts).eval()
        missing = original.load_state_dict(model.volume_gen.state_dict(), strict=False)
        self.assertFalse(missing.unexpected_keys)
        self.assertTrue(all(k.startswith(("reference_mapping.", "target_mapping.")) for k in missing.missing_keys))
        original.reference_mapping = SelectFirst()
        original.target_mapping = SelectFirst()
        left = torch.randn(1, 32, 8, 16, 16)
        right = torch.randn_like(left)
        coords = torch.zeros(1, 2, 8, 16, 16)
        with torch.no_grad():
            actual = model.volume_gen([left, right])
            expected = original([left, right, left, right, coords, coords, coords, coords])
        # 同权重、同 Reference/Target，比较完整 GWC / ACV / OT 后的结果及 VE/VC 查表。
        for a, b in zip(actual[1:], expected[1:]):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual[0](actual[1]), expected[0](expected[1]), rtol=2e-5, atol=2e-6)

    def test_uniform_classifier_regresses_candidate_index(self):
        model = small_model().eval()
        nn.init.zeros_(model.volume_gen.classifier.weight)
        features = [torch.randn(1, 32, 8, 16, 16) for _ in range(2)]
        with torch.no_grad():
            _, disp, _, prob = model.volume_gen(features)
        torch.testing.assert_close(prob, torch.full_like(prob, 1/16))
        torch.testing.assert_close(disp, torch.full_like(disp, 7.5))

    def test_ot_masks_empty_rows_pixels_and_entire_sample_with_finite_gradients(self):
        generator = small_model().volume_gen
        logits = torch.randn(2, 16, 8, 16, requires_grad=True)
        valid = torch.ones_like(logits, dtype=torch.bool)
        valid[0, 0] = False
        valid[0, :, 0, 0] = False
        valid[0, :8, 1, 1] = False
        valid[1] = False
        mass = generator._sinkhorn(logits, 3, valid).exp()
        prob = mass / (mass.sum(dim=1, keepdim=True) + 1e-9)
        self.assertTrue(torch.isfinite(prob).all())
        self.assertEqual(prob[~valid].count_nonzero(), 0)
        torch.testing.assert_close(prob.sum(1)[valid.any(1)], torch.ones_like(prob.sum(1)[valid.any(1)]))
        prob.square().sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[~valid].count_nonzero(), 0)

    def test_context_heads_preserve_resolution_and_have_independent_gradients(self):
        context_net = ContextNetwork()
        reference = torch.randn(2, 32, 8, 16, requires_grad=True)
        hidden, context = context_net(reference)
        self.assertEqual(hidden.shape, (2, 64, 8, 16))
        self.assertEqual(context.shape, (2, 32, 8, 16))
        self.assertTrue((hidden.abs() <= 1).all())
        self.assertTrue((context >= 0).all())
        self.assertIsNot(context_net.hidden_head.weight, context_net.context_head.weight)
        (hidden.square().mean() + context.square().mean()).backward()
        for parameter in context_net.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(reference.grad.abs().sum().item(), 0)

    def test_forward_and_backward_through_fixed_context_and_recurrent_hidden(self):
        model = small_model().train()
        seen = {"contexts": [], "hidden_states": []}
        handle = model.volume_gen.register_forward_hook(lambda m, args, out: seen.update(volume=out))
        def capture_context(module, args, out):
            seen["context_input"] = args[0]
            seen["hidden_init"], seen["context_feature"] = out
            for feature in out:
                feature.retain_grad()
        def capture_update(module, args, out):
            seen["contexts"].append(args[1])
            out[0].retain_grad()
            seen["hidden_states"].append(out[0])
        context_handle = model.context_net.register_forward_hook(capture_context)
        update_handle = model.update_block.register_forward_hook(capture_update)
        valid = torch.ones(2, 1, 16, 8, 16, dtype=torch.bool)
        valid[0, :, :8, 1, 1] = False
        valid[1] = False
        left = torch.randn(2, 3, 64, 64, requires_grad=True)
        right = torch.randn(2, 3, 64, 64, requires_grad=True)
        result = model([left, right], make_grids(batch=2), valid, iters=3)
        for hook in (handle, context_handle, update_handle):
            hook.remove()
        _, disp, context, prob = seen["volume"]
        torch.testing.assert_close(seen["context_input"], context[..., 0])
        self.assertEqual(len(result["disp_predictions"]), 3)
        self.assertIs(result["disp"], result["disp_predictions"][-1])
        torch.testing.assert_close(result["valid_mask"], valid.any(2))
        for prediction in [result["init_disp"], *result["disp_predictions"]]:
            self.assertEqual(prediction.shape, (2, 1, 8, 16))
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertTrue(((prediction >= 0) & (prediction <= 15)).all())
            self.assertEqual(prediction[~result["valid_mask"]].count_nonzero(), 0)
        # 同一份 Context 进入全部三轮；Hidden State 保留跨轮梯度。
        self.assertTrue(all(feature is seen["context_feature"] for feature in seen["contexts"]))
        self.assertEqual(disp[1].count_nonzero(), 0)
        self.assertEqual(prob.masked_select(~valid[:, 0]).count_nonzero(), 0)
        target = torch.full_like(disp, 5.0)
        # 初值与各轮预测仅用数值回归监督；不向损失传入 init_prob。
        loss = sequence_loss([result["init_disp"], *result["disp_predictions"]], target, result["valid_mask"])
        loss.backward()
        for name, parameter in model.named_parameters():
            if ".conv2_down." in name or name.startswith("update_block.mask."):
                # 原 CNN 未使用的 1/4 分支和本阶段不启用的上采样掩码头。
                self.assertIsNone(parameter.grad)
            else:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for encoder in (model.encoder_left, model.encoder_right):
            self.assertGreater(encoder.conv0[0].weight.grad.abs().sum().item(), 0)
        for feature in [seen["hidden_init"], seen["context_feature"], *seen["hidden_states"]]:
            self.assertGreater(feature.grad.abs().sum().item(), 0)
        self.assertGreater(left.grad.abs().sum().item(), 0)
        self.assertGreater(right.grad.abs().sum().item(), 0)
        self.assertGreater(model.volume_gen.att_branch.classif_att[-1].conv.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.volume_gen.cost_agg.up1.conv.weight.grad.abs().sum().item(), 0)

    def test_iteration_uses_candidate_index_units_and_clamps_without_upsampling(self):
        model = small_model().eval()
        nn.init.zeros_(model.volume_gen.classifier.weight)
        nn.init.zeros_(model.update_block.depth_head.conv2.weight)
        nn.init.constant_(model.update_block.depth_head.conv2.bias, .25)
        images = [torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)]
        grids = make_grids()
        with torch.no_grad():
            result = model(images, grids, iters=3)
        torch.testing.assert_close(result["init_disp"], torch.full_like(result["init_disp"], 7.5))
        for i, prediction in enumerate(result["disp_predictions"]):
            torch.testing.assert_close(prediction, torch.full_like(prediction, 7.5 + (i+1)*.25))
        for delta, expected in [(100., 15.), (-100., 0.)]:
            nn.init.constant_(model.update_block.depth_head.conv2.bias, delta)
            with torch.no_grad():
                result = model(images, grids, iters=2)
            torch.testing.assert_close(result["disp"], torch.full_like(result["disp"], expected))
        with torch.no_grad():
            initial_only = model(images, grids, iters=0)
        self.assertEqual(initial_only["disp_predictions"], [])
        torch.testing.assert_close(initial_only["disp"], initial_only["init_disp"])

    def test_entirely_invalid_pipeline_has_zero_outputs_and_finite_backward(self):
        model = small_model().train()
        valid = torch.zeros(1, 1, 16, 8, 16, dtype=torch.bool)
        grids = make_grids()
        grids[0][:, 0, 0] = float("nan")
        grids[1][:, 1, 1] = float("inf")
        result = model([torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)], grids, valid, iters=2)
        for value in [result["init_disp"], result["init_prob"], *result["disp_predictions"]]:
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(value.count_nonzero(), 0)
        loss = sequence_loss([result["init_disp"], *result["disp_predictions"]],
                             torch.full_like(result["disp"], float("nan")), result["valid_mask"])
        self.assertEqual(loss.item(), 0)
        loss.backward()
        for name, parameter in model.named_parameters():
            if ".conv2_down." in name or name.startswith("update_block.mask."):
                continue
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_default_shapes_without_allocating_large_volumes(self):
        with torch.device("meta"):
            model = StereoBaseline(num_depth=96, sweep_size=(64, 128), volume_update_mode="joint").eval()
        seen = {}
        def capture(name):
            def hook(module, args, out):
                seen[name] = tuple(out.shape)
            return hook
        modules = {"left": model.encoder_left.conv2, "gwc_patch": model.volume_gen.att_branch.patch_all,
                   "attention": model.volume_gen.att_branch, "ref_c": model.volume_gen.ref_compress,
                   "tgt_c": model.volume_gen.tgt_compress, "gev": model.volume_gen.cost_agg,
                   "classifier": model.volume_gen.classifier,
                   "hidden_head": model.context_net.hidden_head, "context_head": model.context_net.context_head,
                   "motion": model.update_block.encoder, "gru": model.update_block.gru}
        handles = [m.register_forward_hook(capture(n)) for n, m in modules.items()]
        volume_handle = model.volume_gen.register_forward_hook(
            lambda m, args, out: seen.update(disp=tuple(out[1].shape), context=tuple(out[2].shape), prob=tuple(out[3].shape)))
        lookup_handle = model.update_block.encoder.register_forward_pre_hook(
            lambda m, args: seen.update(lookup=tuple(args[0].shape)))
        image = torch.empty(1, 3, 512, 512, device="meta")
        with torch.no_grad():
            result = model([image, image], make_grids(height=64, width=128, depth=96, device="meta"), iters=2)
        for handle in handles + [volume_handle, lookup_handle]:
            handle.remove()
        self.assertEqual(seen, {"left": (1, 32, 256, 256), "gwc_patch": (1, 8, 96, 64, 128),
                                "attention": (1, 1, 96, 64, 128), "ref_c": (1, 16, 96, 64, 128),
                                "tgt_c": (1, 16, 96, 64, 128), "gev": (1, 8, 96, 64, 128),
                                "classifier": (1, 1, 96, 64, 128), "disp": (1, 1, 64, 128),
                                "context": (1, 32, 64, 128, 96), "prob": (1, 96, 64, 128),
                                "hidden_head": (1, 64, 64, 128), "context_head": (1, 32, 64, 128),
                                "lookup": (1, 162, 64, 128), "motion": (1, 128, 64, 128), "gru": (1, 64, 64, 128)})
        self.assertEqual(result["disp"].shape, (1, 1, 64, 128))
        self.assertEqual(len(result["disp_predictions"]), 2)


if __name__ == "__main__":
    unittest.main()
