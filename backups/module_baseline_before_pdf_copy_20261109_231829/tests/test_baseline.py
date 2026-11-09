"""小尺寸真实数值检查 + 默认大尺寸的 meta 形状检查；无需数据或权重。"""

import unittest

import torch

from module_baseline import StereoBaseline
from module_baseline.encoder import CNNEncoder
from module_baseline.sweeping import SphericalSweeping
from module_baseline.volume import masked_softmax
from module.mrt_layers import CNNEncoder as OriginalCNNEncoder
from module.network import PDFOmni


def make_grids(height=8, width=16, num_depth=16, batch=1, device="cpu"):
    yy, xx = torch.meshgrid(
        torch.linspace(-0.8, 0.8, height, device=device),
        torch.linspace(-0.8, 0.8, width, device=device), indexing="ij")
    left = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1).clone()
    right = left.unsqueeze(3).expand(-1, -1, -1, num_depth, -1).clone()
    return left, right


def small_model():
    return StereoBaseline(num_depth=16, sweep_size=(8, 16), output_size=(16, 32))


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(7)

    def test_sampling_preserves_pixel_and_depth_axes(self):
        # 数字本身标识采样位置，能够发现 H/W/D 颠倒或左右取样错误。
        left_feature = torch.tensor([[[[1., 2.], [3., 4.]]]])
        right_feature = 10 * left_feature
        left_grid = torch.tensor([[[[-1., -1.], [1., 1.]]]])
        right_grid = torch.tensor([[[[[-1., -1.], [1., -1.]],
                                     [[1., 1.], [-1., 1.]]]]])
        left, right, valid = SphericalSweeping()(left_feature, right_feature, left_grid, right_grid)
        self.assertEqual(left.shape, (1, 1, 2, 1, 2))
        torch.testing.assert_close(left[0, 0, :, 0], torch.tensor([[1., 4.], [1., 4.]]))
        torch.testing.assert_close(right[0, 0, :, 0], torch.tensor([[10., 40.], [20., 30.]]))
        self.assertTrue(valid.all())

    def test_invalid_coordinates_and_explicit_mask_are_zero(self):
        left_grid, right_grid = make_grids()
        left_grid[:, 0, 0, 0] = float("nan")
        right_grid[:, 1, 1, 0, 0] = float("inf")
        right_grid[:, 1, 1, 1, 0] = 1.01
        mask = torch.ones(1, 1, 16, 8, 16, dtype=torch.bool)
        mask[:, :, 2, 1, 1] = False
        feature = torch.ones(1, 2, 4, 4)
        left, right, valid = SphericalSweeping()(feature, feature, left_grid, right_grid, mask)
        self.assertFalse(valid[:, :, :, 0, 0].any())
        self.assertFalse(valid[:, :, :3, 1, 1].any())
        self.assertTrue(torch.isfinite(left).all() and torch.isfinite(right).all())
        self.assertEqual(left.masked_select(~valid.expand_as(left)).count_nonzero(), 0)
        self.assertEqual(right.masked_select(~valid.expand_as(right)).count_nonzero(), 0)

    def test_sweeping_matches_original_pdf_omni_with_different_batch_grids(self):
        left_feature = torch.randn(2, 3, 5, 7)
        right_feature = torch.randn(2, 3, 6, 8)
        left_grid = torch.rand(2, 4, 6, 2) * 2 - 1
        right_grid = torch.rand(2, 4, 6, 3, 2) * 2 - 1
        left, right, valid = SphericalSweeping()(left_feature, right_feature, left_grid, right_grid)
        # 原入口使用无批次维 LUT；逐样本调用它，与 baseline 的批量采样逐元素比较。
        for i in range(2):
            grids = [left_grid[i].unsqueeze(2).expand(-1, -1, 3, -1), right_grid[i]]
            original = PDFOmni.spherical_sweep(None, [left_feature[i:i+1], right_feature[i:i+1]], grids)
            torch.testing.assert_close(left[i:i+1], original[0].permute(0, 1, 4, 2, 3), rtol=0, atol=0)
            torch.testing.assert_close(right[i:i+1], original[1].permute(0, 1, 4, 2, 3), rtol=0, atol=0)
        self.assertTrue(valid.all())

    def test_masked_softmax_excludes_even_high_scoring_invalid_candidates(self):
        logits = torch.tensor([[[[[100., 2., 0.]], [[0., 1., 1.]], [[1., 0., 2.]]]]], requires_grad=True)
        valid = torch.tensor([[[[[False, False, True]], [[True, False, True]], [[True, False, True]]]]])
        prob = masked_softmax(logits, valid)
        torch.testing.assert_close(prob.sum(dim=2), torch.tensor([[[[1., 0., 1.]]]]))
        self.assertEqual(prob[~valid].count_nonzero(), 0)
        prob.square().sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[~valid].count_nonzero(), 0)

    def test_default_pipeline_shapes_without_allocating_large_volumes(self):
        # meta 张量只追踪尺寸；下面另用 CPU 真正验证前向/反向数值。
        with torch.device("meta"):
            model = StereoBaseline().eval()
        left_grid, right_grid = make_grids(height=64, width=128, num_depth=96, device="meta")
        image = torch.empty(1, 3, 512, 512, device="meta")
        with torch.no_grad():
            out = model(image, image, left_grid, right_grid, return_intermediates=True)
        expected = {
            "left_feature": (1, 32, 256, 256), "right_feature": (1, 32, 256, 256),
            "left_volume": (1, 32, 96, 64, 128), "right_volume": (1, 32, 96, 64, 128),
            "gwc": (1, 8, 96, 64, 128), "attention_logits": (1, 1, 96, 64, 128),
            "left_compressed": (1, 16, 96, 64, 128), "right_compressed": (1, 16, 96, 64, 128),
            "concat_volume": (1, 32, 96, 64, 128), "acv": (1, 32, 96, 64, 128),
            "geometry_volume": (1, 8, 96, 64, 128), "cost": (1, 1, 96, 64, 128),
        }
        for key, shape in expected.items():
            with self.subTest(key=key):
                self.assertEqual(tuple(out["intermediates"][key].shape), shape)
        self.assertEqual(out["probability"].shape, (1, 96, 64, 128))
        self.assertEqual(out["inv_depth_low"].shape, (1, 1, 64, 128))
        self.assertEqual(out["depth"].shape, (1, 1, 128, 256))

    def test_uniform_cost_regresses_physical_inverse_depth_without_scale_factor(self):
        model = small_model().eval()
        torch.nn.init.zeros_(model.volume_generator.classifier.weight)
        left_grid, right_grid = make_grids()
        valid = torch.ones(1, 1, 16, 8, 16, dtype=torch.bool)
        valid[:, :, :, 2:4, 3:5] = False
        with torch.no_grad():
            out = model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64),
                        left_grid, right_grid, valid)
        expected = (1 / model.max_depth + 1 / model.min_depth) / 2
        for value, mask in [(out["inv_depth_low"], out["valid_mask_low"]),
                            (out["inv_depth"], out["valid_mask"])]:
            torch.testing.assert_close(value[mask], torch.full_like(value[mask], expected))
            self.assertEqual(value[~mask].count_nonzero(), 0)
        torch.testing.assert_close(out["depth"][out["valid_mask"]],
                                   torch.full_like(out["depth"][out["valid_mask"]], 1 / expected))

    def test_backward_reaches_both_encoders_attention_and_aggregation(self):
        model = small_model().train()
        left = torch.randn(1, 3, 64, 64, requires_grad=True)
        right = torch.randn(1, 3, 64, 64, requires_grad=True)
        left_grid, right_grid = make_grids()
        valid = torch.ones(1, 1, 16, 8, 16, dtype=torch.bool)
        valid[:, :, :, 0, 0] = False
        valid[:, :, :8, 1, 1] = False
        out = model(left, right, left_grid, right_grid, valid, return_intermediates=True)
        self.assertTrue(torch.isfinite(out["depth"]).all())
        self.assertEqual(out["probability"].masked_select(~valid[:, 0]).count_nonzero(), 0)
        self.assertEqual(out["attention"].masked_select(~valid).count_nonzero(), 0)
        out["inv_depth"][out["valid_mask"]].square().mean().backward()
        for name, param in model.named_parameters():
            with self.subTest(parameter=name):
                if ".conv2_down." in name:
                    self.assertIsNone(param.grad)  # 原版保留的 1/4 分支没有接入当前匹配路径。
                    continue
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
        self.assertGreater(left.grad.abs().sum().item(), 0)
        self.assertGreater(right.grad.abs().sum().item(), 0)
        self.assertGreater(model.volume_generator.attention.classif_att[-1].conv.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.volume_generator.aggregation.up1.conv.weight.grad.abs().sum().item(), 0)

    def test_all_invalid_forward_and_backward_are_finite(self):
        model = small_model().train()
        left_grid, right_grid = make_grids()
        valid = torch.zeros(1, 1, 16, 8, 16, dtype=torch.bool)
        out = model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64),
                    left_grid, right_grid, valid)
        for value in out.values():
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(value.count_nonzero(), 0)
        (out["depth"].sum() + out["inv_depth"].sum()).backward()
        for name, param in model.named_parameters():
            if ".conv2_down." in name:
                self.assertIsNone(param.grad)
                continue
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())

    def test_encoder_is_original_cnn_with_both_outputs(self):
        original = OriginalCNNEncoder(output_dim=32)
        encoder = CNNEncoder(output_dim=32)
        encoder.load_state_dict(original.state_dict())
        image = torch.randn(2, 3, 31, 35)
        actual = encoder(image)
        expected = original(image)
        self.assertEqual(actual[0].shape, (2, 32, 8, 9))
        self.assertEqual(actual[1].shape, (2, 32, 16, 18))
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)

    def test_encoder_sharing_is_optional(self):
        independent = small_model()
        self.assertIsInstance(independent.encoder_left, CNNEncoder)
        self.assertIsInstance(independent.encoder_right, CNNEncoder)
        self.assertIsNot(independent.encoder_left, independent.encoder_right)
        self.assertFalse({id(p) for p in independent.encoder_left.parameters()}
                         & {id(p) for p in independent.encoder_right.parameters()})
        shared = StereoBaseline(share_encoder=True)
        self.assertIs(shared.encoder_left, shared.encoder_right)

    def test_rejects_incompatible_hourglass_and_grid_sizes(self):
        for kwargs in [dict(num_depth=15), dict(sweep_size=(10, 16)),
                       dict(min_depth=0), dict(max_depth=float("inf"))]:
            with self.subTest(arguments=kwargs), self.assertRaises(AssertionError):
                StereoBaseline(**kwargs)
        model = small_model()
        left_grid, right_grid = make_grids()
        with self.assertRaisesRegex(AssertionError, "right_grid shape"):
            model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64), left_grid, right_grid[:, :, :, :-1])


if __name__ == "__main__":
    unittest.main()
