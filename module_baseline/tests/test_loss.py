import unittest
import torch

from module_baseline.loss_functions import sequence_loss


class RegressionLossTests(unittest.TestCase):
    def test_weighted_regression_excludes_invalid_nan_values(self):
        gt = torch.tensor([[[[1., 3., float("nan")]]]])
        valid = torch.tensor([[[[True, True, False]]]])
        first = torch.tensor([[[[2., 5., float("inf")]]]], requires_grad=True)
        last = torch.tensor([[[[1., 4., float("nan")]]]], requires_grad=True)
        loss = sequence_loss([first, last], gt, valid)
        self.assertAlmostEqual(loss.item(), 1.5 * .9**7.5 + .5, places=6)
        loss.backward()
        for prediction in (first, last):
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertEqual(prediction.grad[~valid].count_nonzero(), 0)
        smooth = sequence_loss([first, last], gt, valid, loss_gamma=1., use_smooth_l1=True)
        self.assertAlmostEqual(smooth.item(), 1.25, places=6)

    def test_all_invalid_regression_returns_differentiable_zero(self):
        prediction = torch.full((1, 1, 2, 2), float("nan"), requires_grad=True)
        valid = torch.zeros_like(prediction, dtype=torch.bool)
        for smooth in (False, True):
            loss = sequence_loss([prediction], torch.full_like(prediction, float("nan")), valid,
                                 use_smooth_l1=smooth)
            self.assertEqual(loss.item(), 0)
            loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(prediction.grad.count_nonzero(), 0)


if __name__ == "__main__":
    unittest.main()
