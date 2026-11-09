import itertools

import numpy as np
import torch

from qvla.action_scores import directional_score, variance_jacobian_score
from qvla.runtime import native_actions, verify_bidirectional_attention


def test_two_camera_scores_match_explicit_action_jacobian():
    # Two cameras with unequal token counts share each output channel. Compute
    # an explicit Jacobian, then average every possible Rademacher projection.
    cameras = [
        torch.tensor([[0.2, -0.7], [1.1, 0.8]], dtype=torch.float64),
        torch.tensor([[0.4, 0.3]], dtype=torch.float64),
    ]
    noise = [
        torch.tensor([[0.1, 0.2], [-0.3, 0.4]], dtype=torch.float64),
        torch.tensor([[0.5, -0.6]], dtype=torch.float64),
    ]

    def actions(first, second):
        return torch.stack((first[0, 0] + 2 * second[0, 1], first[1, 1] - first[0, 1] + 3 * second[0, 0]))

    jacobians = torch.autograd.functional.jacobian(actions, tuple(cameras))
    positions = sum(camera.shape[0] for camera in cameras)
    noise_sum = sum(delta.square().sum(0) for delta in noise)
    explicit_gain = sum(jac.square().sum((0, 1)) for jac in jacobians) / 2
    explicit_delta = sum((jac * delta).sum(1) for jac, delta in zip(jacobians, noise))
    expected_directional = explicit_delta.square().sum(0) / 2
    primary, directional = [], []
    for signs in itertools.product((-1.0, 1.0), repeat=2):
        r = torch.tensor(signs, dtype=torch.float64) / np.sqrt(2)
        gradients = [torch.einsum("a,atc->tc", r, jac) for jac in jacobians]
        gain = sum(gradient.square().sum(0) for gradient in gradients)
        dot = sum((gradient * delta).sum(0) for gradient, delta in zip(gradients, noise))
        primary.append(variance_jacobian_score(gain, noise_sum, positions))
        directional.append(directional_score(dot))
    torch.testing.assert_close(torch.stack(primary).mean(0), explicit_gain * noise_sum / positions)
    torch.testing.assert_close(torch.stack(directional).mean(0), expected_directional)


def test_native_action_dtype_order_matches_official_numpy():
    values = torch.tensor([1e-9, -1e-9, -0.5, 0.25, 0.75, -1.0, 1.0], dtype=torch.bfloat16).repeat(1, 8, 1)
    low = np.array([-0.31, -0.42, -0.53, -0.64, -0.75, -0.86, -0.97], dtype=np.float64)
    high = low + np.array([0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77])
    mask = np.array([True, True, False, True, True, False, True])
    original = values.float().numpy()
    expected = np.where(mask, 0.5 * (original + 1) * (high - low + 1e-8) + low, original)
    actual = native_actions(values, {"q01": low, "q99": high, "mask": mask})
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_installed_fork_is_bidirectional_and_does_not_change_rng():
    before = torch.get_rng_state().clone()
    assert verify_bidirectional_attention() > 1e-7
    assert torch.equal(torch.get_rng_state(), before)
