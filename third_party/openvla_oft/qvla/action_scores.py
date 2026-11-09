"""Channel score reductions after aggregating all calls of a shared module."""


def variance_jacobian_score(gain, noise_sum, positions):
    if positions < 1:
        raise ValueError("A channel must have at least one observed position")
    return gain * noise_sum / positions


def directional_score(dot):
    return dot.square()
