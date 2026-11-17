"""Noise processes: forward corruption q(x_t | x_0) and the reverse posterior.

    absorbing   tokens fall to MASK                         (GraphDiffBERT)
    uniform     tokens resample uniformly                   (GraphDiffBERTUniform)
    marginal    tokens resample from the data marginal      (GraphDiffBERTMarginal / Joint)
    gaussian    Gaussian noise in embedding space, DDIM     (ContinuousGraphDiffusion)
"""

from graphpath.processes.absorbing import AbsorbingMaskSchedule
from graphpath.processes.alphas import _cosine_alphas_cumprod, _linear_alphas_cumprod
from graphpath.processes.gaussian import ContinuousCosineSchedule
from graphpath.processes.marginal import MarginalTransitionSchedule, estimate_marginals
from graphpath.processes.uniform import UniformTransitionSchedule

__all__ = ["AbsorbingMaskSchedule", "UniformTransitionSchedule", "MarginalTransitionSchedule",
           "ContinuousCosineSchedule", "estimate_marginals",
           "_cosine_alphas_cumprod", "_linear_alphas_cumprod"]
