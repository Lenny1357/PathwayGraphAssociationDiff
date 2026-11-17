"""Reverse-time generation, one function per noise process.

Each takes the model as its first argument and returns `(node_tokens (B,N),
adjacency (B,N,N))`. The models expose them as `model.sample(...)`.
"""

from graphpath.samplers.continuous import sample_continuous
from graphpath.samplers.marginal import sample_marginal
from graphpath.samplers.masked import sample_masked
from graphpath.samplers.uniform import sample_uniform
from graphpath.samplers.vae import sample_vae

__all__ = ["sample_masked", "sample_uniform", "sample_marginal", "sample_continuous",
           "sample_vae"]
