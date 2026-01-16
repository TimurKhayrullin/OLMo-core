"""
Custom model configurations for Relace experiments.

Usage:
    from scripts.relace.model_configs import llama2_135M

    model_config = llama2_135M(vocab_size=tokenizer_config.padded_vocab_size())
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Dict, List, Optional

from olmo_core.config import DType, StrEnum
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import ensure_multiple_of

from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.config import ModelConfig, ModuleConfig
from olmo_core.nn.feed_forward import FeedForwardConfig, FeedForwardType
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMHeadType
from olmo_core.nn.moe import MoEConfig, MoERouterConfig, MoEType
from olmo_core.nn.rope import RoPEConfig, RoPEScalingConfig, RoPEType
from olmo_core.nn.transformer.init import InitMethod

from olmo_core.nn.transformer import TransformerConfig, TransformerType, TransformerBlockType


def llama2_271M(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    A 271M Llama2-like model config.
    """
    return TransformerConfig.llama_like(
        d_model=1024,
        vocab_size=vocab_size,
        n_layers=kwargs.pop("n_layers", 16),
        n_heads=kwargs.pop("n_heads", 8),
        rope_theta=kwargs.pop("rope_theta", 10_000),
        **kwargs,
    )


def llama2_135M(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    A ~135M Llama2-like model config (half of 271M).

    Scaled down by reducing d_model from 1024 to 768 and n_layers from 16 to 12.
    n_heads=12 gives 64 dims per head (standard).
    """
    return TransformerConfig.llama_like(
        d_model=768,
        vocab_size=vocab_size,
        n_layers=kwargs.pop("n_layers", 12),
        n_heads=kwargs.pop("n_heads", 12),
        rope_theta=kwargs.pop("rope_theta", 10_000),
        **kwargs,
    )

# ==============================================================================
# MoE Model Configurations
# ==============================================================================
#
# GRANULARITY DEFINITION:
#   granularity = 2 * d_model / expert_hidden
#   For granularity=12: expert_hidden = d_model / 6
#
# SHARED EXPERT:
#   All models (except OLMo-style 1B) use shared_hidden = expert_hidden,
#   meaning the shared expert is the SAME SIZE as each routed expert.
#
# ACTIVE PARAMETER NOTES:
# -----------------------
# Always-active components (cannot be reduced via sparsity):
#   - Embeddings: vocab_size * d_model
#   - Attention: ~4 * d_model^2 * n_layers
#   - Shared MLP: 3 * d_model * shared_hidden * n_layers
#   - Router: d_model * num_experts * n_layers
#   - LM Head: d_model * vocab_size
#
# The active ratio is bounded below by these components.
# ==============================================================================


def smallmoe_300M(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    ~267M total params MoE with granularity=12 and uniform expert sizing.

    Target: 250-300M total
    Achieved: ~267M total, ~72M active (~27%) for vocab_size=50k

    Granularity = 2 * 576 / 96 = 12 ✓
    Shared expert size = routed expert size = 96 ✓

    Architecture:
    -------------
    d_model=576:
        - Chosen so d_model/6 = 96 (clean division for granularity=12)
        - 576 / 9 heads = 64 dims per head

    expert_hidden=96 (d_model/6):
        - Gives granularity = 2 * 576 / 96 = 12

    shared_hidden=96:
        - Same size as routed experts (1 shared expert requirement)

    num_experts=96, top_k=1:
        - 96 experts for param budget, top_k=1 for sparsity

    Parameter breakdown (vocab_size=50k):
    -------------------------------------
    Per layer MoE:
      - Router: 576 * 96 = 55,296
      - Experts: 3 * 576 * 96 * 96 = 15,925,248
      - Shared: 3 * 576 * 96 = 165,888
      - Total MoE: 16,146,432

    Per layer attention: 4 * 576^2 = 1,327,104
    Per layer total: ~17.5M
    12 layers: ~210M
    Embeddings + LM head: ~57.6M
    Total: ~267M

    Active params:
      - Router + shared + 1 expert: ~2.6M per layer * 12 = ~31M
      - Attention: ~16M
      - Embeddings + LM: ~57.6M
      - Total active: ~72M (~27%)
    """
    d_model = kwargs.pop("d_model", 576)
    n_heads = kwargs.pop("n_heads", 9)  # 576/9 = 64 dims per head
    n_layers = kwargs.pop("n_layers", 12)
    num_experts = kwargs.pop("num_experts", 96)
    top_k = kwargs.pop("top_k", 1)
    # Granularity = 12 → expert_hidden = d_model / 6
    expert_hidden = kwargs.pop("expert_hidden", d_model // 6)  # 96
    # Shared expert same size as routed experts
    shared_hidden = kwargs.pop("shared_hidden", expert_hidden)  # 96

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        name=kwargs.pop("name", TransformerType.moe),
        block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        feed_forward_moe=MoEConfig(
            name=MoEType.default,
            num_experts=num_experts,
            hidden_size=expert_hidden,
            router=MoERouterConfig(top_k=top_k),
            # Shared expert = same size as routed experts
            shared_mlp=FeedForwardConfig(hidden_size=shared_hidden, bias=False),
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
        ),
        **kwargs,
    )


def smallmoe_150M(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    ~132M total params MoE with granularity=12 and uniform expert sizing.

    Target: 100-150M total
    Achieved: ~132M total, ~50M active (~38%) for vocab_size=50k

    Granularity = 2 * 384 / 64 = 12 ✓
    Shared expert size = routed expert size = 64 ✓

    Architecture:
    -------------
    d_model=384:
        - Chosen so d_model/6 = 64 (clean division for granularity=12)
        - 384 / 6 heads = 64 dims per head

    expert_hidden=64 (d_model/6):
        - Gives granularity = 2 * 384 / 64 = 12

    shared_hidden=64:
        - Same size as routed experts

    num_experts=96, top_k=1:
        - 96 experts to hit param target

    Parameter breakdown (vocab_size=50k):
    -------------------------------------
    Per layer MoE:
      - Router: 384 * 96 = 36,864
      - Experts: 3 * 384 * 64 * 96 = 7,077,888
      - Shared: 3 * 384 * 64 = 73,728
      - Total MoE: 7,188,480

    Per layer attention: 4 * 384^2 = 589,824
    Per layer total: ~7.8M
    12 layers: ~93.4M
    Embeddings + LM head: ~38.4M
    Total: ~132M

    Active params:
      - Router + shared + 1 expert: ~1.2M per layer * 12 = ~14M
      - Attention: ~7M
      - Embeddings + LM: ~38.4M
      - Total active: ~50M (~38%)
    """
    d_model = kwargs.pop("d_model", 384)
    n_heads = kwargs.pop("n_heads", 6)  # 384/6 = 64 dims per head
    n_layers = kwargs.pop("n_layers", 12)
    num_experts = kwargs.pop("num_experts", 96)
    top_k = kwargs.pop("top_k", 1)
    # Granularity = 12 → expert_hidden = d_model / 6
    expert_hidden = kwargs.pop("expert_hidden", d_model // 6)  # 64
    # Shared expert same size as routed experts
    shared_hidden = kwargs.pop("shared_hidden", expert_hidden)  # 64

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        name=kwargs.pop("name", TransformerType.moe),
        block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        feed_forward_moe=MoEConfig(
            name=MoEType.default,
            num_experts=num_experts,
            hidden_size=expert_hidden,
            router=MoERouterConfig(top_k=top_k),
            shared_mlp=FeedForwardConfig(hidden_size=shared_hidden, bias=False),
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
        ),
        **kwargs,
    )


def smallmoe_1B_A390M(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    A ~1B total param MoE model mirroring OLMo smallmoe architecture.

    Target: 0.9-1B total, ~39% active (like original OLMo smallmoe)
    Achieved: ~1.04B total, ~370M active (~36% ratio) for vocab_size=50k

    Architecture choices (scaled from OLMo smallmoe):
    -------------------------------------------------
    d_model=1152:
        - Scaled up from OLMo's 768 to reach 1B params
        - 1152 / 18 heads = 64 dims per head (standard)
        - Scaling factor: 1152/768 = 1.5x

    n_layers=12:
        - Same as OLMo smallmoe

    num_experts=32, top_k=4:
        - IDENTICAL to OLMo smallmoe
        - This gives the ~39% active ratio characteristic of OLMo

    expert_hidden_size=576 (0.5 * d_model):
        - Same ratio as OLMo smallmoe
        - Expert params: 3 * 1152 * 576 * 32 = 63.7M per layer

    shared_mlp_hidden=2304 (2 * d_model):
        - Same ratio as OLMo smallmoe

    Parameter breakdown (vocab_size=50k):
    -------------------------------------
    Per layer MoE:
      - Router: 1152 * 32 = 36,864
      - Experts: 3 * 1152 * 576 * 32 = 63,700,992
      - Shared: 3 * 1152 * 2304 = 7,962,624
      - Total MoE: 71,700,480

    Per layer attention:
      - Q,K,V,O: 4 * 1152^2 = 5,308,416
      - QK norm: ~2,304

    12 layers: ~924M
    Embeddings: 1152 * 50k = 57.6M
    LM Head: 57.6M
    Total: ~1.04B

    Active params:
      - Router: 36,864 * 12 = 442,368
      - Active experts (k=4): 3 * 1152 * 576 * 4 * 12 = 95.5M
      - Shared: 7,962,624 * 12 = 95.5M
      - Attention: 5.3M * 12 = 63.7M
      - Embeddings + LM: 115.2M
      - Total active: ~370M (~36%)
    """
    d_model = kwargs.pop("d_model", 1152)
    n_heads = kwargs.pop("n_heads", 18)  # 1152/18 = 64 dims per head
    n_layers = kwargs.pop("n_layers", 12)
    # Mirror OLMo smallmoe exactly for expert config
    num_experts = kwargs.pop("num_experts", 32)
    top_k = kwargs.pop("top_k", 4)
    expert_hidden = kwargs.pop("expert_hidden", int(0.5 * d_model))  # 576
    shared_hidden = kwargs.pop("shared_hidden", d_model * 2)  # 2304

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        name=kwargs.pop("name", TransformerType.moe),
        block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        feed_forward_moe=MoEConfig(
            name=MoEType.default,
            num_experts=num_experts,
            hidden_size=expert_hidden,
            router=MoERouterConfig(top_k=top_k),
            shared_mlp=FeedForwardConfig(hidden_size=shared_hidden, bias=False),
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
        ),
        **kwargs,
    )


def smallmoe_1B(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    ~1.02B total params MoE with granularity=12 and uniform expert sizing.

    Target: 0.9-1B total
    Achieved: ~1.02B total, ~105M active (~10%) for vocab_size=50k

    Granularity = 2 * 768 / 128 = 12 ✓
    Shared expert size = routed expert size = 128 ✓

    Architecture:
    -------------
    d_model=768:
        - Standard size, 768/6 = 128 (clean division for granularity=12)
        - 768 / 12 heads = 64 dims per head

    expert_hidden=128 (d_model/6):
        - Gives granularity = 2 * 768 / 128 = 12

    shared_hidden=128:
        - Same size as routed experts

    num_experts=256, top_k=1:
        - 256 experts to reach 1B params
        - top_k=1 for maximum sparsity

    Parameter breakdown (vocab_size=50k):
    -------------------------------------
    Per layer MoE:
      - Router: 768 * 256 = 196,608
      - Experts: 3 * 768 * 128 * 256 = 75,497,472
      - Shared: 3 * 768 * 128 = 294,912
      - Total MoE: 75,988,992

    Per layer attention: 4 * 768^2 = 2,359,296
    Per layer total: ~78.3M
    12 layers: ~940M
    Embeddings + LM head: ~76.8M
    Total: ~1.02B

    Active params:
      - Router: 196,608 * 12 = 2.4M
      - Active experts (k=1): 3 * 768 * 128 * 1 * 12 = 3.5M
      - Shared: 294,912 * 12 = 3.5M
      - Attention: 2.4M * 12 = 28.3M
      - Embeddings + LM: 76.8M
      - Total active: ~105M (~10%)

    Note: This achieves a much lower active ratio than the OLMo-style
    smallmoe_1B_A390M (~36%) because:
      1. More experts (256 vs 32)
      2. Smaller expert hidden size (128 vs 576)
      3. Smaller shared MLP (128 vs 2304)
      4. top_k=1 vs top_k=4
    """
    d_model = kwargs.pop("d_model", 768)
    n_heads = kwargs.pop("n_heads", 12)  # 768/12 = 64 dims per head
    n_layers = kwargs.pop("n_layers", 12)
    num_experts = kwargs.pop("num_experts", 256)
    top_k = kwargs.pop("top_k", 1)
    # Granularity = 12 → expert_hidden = d_model / 6
    expert_hidden = kwargs.pop("expert_hidden", d_model // 6)  # 128
    # Shared expert same size as routed experts
    shared_hidden = kwargs.pop("shared_hidden", expert_hidden)  # 128

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        name=kwargs.pop("name", TransformerType.moe),
        block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        feed_forward_moe=MoEConfig(
            name=MoEType.default,
            num_experts=num_experts,
            hidden_size=expert_hidden,
            router=MoERouterConfig(top_k=top_k),
            shared_mlp=FeedForwardConfig(hidden_size=shared_hidden, bias=False),
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
        ),
        **kwargs,
    )

# ==============================================================================
# Tiny Dense Models for Scaling Law Experiments
# ==============================================================================
#
# NON-EMBEDDING PARAMETER CALCULATION:
# For a LLaMA-like dense transformer, non-embedding params per layer:
#   - Attention (Q,K,V,O): 4 * d_model^2
#   - FFN (SwiGLU with hidden = 8/3 * d_model): 3 * d_model * hidden_size
#   - Layer norms: ~4 * d_model (negligible)
#
# Total per layer ≈ 4*d^2 + 3*d*h where h ≈ 8/3 * d ≈ 2.67d
# So per layer ≈ 4*d^2 + 8*d^2 = 12*d^2 (rough approximation)
#
# ==============================================================================


def dense_100K(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    A tiny ~100K non-embedding parameter dense LLaMA-like model.

    Target: ~100K non-embedding parameters

    Design rationale:
    -----------------
    We need: n_layers * (4*d^2 + 3*d*h) ≈ 100,000

    With d_model=48, n_layers=4:
      - Attention: 4 * 48^2 = 9,216 per layer
      - FFN hidden ≈ 8/3 * 48 = 128 (rounded to multiple of 8)
      - FFN: 3 * 48 * 128 = 18,432 per layer
      - Per layer total: ~27,648
      - 4 layers: ~110,592 ≈ 100K ✓

    Architecture:
    -------------
    d_model=48:
        - Small but functional embedding dimension
        - 48 / 4 heads = 12 dims per head (minimal but works)

    n_layers=4:
        - Enough depth for some representation learning
        - 4 layers * ~28K = ~110K non-embed params

    n_heads=4:
        - Gives 12 dimensions per head
        - Minimum reasonable for attention patterns

    Note: Embedding params (2 * vocab_size * d_model) are NOT included
    in the 100K target. For vocab_size=50K, embeddings add ~4.8M params.
    """
    # d_model=48 chosen so 4 layers gives ~100K non-embed params
    # 48 is divisible by 4 (n_heads) giving 12 dims per head
    d_model = kwargs.pop("d_model", 48)
    n_heads = kwargs.pop("n_heads", 4)  # 48/4 = 12 dims per head
    n_layers = kwargs.pop("n_layers", 4)

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        # hidden_size_multiplier=1.0 uses default LLaMA FFN sizing (8/3 * d_model)
        hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 1.0),
        block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        **kwargs,
    )


def dense_10K(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    A very tiny ~10K non-embedding parameter dense LLaMA-like model.

    Target: ~10K non-embedding parameters

    Design rationale:
    -----------------
    We need: n_layers * (4*d^2 + 3*d*h) ≈ 10,000

    With d_model=20, n_layers=2:
      - Attention: 4 * 20^2 = 1,600 per layer
      - FFN hidden ≈ 8/3 * 20 ≈ 56 (rounded to multiple of 8)
      - FFN: 3 * 20 * 56 = 3,360 per layer
      - Per layer total: ~4,960
      - 2 layers: ~9,920 ≈ 10K ✓

    Architecture:
    -------------
    d_model=20:
        - Extremely small embedding dimension
        - 20 / 2 heads = 10 dims per head (very minimal)
        - This is at the edge of what's functional

    n_layers=2:
        - Minimum depth for any compositional computation
        - 2 layers * ~5K = ~10K non-embed params

    n_heads=2:
        - Gives 10 dimensions per head
        - Bare minimum for attention to be meaningful

    Warning: This model is EXTREMELY small and may not learn well.
    It's primarily useful for:
      - Sanity checking training pipelines
      - Extrapolating scaling laws to tiny compute
      - Fast iteration on hyperparameter searches

    Note: Embedding params (2 * vocab_size * d_model) are NOT included
    in the 10K target. For vocab_size=50K, embeddings add ~2M params.
    """
    # d_model=20 chosen so 2 layers gives ~10K non-embed params
    # 20 is divisible by 2 (n_heads) giving 10 dims per head
    d_model = kwargs.pop("d_model", 20)
    n_heads = kwargs.pop("n_heads", 2)  # 20/2 = 10 dims per head
    n_layers = kwargs.pop("n_layers", 2)

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        # hidden_size_multiplier=1.0 uses default LLaMA FFN sizing (8/3 * d_model)
        hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 1.0),
        block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        **kwargs,
    )


# ==============================================================================
# 3B MoE Model Configurations
# ==============================================================================


def smallmoe_3BA300M(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    ~3B total params MoE with granularity=12, ~300M active (~10% ratio).

    This mimics smallmoe_1B architecture scaled up to 3B total params.

    Target: ~3B total, ~300M active (10% ratio like smallmoe_1B)
    Achieved: ~3.0B total, ~299M active (~10%) for vocab_size=50k

    Granularity = 2 * 1536 / 256 = 12 ✓
    Shared expert size = routed expert size = 256 ✓

    Architecture:
    -------------
    d_model=1536:
        - Scaled up from smallmoe_1B's 768 to reach 3B params
        - 1536 / 24 heads = 64 dims per head (standard)
        - 1536 / 6 = 256 (clean division for granularity=12)

    expert_hidden=256 (d_model/6):
        - Gives granularity = 2 * 1536 / 256 = 12
        - Same granularity as smallmoe_1B

    shared_hidden=256:
        - Same size as routed experts (uniform sizing)

    num_experts=192, top_k=1:
        - 192 experts to reach 3B params while maintaining sparsity
        - top_k=1 for maximum sparsity (same as smallmoe_1B)

    Parameter breakdown (vocab_size=50k):
    -------------------------------------
    Per layer MoE:
      - Router: 1536 * 192 = 294,912
      - Experts: 3 * 1536 * 256 * 192 = 226,492,416
      - Shared: 3 * 1536 * 256 = 1,179,648
      - Total MoE: 227,966,976

    Per layer attention: 4 * 1536^2 = 9,437,184
    Per layer total: ~237.4M
    12 layers: ~2.85B
    Embeddings + LM head: ~153.6M
    Total: ~3.0B

    Active params:
      - Router: 294,912 * 12 = 3.54M
      - Active experts (k=1): 3 * 1536 * 256 * 1 * 12 = 14.16M
      - Shared: 1,179,648 * 12 = 14.16M
      - Attention: 9.44M * 12 = 113.25M
      - Embeddings + LM: 153.6M
      - Total active: ~299M (~10%)
    """
    d_model = kwargs.pop("d_model", 1536)
    n_heads = kwargs.pop("n_heads", 24)  # 1536/24 = 64 dims per head
    n_layers = kwargs.pop("n_layers", 12)
    num_experts = kwargs.pop("num_experts", 192)
    top_k = kwargs.pop("top_k", 1)
    # Granularity = 12 → expert_hidden = d_model / 6
    expert_hidden = kwargs.pop("expert_hidden", d_model // 6)  # 256
    # Shared expert same size as routed experts
    shared_hidden = kwargs.pop("shared_hidden", expert_hidden)  # 256

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        name=kwargs.pop("name", TransformerType.moe),
        block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        feed_forward_moe=MoEConfig(
            name=MoEType.default,
            num_experts=num_experts,
            hidden_size=expert_hidden,
            router=MoERouterConfig(top_k=top_k),
            shared_mlp=FeedForwardConfig(hidden_size=shared_hidden, bias=False),
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
        ),
        **kwargs,
    )


def smallmoe_3BA1B(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    ~3B total params MoE mirroring OLMo smallmoe architecture, ~1.17B active (~37%).

    This mimics smallmoe_1B_A390M architecture scaled up to 3B total params.

    Target: ~3B total, ~1.17B active (37% ratio like smallmoe_1B_A390M's 36%)
    Achieved: ~3.12B total, ~1.16B active (~37%) for vocab_size=50k

    Architecture (scaled from smallmoe_1B_A390M):
    ---------------------------------------------
    d_model=2048:
        - Scaled up from smallmoe_1B_A390M's 1152 to reach 3B params
        - 2048 / 32 heads = 64 dims per head (standard)
        - Scaling factor: 2048/1152 ≈ 1.78x

    n_layers=12:
        - Same as smallmoe_1B_A390M

    num_experts=32, top_k=6:
        - Same num_experts as smallmoe_1B_A390M
        - top_k=6 (increased from 4) to maintain ~37% active ratio at larger scale

    expert_hidden=1024 (0.5 * d_model):
        - Same ratio as smallmoe_1B_A390M

    shared_hidden=4096 (2 * d_model):
        - Same ratio as smallmoe_1B_A390M

    Parameter breakdown (vocab_size=50k):
    -------------------------------------
    Per layer MoE:
      - Router: 2048 * 32 = 65,536
      - Experts: 3 * 2048 * 1024 * 32 = 201,326,592
      - Shared: 3 * 2048 * 4096 = 25,165,824
      - Total MoE: 226,557,952

    Per layer attention: 4 * 2048^2 = 16,777,216
    Per layer total: ~243.3M
    12 layers: ~2.92B
    Embeddings + LM head: ~204.8M
    Total: ~3.12B

    Active params:
      - Router: 65,536 * 12 = 0.79M
      - Active experts (k=6): 3 * 2048 * 1024 * 6 * 12 = 452.98M
      - Shared: 25,165,824 * 12 = 302M
      - Attention: 16.78M * 12 = 201.3M
      - Embeddings + LM: 204.8M
      - Total active: ~1.16B (~37%)
    """
    d_model = kwargs.pop("d_model", 2048)
    n_heads = kwargs.pop("n_heads", 32)  # 2048/32 = 64 dims per head
    n_layers = kwargs.pop("n_layers", 12)
    # Mirror OLMo smallmoe style for expert config
    num_experts = kwargs.pop("num_experts", 32)
    top_k = kwargs.pop("top_k", 6)  # Increased from 4 to maintain active ratio at scale
    expert_hidden = kwargs.pop("expert_hidden", int(0.5 * d_model))  # 1024
    shared_hidden = kwargs.pop("shared_hidden", d_model * 2)  # 4096

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        name=kwargs.pop("name", TransformerType.moe),
        block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        feed_forward_moe=MoEConfig(
            name=MoEType.default,
            num_experts=num_experts,
            hidden_size=expert_hidden,
            router=MoERouterConfig(top_k=top_k),
            shared_mlp=FeedForwardConfig(hidden_size=shared_hidden, bias=False),
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
        ),
        **kwargs,
    )


def smallmoe_3BA100M(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    ~3B total params MoE with extreme sparsity, ~100M active (~3.4% ratio).

    This mimics smallmoe_1B architecture but pushes sparsity to the extreme.

    Target: ~3B total, ~100M active (3.4% ratio)
    Achieved: ~3.05B total, ~103M active (~3.4%) for vocab_size=50k

    Granularity = 2 * 672 / 112 = 12 ✓
    Shared expert size = routed expert size = 112 ✓

    Architecture:
    -------------
    d_model=672:
        - Smaller than smallmoe_1B's 768 to reduce always-active params
        - 672 / 12 heads = 56 dims per head
        - 672 / 6 = 112 (clean division for granularity=12)

    expert_hidden=112 (d_model/6):
        - Gives granularity = 2 * 672 / 112 = 12
        - Same granularity pattern as smallmoe_1B

    shared_hidden=112:
        - Same size as routed experts (uniform sizing)

    num_experts=1088, top_k=1:
        - Many more experts (1088 vs smallmoe_1B's 256) for extreme sparsity
        - top_k=1 for maximum sparsity
        - This is what enables ~3% active ratio

    Parameter breakdown (vocab_size=50k):
    -------------------------------------
    Per layer MoE:
      - Router: 672 * 1088 = 731,136
      - Experts: 3 * 672 * 112 * 1088 = 245,661,696
      - Shared: 3 * 672 * 112 = 225,792
      - Total MoE: 246,618,624

    Per layer attention: 4 * 672^2 = 1,806,336
    Per layer total: ~248.4M
    12 layers: ~2.98B
    Embeddings + LM head: ~67.2M
    Total: ~3.05B

    Active params:
      - Embeddings + LM: 67.2M
      - Attention: 1.81M * 12 = 21.7M
      - Router: 0.73M * 12 = 8.77M
      - Shared: 0.23M * 12 = 2.71M
      - Active expert (k=1): 0.23M * 12 = 2.71M
      - Total active: ~103M (~3.4%)

    Note: This achieves extremely low active ratio by:
      1. Many more experts (1088 vs typical 32-256)
      2. Smaller d_model (less attention/embedding overhead)
      3. Tiny expert and shared MLP sizes
      4. top_k=1 for maximum routing sparsity
    """
    d_model = kwargs.pop("d_model", 672)
    n_heads = kwargs.pop("n_heads", 12)  # 672/12 = 56 dims per head
    n_layers = kwargs.pop("n_layers", 12)
    num_experts = kwargs.pop("num_experts", 1088)
    top_k = kwargs.pop("top_k", 1)
    # Granularity = 12 → expert_hidden = d_model / 6
    expert_hidden = kwargs.pop("expert_hidden", d_model // 6)  # 112
    # Shared expert same size as routed experts
    shared_hidden = kwargs.pop("shared_hidden", expert_hidden)  # 112

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        name=kwargs.pop("name", TransformerType.moe),
        block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        feed_forward_moe=MoEConfig(
            name=MoEType.default,
            num_experts=num_experts,
            hidden_size=expert_hidden,
            router=MoERouterConfig(top_k=top_k),
            shared_mlp=FeedForwardConfig(hidden_size=shared_hidden, bias=False),
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
        ),
        **kwargs,
    )


def tinydense_500K(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    A tiny ~500K non-embedding parameter dense LLaMA-like model.

    Target: ~500K non-embedding parameters
    Achieved: 500,446 params (+0.1%)

    Design rationale (based on Kaplan et al. 2020 "Scaling Laws for Neural Language Models"):
    -----------------------------------------------------------------------------------------
    The paper found that performance depends weakly on model shape when total
    non-embedding parameter count N is held fixed. Models with at least 2 layers
    and non-extreme depth-to-width ratios follow the same scaling trends.

    We need: n_layers * (4*d^2 + 3*d*h + 2*d) + d ≈ 500,000
    where h = hidden_size (FFN), approximately 8/3 * d for standard LLaMA

    With d_model=102, n_layers=4, n_heads=6:
      - head_dim = 102/6 = 17
      - Attention: 4 * 102^2 + 2*17 = 41,650 per layer (includes qk_norm)
      - FFN hidden = 272 (rounded 8/3 * 102 to multiple of 8)
      - FFN: 3 * 102 * 272 = 83,232 per layer
      - Layer norms: 2 * 102 = 204 per layer
      - Per layer total: 125,086
      - 4 layers: 500,344
      - LM head norm: 102
      - Total: 500,446 ≈ 500K ✓

    Architecture:
    -------------
    d_model=102:
        - Optimized for ~500K non-embed params
        - 102 / 6 heads = 17 dims per head

    n_layers=4:
        - Reasonable depth per Kaplan et al.
        - Models with <2 layers deviate from scaling trends

    n_heads=6:
        - Gives 17 dimensions per head
        - Good for pattern learning

    Note: Embedding params (vocab_size * d_model) are NOT included in the 500K target.
    """
    d_model = kwargs.pop("d_model", 102)
    n_heads = kwargs.pop("n_heads", 6)  # 102/6 = 17 dims per head
    n_layers = kwargs.pop("n_layers", 4)

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 1.0),
        hidden_size_multiple_of=kwargs.pop("hidden_size_multiple_of", 8),  # Allow smaller FFN
        block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        **kwargs,
    )


def tinydense_100K(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    A tiny ~100K non-embedding parameter dense LLaMA-like model.

    Target: ~100K non-embedding parameters
    Achieved: 99,840 params (-0.2%)

    Design rationale (based on Kaplan et al. 2020 "Scaling Laws for Neural Language Models"):
    -----------------------------------------------------------------------------------------
    With d_model=40, n_layers=5, n_heads=2:
      - head_dim = 40/2 = 20
      - Attention: 4 * 40^2 + 2*20 = 6,440 per layer (includes qk_norm)
      - FFN hidden = 112 (rounded 8/3 * 40 to multiple of 8)
      - FFN: 3 * 40 * 112 = 13,440 per layer
      - Layer norms: 2 * 40 = 80 per layer
      - Per layer total: 19,960
      - 5 layers: 99,800
      - LM head norm: 40
      - Total: 99,840 ≈ 100K ✓

    Architecture:
    -------------
    d_model=40:
        - Optimized for ~100K non-embed params
        - 40 / 2 heads = 20 dims per head

    n_layers=5:
        - Reasonable depth per Kaplan et al.
        - More depth than width for small models

    n_heads=2:
        - Gives 20 dimensions per head
        - Reasonable for small-scale attention

    Note: Embedding params are NOT included in the 100K target.
    """
    d_model = kwargs.pop("d_model", 40)
    n_heads = kwargs.pop("n_heads", 2)  # 40/2 = 20 dims per head
    n_layers = kwargs.pop("n_layers", 5)

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 1.0),
        hidden_size_multiple_of=kwargs.pop("hidden_size_multiple_of", 8),  # Allow smaller FFN
        block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        **kwargs,
    )


def tinydense_10K(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    A very tiny ~10K non-embedding parameter dense LLaMA-like model.

    Target: ~10K non-embedding parameters
    Achieved: 10,060 params (+0.6%)

    Design rationale (based on Kaplan et al. 2020 "Scaling Laws for Neural Language Models"):
    -----------------------------------------------------------------------------------------
    WARNING: The paper notes that models with <2 layers deviate from scaling trends.
    This 2-layer model is at the edge of validity for scaling law predictions.

    With d_model=20, n_layers=2, n_heads=2:
      - head_dim = 20/2 = 10
      - Attention: 4 * 20^2 + 2*10 = 1,620 per layer (includes qk_norm)
      - FFN hidden = 56 (rounded 8/3 * 20 to multiple of 8)
      - FFN: 3 * 20 * 56 = 3,360 per layer
      - Layer norms: 2 * 20 = 40 per layer
      - Per layer total: 5,020
      - 2 layers: 10,040
      - LM head norm: 20
      - Total: 10,060 ≈ 10K ✓

    Architecture:
    -------------
    d_model=20:
        - Very small embedding dimension
        - 20 / 2 heads = 10 dims per head

    n_layers=2:
        - Minimum depth for compositional computation
        - At the boundary of Kaplan et al. scaling law validity

    n_heads=2:
        - Minimum for multi-head attention
        - 10 dims per head is reasonable for this scale

    Warning: This model is EXTREMELY small and may not learn well.
    It's primarily useful for:
      - Sanity checking training pipelines
      - Extrapolating scaling laws to tiny compute
      - Fast iteration on hyperparameter searches
    """
    d_model = kwargs.pop("d_model", 20)
    n_heads = kwargs.pop("n_heads", 2)  # 20/2 = 10 dims per head
    n_layers = kwargs.pop("n_layers", 2)

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 1.0),
        hidden_size_multiple_of=kwargs.pop("hidden_size_multiple_of", 8),  # Allow smaller FFN
        block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        **kwargs,
    )


def tinydense_1K(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    An extremely tiny ~1K non-embedding parameter dense LLaMA-like model.

    Target: ~1K non-embedding parameters

    Design rationale:
    -----------------
    WARNING: This model is far outside the regime studied by Kaplan et al.
    It exists primarily for:
      - Testing training pipeline overhead
      - Sanity checking code paths
      - Exploring extreme extrapolation of scaling laws

    With d_model=8, n_layers=2:
      - head_dim = 8/2 = 4 (2 heads)
      - Attention: 4 * 8^2 = 256 per layer
      - FFN hidden = round(8/3 * 8) = 24 (multiple of 8)
      - FFN: 3 * 8 * 24 = 576 per layer
      - Layer norms: ~2 * 8 = 16 per layer
      - Per layer total: ~848
      - 2 layers: ~1,696
      - LM head norm: 8
      - Total: ~1,704 ≈ 1K (within 70%)

    For closer to 1K, use d_model=6, n_layers=2:
      - Attention: 4 * 6^2 = 144 per layer
      - FFN hidden = 16 (minimum practical)
      - FFN: 3 * 6 * 16 = 288 per layer
      - Layer norms: ~2 * 6 = 12 per layer
      - Per layer total: ~444
      - 2 layers: ~888
      - LM head norm: 6
      - Total: ~894 ≈ 1K ✓

    Architecture:
    -------------
    d_model=6:
        - Extremely small, barely functional
        - 6 / 2 heads = 3 dims per head (!)

    n_layers=2:
        - Minimum depth

    n_heads=2:
        - Minimum heads, but only 3 dims per head

    WARNING: This model is so small it may not meaningfully learn language.
    Use only for pipeline testing and extreme scaling law extrapolation.
    """
    d_model = kwargs.pop("d_model", 6)
    n_heads = kwargs.pop("n_heads", 2)  # 6/2 = 3 dims per head (!)
    n_layers = kwargs.pop("n_layers", 2)

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 1.0),
        hidden_size_multiple_of=kwargs.pop("hidden_size_multiple_of", 8),  # Allow smaller FFN
        block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        **kwargs,
    )


def smallmoe_3BA30M(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    ~3B total params MoE with ultra-extreme sparsity, ~30M active (~1% ratio).

    Target: ~3B total, ~30M active (1% ratio)
    Achieved: ~3.1B total, ~29M active (~0.94%) for vocab_size=50k

    IMPORTANT: This config deviates from the granularity=12 pattern used in other
    smallmoe configs. Achieving ~1% active ratio requires a different architecture:
    - Very small d_model (to minimize embedding/attention overhead)
    - Fewer but LARGER experts (to pack params without inflating router)
    - Tiny shared MLP

    The granularity=12 pattern would require thousands of tiny experts, making
    the router (d_model * num_experts * n_layers) exceed the active param budget.

    Architecture:
    -------------
    d_model=192:
        - Very small to minimize always-active attention/embedding params
        - 192 / 3 heads = 64 dims per head (standard)
        - Embeddings: 2 * 192 * 50k = 19.2M (already 64% of active budget!)

    expert_hidden=768 (4 * d_model):
        - INVERTED from granularity=12 pattern (which would use d_model/6=32)
        - Larger experts = fewer needed for 3B total = smaller router
        - This is key to achieving 1% active ratio

    shared_hidden=192:
        - Minimal shared MLP (same as d_model)
        - Keeps shared contribution small

    num_experts=580, top_k=1:
        - Moderate number of large experts
        - top_k=1 for maximum sparsity

    Parameter breakdown (vocab_size=50k):
    -------------------------------------
    Per layer:
      - Attention: 4 * 192^2 = 147,456
      - Router: 192 * 580 = 111,360
      - Experts: 3 * 192 * 768 * 580 = 256,573,440
      - Shared: 3 * 192 * 192 = 110,592
      - Total per layer: ~256.9M

    12 layers: ~3.08B
    Embeddings + LM head: ~19.2M
    Total: ~3.1B

    Active params:
      - Embeddings + LM: 19.2M
      - Attention: 0.15M * 12 = 1.77M
      - Router: 0.11M * 12 = 1.34M
      - Shared: 0.11M * 12 = 1.33M
      - Active expert (k=1): 3 * 192 * 768 * 1 * 12 = 5.31M
      - Total active: ~29M (~0.94%)

    Trade-offs vs other smallmoe configs:
    -------------------------------------
    - Much smaller d_model (192 vs 672-2048) limits model capacity per token
    - Larger experts (768 vs 32-256 hidden) may affect expert specialization
    - Very high total/active ratio (100:1) is at the extreme of MoE design
    - May require different training dynamics (higher LR, different warmup)
    """
    d_model = kwargs.pop("d_model", 192)
    n_heads = kwargs.pop("n_heads", 3)  # 192/3 = 64 dims per head
    n_layers = kwargs.pop("n_layers", 12)
    num_experts = kwargs.pop("num_experts", 580)
    top_k = kwargs.pop("top_k", 1)
    # INVERTED from granularity=12: use large experts to minimize router overhead
    expert_hidden = kwargs.pop("expert_hidden", d_model * 4)  # 768
    # Minimal shared MLP
    shared_hidden = kwargs.pop("shared_hidden", d_model)  # 192

    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        name=kwargs.pop("name", TransformerType.moe),
        block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        feed_forward_moe=MoEConfig(
            name=MoEType.default,
            num_experts=num_experts,
            hidden_size=expert_hidden,
            router=MoERouterConfig(top_k=top_k),
            shared_mlp=FeedForwardConfig(hidden_size=shared_hidden, bias=False),
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
        ),
        **kwargs,
    )
