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
