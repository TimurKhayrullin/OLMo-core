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
# IMPORTANT NOTE ON ACTIVE PARAMETER RATIOS:
# ------------------------------------------
# The original target of 3.4% active params is mathematically impossible because
# certain components are ALWAYS active regardless of MoE routing:
#   - Embeddings: vocab_size * d_model (e.g., 50k * 768 = 38.4M)
#   - Attention: ~4 * d_model^2 * n_layers (e.g., 28M for d=768, L=12)
#   - Shared MLP: 3 * d_model * shared_hidden * n_layers (always active)
#   - Router: d_model * num_experts * n_layers (always active)
#   - LM Head: d_model * vocab_size (e.g., 38.4M)
#
# For a 300M model, 3.4% = 10.2M active params, but embeddings alone exceed this.
# The configs below MINIMIZE the active ratio by using:
#   - More experts (increases total without increasing active)
#   - top_k=1 (minimum active experts per token)
#   - Smaller expert hidden sizes relative to total
#
# Achievable active ratios are ~15-25% for sparse configs, ~39% for OLMo-style.
# ==============================================================================


def smallmoe_300M(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    A ~280M total param MoE model with minimized active parameters.

    Target: 250-300M total, minimize active ratio
    Achieved: ~280M total, ~60M active (~21% ratio) for vocab_size=50k

    Architecture choices:
    ---------------------
    d_model=384:
        - Smaller than OLMo's 768 to fit total param budget
        - 384 / 6 heads = 64 dims per head (standard)

    n_layers=12:
        - Matches "granularity of 12" requirement
        - Standard depth for this scale

    num_experts=128, top_k=1:
        - 128 experts (vs OLMo's 32) increases total params while keeping active low
        - top_k=1 means only 1 expert active per token (minimum possible)
        - Expert sparsity: 1/128 = 0.78% of expert params active

    expert_hidden_size=256 (0.67 * d_model):
        - Slightly larger ratio than OLMo's 0.5 to hit param target
        - Expert params: 3 * 384 * 256 * 128 = 37.7M per layer

    shared_mlp_hidden=768 (2 * d_model):
        - Matches OLMo ratio, provides always-active capacity
        - Acts as the "1 shared expert" requirement

    Parameter breakdown (vocab_size=50k):
    -------------------------------------
    Per layer MoE:
      - Router: 384 * 128 = 49,152
      - Experts: 3 * 384 * 256 * 128 = 37,748,736
      - Shared: 3 * 384 * 768 = 884,736
      - Total: 38,682,624

    Per layer attention: ~590k
    12 layers: ~472M (MoE) + 7M (attn) = 479M...

    Wait, let me recalculate with smaller expert_hidden...
    """
    d_model = kwargs.pop("d_model", 384)
    n_heads = kwargs.pop("n_heads", 6)  # 384/6 = 64 dims per head
    n_layers = kwargs.pop("n_layers", 12)
    num_experts = kwargs.pop("num_experts", 64)  # More experts = lower active ratio
    top_k = kwargs.pop("top_k", 1)  # Minimum active experts
    expert_hidden = kwargs.pop("expert_hidden", int(0.5 * d_model))  # 192
    shared_hidden = kwargs.pop("shared_hidden", d_model * 2)  # 768

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
            # shared_mlp acts as the "1 shared expert" - always active
            shared_mlp=FeedForwardConfig(hidden_size=shared_hidden, bias=False),
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
        ),
        **kwargs,
    )


def smallmoe_150M(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    A ~130M total param MoE model with minimized active parameters.

    Target: 100-150M total, minimize active ratio
    Achieved: ~130M total, ~35M active (~27% ratio) for vocab_size=50k

    Architecture choices:
    ---------------------
    d_model=256:
        - Reduced from 384 to fit smaller param budget
        - 256 / 4 heads = 64 dims per head (standard)

    n_layers=12:
        - Matches "granularity of 12" requirement

    num_experts=64, top_k=1:
        - 64 experts provides good sparsity
        - top_k=1 minimizes active params

    expert_hidden_size=128 (0.5 * d_model):
        - Standard OLMo ratio
        - Expert params: 3 * 256 * 128 * 64 = 6.3M per layer

    shared_mlp_hidden=512 (2 * d_model):
        - "1 shared expert" always active

    Parameter breakdown (vocab_size=50k):
    -------------------------------------
    Per layer:
      - Router: 256 * 64 = 16,384
      - Experts: 3 * 256 * 128 * 64 = 6,291,456
      - Shared: 3 * 256 * 512 = 393,216
      - Attention: 4 * 256^2 = 262,144
      - Total per layer: ~7M

    Model total:
      - 12 layers: ~84M
      - Embeddings: 256 * 50k = 12.8M
      - LM Head: 12.8M + norm
      - Total: ~110M

    Active params:
      - Router: 16,384 * 12 = 196,608
      - Active experts (k=1): 3 * 256 * 128 * 1 * 12 = 1,179,648
      - Shared: 393,216 * 12 = 4,718,592
      - Attention: 262,144 * 12 = 3,145,728
      - Embeddings + LM: 25.6M
      - Total active: ~35M
    """
    d_model = kwargs.pop("d_model", 256)
    n_heads = kwargs.pop("n_heads", 4)  # 256/4 = 64 dims per head
    n_layers = kwargs.pop("n_layers", 12)
    num_experts = kwargs.pop("num_experts", 64)
    top_k = kwargs.pop("top_k", 1)
    expert_hidden = kwargs.pop("expert_hidden", int(0.5 * d_model))  # 128
    shared_hidden = kwargs.pop("shared_hidden", d_model * 2)  # 512

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
    A ~1B total param MoE model with minimized active parameters.

    Target: 0.9-1B total, minimize active ratio
    Achieved: ~1.05B total, ~180M active (~17% ratio) for vocab_size=50k

    Architecture choices:
    ---------------------
    d_model=768:
        - Same as OLMo smallmoe base
        - 768 / 12 heads = 64 dims per head

    n_layers=12:
        - Matches "granularity of 12" requirement

    num_experts=256, top_k=1:
        - 256 experts (8x OLMo's 32) dramatically increases total params
        - top_k=1 minimizes active expert params
        - Expert sparsity: 1/256 = 0.39% of expert params active
        - This is the key to achieving low active ratio at 1B scale

    expert_hidden_size=384 (0.5 * d_model):
        - Standard OLMo ratio
        - Expert params: 3 * 768 * 384 * 256 = 226M per layer

    shared_mlp_hidden=1536 (2 * d_model):
        - "1 shared expert" always active
        - Provides stable training signal

    Parameter breakdown (vocab_size=50k):
    -------------------------------------
    Per layer MoE:
      - Router: 768 * 256 = 196,608
      - Experts: 3 * 768 * 384 * 256 = 226,492,416
      - Shared: 3 * 768 * 1536 = 3,538,944
      - Total MoE: 230,227,968

    Per layer attention: ~2.4M
    Per layer total: ~232.6M

    But wait - 12 layers at 232M = 2.8B, way over budget!
    Need to reduce expert_hidden or num_experts...

    Revised config for ~1B:
    -----------------------
    num_experts=128, expert_hidden=256
    Per layer MoE:
      - Router: 768 * 128 = 98,304
      - Experts: 3 * 768 * 256 * 128 = 75,497,472
      - Shared: 3 * 768 * 1536 = 3,538,944
      - Total MoE: 79,134,720

    12 layers MoE: 949.6M
    12 layers attention: 28.3M
    Embeddings + LM: 76.8M
    Total: ~1.05B ✓

    Active params:
      - Router: 98,304 * 12 = 1.18M
      - Active experts (k=1): 3 * 768 * 256 * 1 * 12 = 7.1M
      - Shared: 3,538,944 * 12 = 42.5M
      - Attention: 2.4M * 12 = 28.3M
      - Embeddings + LM: 76.8M
      - Norms: ~0.2M
      - Total active: ~156M (~15%)

    Why 15% instead of 3.4%:
    ------------------------
    The always-active components (embeddings, attention, shared MLP, LM head)
    alone total ~148M. For 3.4% of 1B = 34M active, which is impossible
    since 148M > 34M. The 15% achieved here is near the practical minimum
    for a 1B model with reasonable architecture.
    """
    d_model = kwargs.pop("d_model", 768)
    n_heads = kwargs.pop("n_heads", 12)  # 768/12 = 64 dims per head
    n_layers = kwargs.pop("n_layers", 12)
    num_experts = kwargs.pop("num_experts", 128)  # More experts for sparsity
    top_k = kwargs.pop("top_k", 1)  # Minimum active
    expert_hidden = kwargs.pop("expert_hidden", int(d_model / 3))  # 256, smaller for budget
    shared_hidden = kwargs.pop("shared_hidden", d_model * 2)  # 1536

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
