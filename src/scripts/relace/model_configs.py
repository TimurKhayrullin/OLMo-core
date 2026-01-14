"""
Custom model configurations for Relace experiments.

Usage:
    from scripts.relace.model_configs import llama2_135M

    model_config = llama2_135M(vocab_size=tokenizer_config.padded_vocab_size())
"""

from olmo_core.nn.transformer import TransformerConfig


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
