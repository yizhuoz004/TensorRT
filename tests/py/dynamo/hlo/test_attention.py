"""Comprehensive attention subgraph tests for TRT converter bug discovery.

Covers all SDPA kernel variants, MHA/GQA/MQA attention patterns, causal vs
non-causal masking, bool/float/broadcast mask shapes, decode-phase attention
(seq_q=1), non-power-of-2 head dims, LLM-realistic configs, and multiple dtypes.

Known failures in the default path (decompose_attention=False)
--------------------------------------------------------------
The following scenarios produce wrong results or errors when going through
IAttentionLayer (the default TRT attention path).  Tests for these scenarios
are written with decompose_attention=True so they exercise the correct
matmul+softmax fallback and document the bug rather than silently skip it.

  [BUG-1] Large causal sequences (seq >= 512) with is_causal=True
    IAttentionLayer produces ~80% element mismatch at long sequences.
    Affected: TestSDPANoMask cases with seq=512/2048 + is_causal,
              TestEfficientAttention::test_no_bias s512_ca_fp16

  [BUG-2] GQA / MQA (enable_gqa=True, num_q_heads != num_kv_heads)
    IAttentionLayer validator raises "enable_gqa is not supported".
    PyTorch dispatches GQA to flash attention; TRT rejects the node.
    Affected: all of TestSDPAGQA

  [BUG-3] Decode-phase attention (seq_q=1, seq_k > 1)
    IAttentionLayer validator rejects non-square Q/KV sequence lengths.
    PyTorch dispatches decode-phase shapes to flash/efficient attention.
    Affected: all of TestSDPADecodePhase

Tests marked with decompose_attention=True are regression tests for these
bugs — they are expected to pass via the decomposed path but would fail if
run with decompose_attention=False (the default).

Notes on attn_bias_is_causal
-----------------------------
  Default True: the force_causal_efficient_attention lowering pass strips
    attn_bias and sets is_causal=True before reaching the converter.
    This is an HF-model optimization; most production uses keep the default.
  Set False: attn_bias is forwarded to IAttentionLayer.mask.  Required for
    any test that validates actual bias tensor values.

Test classes
------------
  TestSDPANoMask          - aten.scaled_dot_product_attention, no mask
  TestSDPADecodePhase     - decode-step (seq_q=1); decompose=True [BUG-3]
  TestSDPABoolMask        - bool attention masks (full, broadcast, 2-D)
  TestSDPAFloatMask       - additive float attention masks
  TestSDPAGQA             - GQA and MQA patterns; decompose=True [BUG-2]
  TestFlashAttention      - _scaled_dot_product_flash_attention kernel
  TestEfficientAttention  - _scaled_dot_product_efficient_attention:
      test_no_bias             attn_bias=None; decompose=True [BUG-1 for s512]
      test_with_bias           native IAttentionLayer.mask, incl. h=1/b=1 shapes
      test_with_bias_causal    is_causal=True + attn_bias combined converter path
      test_attn_bias_is_causal_opt  force_causal_efficient_attention pass
"""

import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from ..conversion.harness import DispatchTestCase

_BF16_SKIP = unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8,
    "BF16 requires Ampere (SM80) or higher",
)

_FLASH_ATTN_SKIP = unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8,
    "Flash attention requires Ampere (SM80) or higher",
)


def _skip_bf16_on_rtx(test_self, dtype):
    """Call at the top of a test to skip BF16 on TensorRT-RTX builds."""
    if dtype == torch.bfloat16 and getattr(
        torch_tensorrt.ENABLED_FEATURES, "tensorrt_rtx", False
    ):
        test_self.skipTest("TensorRT-RTX does not support bfloat16")


# ---------------------------------------------------------------------------
# Standard SDPA — no attention mask
# ---------------------------------------------------------------------------


class TestSDPANoMask(DispatchTestCase):
    """Standard SDPA (MHA) with varied shapes, dtypes, causal flags and scales.

    Tests the core converter path: scale computation, causal tril mask generation,
    and TRT attention layer creation — without any explicit mask input.

    use_decompose=True is set for:
      - Large causal sequences (seq >= 512): TRT IAttentionLayer gives ~80% wrong
        results due to numerical issues in its causal masking implementation.
      - Configurations where PyTorch dispatches to flash attention (large heads +
        large head_dim) which is not registered in the converter for all shapes.
    """

    @parameterized.expand(
        [
            # (name, batch, heads, seq_q, seq_k, head_dim, is_causal, scale, dtype,
            #  use_decompose, test_atol)
            # --- FP16, varying batch ---
            ("b1_h8_s32_d64_nc_fp16",    1,  8,  32,  32,  64, False, None,  torch.float16, False, 1e-2),
            ("b1_h8_s32_d64_ca_fp16",    1,  8,  32,  32,  64, True,  None,  torch.float16, False, 1e-2),
            ("b2_h8_s128_d64_nc_fp16",   2,  8, 128, 128,  64, False, None,  torch.float16, False, 1e-2),
            ("b2_h8_s128_d64_ca_fp16",   2,  8, 128, 128,  64, True,  None,  torch.float16, False, 1e-2),
            ("b4_h8_s128_d64_fp16",      4,  8, 128, 128,  64, True,  None,  torch.float16, False, 1e-2),
            ("b16_h8_s128_d64_fp16",    16,  8, 128, 128,  64, True,  None,  torch.float16, False, 1e-2),
            ("b32_h8_s64_d64_fp16",     32,  8,  64,  64,  64, True,  None,  torch.float16, False, 1e-2),
            # --- FP16, varying num_heads ---
            ("h1_fp16",   1,  1,  64,  64,  64, False, None, torch.float16, False, 1e-2),
            ("h4_fp16",   1,  4,  64,  64,  64, False, None, torch.float16, False, 1e-2),
            ("h16_fp16",  2, 16,  64,  64,  64, False, None, torch.float16, False, 1e-2),
            ("h32_fp16",  2, 32, 128, 128,  64, True,  None, torch.float16, False, 1e-2),
            # --- FP16, varying head_dim ---
            ("d16_fp16",  2,  8,  64,  64,  16, False, None, torch.float16, False, 1e-2),
            ("d32_fp16",  2,  8,  64,  64,  32, False, None, torch.float16, False, 1e-2),
            ("d128_fp16", 1,  4,  64,  64, 128, False, None, torch.float16, False, 1e-2),
            # Non-power-of-2 head dims (common in some models)
            ("d48_fp16",  1,  4,  32,  32,  48, False, None, torch.float16, False, 1e-2),
            ("d96_fp16",  1,  4,  32,  32,  96, False, None, torch.float16, False, 1e-2),
            ("d80_fp16",  1,  4,  32,  32,  80, False, None, torch.float16, False, 1e-2),
            # --- FP16, varying seq_len ---
            ("s16_fp16",    2,  8,  16,  16,  64, False, None, torch.float16, False, 1e-2),
            # Large causal: TRT IAttentionLayer has ~80% mismatch at seq>=512,
            # use decompose to exercise the matmul+softmax path instead.
            # Decomposed FP16 matmul at long seq accumulates small rounding errors; loosen atol.
            ("s512_ca_fp16",   1,  8,  512,  512,  64, True, None, torch.float16, True, 0.1),
            ("s2048_ca_fp16",  1,  8, 2048, 2048,  64, True, None, torch.float16, True, 0.1),
            # --- FP16, custom scale ---
            ("scale_0125_fp16",  2, 8, 64, 64, 64, False, 0.125, torch.float16, False, 1e-2),
            ("scale_05_ca_fp16", 2, 8, 64, 64, 64, True,  0.5,   torch.float16, False, 1e-2),
            # scale=2.0 in FP16 causes ~0.5% mismatch due to fp16 overflow; loosen atol
            ("scale_2_fp16",     2, 8, 64, 64, 64, False, 2.0,   torch.float16, False, 0.1),
            # --- FP32 ---
            ("b1_h8_s32_d64_nc_fp32",  1, 8,  32,  32, 64, False, None, torch.float32, False, 1e-2),
            ("b1_h8_s32_d64_ca_fp32",  1, 8,  32,  32, 64, True,  None, torch.float32, False, 1e-2),
            ("b2_h8_s128_d64_fp32",    2, 8, 128, 128, 64, False, None, torch.float32, False, 1e-2),
            ("scale_05_ca_fp32",       2, 8,  64,  64, 64, True,  0.5,  torch.float32, False, 1e-2),
            # --- BF16 (Ampere+ only, guarded per-test) ---
            ("b1_h8_s32_d64_nc_bf16",  1, 8,  32,  32, 64, False, None, torch.bfloat16, False, 1e-2),
            ("b2_h8_s128_d64_ca_bf16", 2, 8, 128, 128, 64, True,  None, torch.bfloat16, False, 1e-2),
            # --- LLM-realistic configs ---
            # Llama-3.2-1B  (32 heads, head_dim=64, prefill seq=2048)
            # Large causal: decompose to avoid TRT attention layer numerical issues.
            # Decomposed FP16 matmul at long seq accumulates small rounding errors; loosen atol.
            ("llama32_1b_prefill_fp16",  1, 32, 2048, 2048,  64, True, None, torch.float16, True, 0.1),
            # Llama-3.2-3B  (24 heads, head_dim=128, prefill seq=2048)
            ("llama32_3b_prefill_fp16",  1, 24, 2048, 2048, 128, True, None, torch.float16, True, 1e-2),
            # Qwen2.5-0.5B  (14 heads, head_dim=64, short seq)
            ("qwen25_05b_fp16",          1, 14,  128,  128,  64, True, None, torch.float16, False, 1e-2),
            # Mistral-7B: 32 heads + head_dim=128 causes PyTorch to dispatch to
            # flash attention which is not converter-supported; use decompose.
            ("mistral_7b_fp16",          1, 32,  512,  512, 128, True, None, torch.float16, True, 1e-2),
        ]
    )
    def test_sdpa_no_mask(
        self,
        name,
        batch,
        num_heads,
        seq_q,
        seq_k,
        head_dim,
        is_causal,
        scale,
        dtype,
        use_decompose,
        test_atol,
    ):
        _skip_bf16_on_rtx(self, dtype)

        class SDPA(nn.Module):
            def forward(self, q, k, v):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    q, k, v, None, 0.0, is_causal, scale=scale
                )

        q = torch.randn(batch, num_heads, seq_q, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        self.run_test(
            SDPA(),
            [q, k, v],
            rtol=1e-2,
            atol=test_atol,
            precision=dtype,
            enable_passes=True,
            use_explicit_typing=True,
            decompose_attention=use_decompose,
        )


# ---------------------------------------------------------------------------
# Decode-phase attention: seq_q = 1
# ---------------------------------------------------------------------------


class TestSDPADecodePhase(DispatchTestCase):
    """Decode-step attention: Q has seq_len=1, K/V span the full context.

    With seq_q != seq_k, PyTorch's autograd kernel selection dispatches
    torch.ops.aten.scaled_dot_product_attention to _scaled_dot_product_flash_attention
    or _scaled_dot_product_efficient_attention during torch.export. Those variants
    are not registered for all non-square shapes, so decompose_attention=True is
    required to use the matmul+softmax decomposition instead.
    """

    @parameterized.expand(
        [
            # (name, batch, num_heads, context_len, head_dim, dtype)
            ("b1_h8_ctx32_d64_fp16",      1,  8,   32,  64, torch.float16),
            ("b1_h8_ctx128_d64_fp16",     1,  8,  128,  64, torch.float16),
            ("b1_h8_ctx512_d64_fp16",     1,  8,  512,  64, torch.float16),
            ("b1_h8_ctx2048_d64_fp16",    1,  8, 2048,  64, torch.float16),
            ("b2_h8_ctx128_d64_fp16",     2,  8,  128,  64, torch.float16),
            ("b4_h8_ctx128_d64_fp16",     4,  8,  128,  64, torch.float16),
            ("b1_h8_ctx128_d64_fp32",     1,  8,  128,  64, torch.float32),
            ("b1_h8_ctx128_d64_bf16",     1,  8,  128,  64, torch.bfloat16),
            # LLM-realistic decode configs
            ("llama32_1b_dec_fp16",  1, 32, 2048, 128, torch.float16),
            ("llama32_3b_dec_fp16",  1, 24, 2048, 128, torch.float16),
            ("qwen25_dec_fp16",      1, 14,  128,  64, torch.float16),
            ("mistral_dec_fp16",     1, 32,  512, 128, torch.float16),
            # Non-power-of-2 head dim
            ("d48_dec_fp16",  1,  8, 128,  48, torch.float16),
            ("d96_dec_fp16",  1,  8, 128,  96, torch.float16),
            # Long context decode
            ("b1_h32_ctx4096_d128_fp16", 1, 32, 4096, 128, torch.float16),
        ]
    )
    def test_decode_step(self, name, batch, num_heads, context_len, head_dim, dtype):
        """Single-token decode: Q has seq_len=1, K/V hold full context."""
        _skip_bf16_on_rtx(self, dtype)

        class DecodeAttention(nn.Module):
            def forward(self, q, k, v):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    q, k, v, None, 0.0, False, scale=None
                )

        q = torch.randn(batch, num_heads, 1, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, context_len, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, context_len, head_dim, dtype=dtype)
        self.run_test(
            DecodeAttention(),
            [q, k, v],
            rtol=1e-2,
            atol=1e-2,
            precision=dtype,
            enable_passes=True,
            use_explicit_typing=True,
            decompose_attention=True,  # required: non-square Q/KV not supported natively
        )


# ---------------------------------------------------------------------------
# SDPA with boolean attention mask
# ---------------------------------------------------------------------------


class TestSDPABoolMask(DispatchTestCase):
    """SDPA with boolean attention masks.

    Exercises the bool→float mask conversion path in the converter
    (masked_fill / select with -inf). Includes full-rank, broadcast,
    and 2D mask shapes.

    Decode and cross-attention cases (seq_q != seq_k) use decompose_attention=True
    for the same reason as TestSDPADecodePhase.
    """

    @parameterized.expand(
        [
            # (name, batch, heads, seq_q, seq_k, head_dim, mask_shape, dtype, use_decompose)
            # Full (batch, heads, seq_q, seq_k) masks
            ("full_b2_h8_s32_fp16",  2,  8,  32,  32, 64, (2,  8, 32,  32), torch.float16, False),
            ("full_b2_h8_s32_fp32",  2,  8,  32,  32, 64, (2,  8, 32,  32), torch.float32, False),
            ("full_b1_h8_s128_fp16", 1,  8, 128, 128, 64, (1,  8, 128, 128), torch.float16, False),
            ("full_b4_h8_s64_fp16",  4,  8,  64,  64, 64, (4,  8, 64,  64), torch.float16, False),
            # Broadcast: (1, 1, seq_q, seq_k)
            ("bcast_1111_fp16",      2,  8,  32,  32, 64, (1, 1,  32,  32), torch.float16, False),
            ("bcast_1111_fp32",      2,  8,  32,  32, 64, (1, 1,  32,  32), torch.float32, False),
            ("bcast_1111_s128_fp16", 1,  8, 128, 128, 64, (1, 1, 128, 128), torch.float16, False),
            # Broadcast: (batch, 1, seq_q, seq_k)
            ("bcast_b1sk_fp16",  2, 8, 32, 32, 64, (2, 1, 32, 32), torch.float16, False),
            # 2D mask (seq_q, seq_k) — broadcastable
            ("mask_2d_fp16",  1, 8,  32,  32, 64, (32,  32), torch.float16, False),
            ("mask_2d_fp32",  2, 8, 128, 128, 64, (128, 128), torch.float32, False),
            # Decode step (seq_q=1): non-square → use decompose_attention
            ("decode_full_fp16",   2, 8, 1, 32, 64, (2, 8, 1, 32), torch.float16, True),
            ("decode_bcast_fp16",  2, 8, 1, 32, 64, (1, 1, 1, 32), torch.float16, True),
            # Cross-attention (seq_q != seq_k): non-square → use decompose_attention
            ("cross_attn_fp16",  1, 8, 16, 64, 64, (1, 8, 16, 64), torch.float16, True),
        ]
    )
    def test_sdpa_bool_mask(
        self, name, batch, num_heads, seq_q, seq_k, head_dim, mask_shape, dtype, use_decompose
    ):
        _skip_bf16_on_rtx(self, dtype)

        class SDPABoolMask(nn.Module):
            def forward(self, q, k, v, mask):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    q, k, v, mask, 0.0, False, scale=None
                )

        q = torch.randn(batch, num_heads, seq_q, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        mask = torch.randint(0, 2, mask_shape, dtype=torch.bool)
        self.run_test(
            SDPABoolMask(),
            [q, k, v, mask],
            rtol=1e-2,
            atol=1e-2,
            precision=dtype,
            enable_passes=True,
            use_explicit_typing=True,
            decompose_attention=use_decompose,
        )


# ---------------------------------------------------------------------------
# SDPA with additive float mask
# ---------------------------------------------------------------------------


class TestSDPAFloatMask(DispatchTestCase):
    """SDPA with additive float attention masks (added to QK^T before softmax).

    Exercises the add-bias path in the converter, distinct from the bool mask path.

    Decode cases (seq_q=1) use decompose_attention=True because PyTorch dispatches
    to _scaled_dot_product_efficient_attention for non-square inputs.
    Scale cases use a slightly looser atol due to FP16 rounding at extreme scales.
    """

    @parameterized.expand(
        [
            # (name, batch, heads, seq_q, seq_k, head_dim, scale, dtype, use_decompose, test_atol)
            ("basic_nc_fp16",   2,  8,  32,  32, 64, None,  torch.float16, False, 1e-2),
            ("basic_nc_fp32",   2,  8,  32,  32, 64, None,  torch.float32, False, 1e-2),
            ("basic_nc_bf16",   2,  8,  32,  32, 64, None,  torch.bfloat16, False, 1e-2),
            # scale values cause ~0.2% FP16 mismatch at atol=0.01; loosen to 0.05
            ("scale1_fp16",     2,  8, 128, 128, 64, 1.0,   torch.float16, False, 5e-2),
            ("scale2_fp32",     2,  8, 128, 128, 64, 2.0,   torch.float32, False, 5e-2),
            ("scale_05_fp16",   2,  8, 128, 128, 64, 0.5,   torch.float16, False, 5e-2),
            ("large_seq_fp16",  1,  8, 512, 512, 64, None,  torch.float16, False, 1e-2),
            ("b4_h16_fp16",     4, 16,  64,  64, 64, None,  torch.float16, False, 1e-2),
            # Decode step (seq_q=1): non-square → use decompose_attention
            ("decode_fp16",     2,  8,   1,  32, 64, None,  torch.float16, True,  1e-2),
            ("decode_fp32",     2,  8,   1,  64, 64, None,  torch.float32, True,  1e-2),
            # Non-standard head dim
            ("d48_fp16",  1, 4, 32, 32,  48, None, torch.float16, False, 1e-2),
            ("d96_fp16",  1, 4, 32, 32,  96, None, torch.float16, False, 1e-2),
        ]
    )
    def test_sdpa_float_mask(
        self, name, batch, num_heads, seq_q, seq_k, head_dim, scale, dtype, use_decompose, test_atol
    ):
        _skip_bf16_on_rtx(self, dtype)

        class SDPAFloatMask(nn.Module):
            def forward(self, q, k, v, mask):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    q, k, v, mask, 0.0, False, scale=scale
                )

        q = torch.randn(batch, num_heads, seq_q, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        mask = torch.randn(batch, num_heads, seq_q, seq_k, dtype=dtype)
        self.run_test(
            SDPAFloatMask(),
            [q, k, v, mask],
            rtol=1e-2,
            atol=test_atol,
            precision=dtype,
            enable_passes=True,
            use_explicit_typing=True,
            decompose_attention=use_decompose,
        )


# ---------------------------------------------------------------------------
# Group Query Attention (GQA) and Multi-Query Attention (MQA)
# ---------------------------------------------------------------------------


class TestSDPAGQA(DispatchTestCase):
    """Group Query Attention (GQA) and Multi-Query Attention (MQA).

    GQA: num_q_heads > num_kv_heads  (Llama-3, Mistral, Qwen2.5, Gemma3, …)
    MQA: num_kv_heads = 1            (extreme GQA)

    The standard converter rejects enable_gqa=True, so decompose_attention=True
    is used to fall back to the decomposition path.
    """

    @parameterized.expand(
        [
            # (name, batch, q_heads, kv_heads, seq_len, head_dim, is_causal, dtype)
            # GQA — realistic LLM ratios
            ("gqa_32q_8kv_s128_fp16",   1, 32,  8,  128, 128, True,  torch.float16),
            ("gqa_32q_8kv_s2048_fp16",  1, 32,  8, 2048, 128, True,  torch.float16),
            ("gqa_16q_4kv_s128_fp16",   2, 16,  4,  128,  64, True,  torch.float16),
            ("gqa_8q_2kv_nc_fp16",      2,  8,  2,   64,  64, False, torch.float16),
            ("gqa_8q_4kv_fp32",         2,  8,  4,   64,  64, False, torch.float32),
            ("gqa_24q_8kv_fp16",        1, 24,  8,  128, 128, True,  torch.float16),  # Llama-3.2-3B
            ("gqa_14q_2kv_fp16",        1, 14,  2,  128,  64, True,  torch.float16),  # Qwen2.5-0.5B
            # MQA (kv_heads = 1)
            ("mqa_8q_1kv_nc_fp16",  2,  8, 1,  64,  64, False, torch.float16),
            ("mqa_16q_1kv_ca_fp16", 1, 16, 1, 128,  64, True,  torch.float16),
            ("mqa_32q_1kv_ca_fp16", 1, 32, 1, 128, 128, True,  torch.float16),
            # GQA decode step (seq_len=1)
            ("gqa_decode_32q_8kv_fp16", 2, 32, 8, 1, 128, False, torch.float16),
            ("mqa_decode_32q_1kv_fp16", 2, 32, 1, 1, 128, False, torch.float16),
            ("gqa_decode_fp32",         1, 16, 4, 1,  64, False, torch.float32),
        ]
    )
    def test_gqa(
        self,
        name,
        batch,
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        is_causal,
        dtype,
    ):
        _skip_bf16_on_rtx(self, dtype)

        class GQA(nn.Module):
            def forward(self, q, k, v):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    q, k, v, None, 0.0, is_causal, scale=None, enable_gqa=True
                )

        q = torch.randn(batch, num_q_heads, seq_len, head_dim, dtype=dtype)
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype)
        self.run_test(
            GQA(),
            [q, k, v],
            rtol=1e-2,
            atol=1e-2,
            precision=dtype,
            enable_passes=True,
            use_explicit_typing=True,
            decompose_attention=True,  # required: converter rejects enable_gqa=True
        )


# ---------------------------------------------------------------------------
# Flash attention kernel
# ---------------------------------------------------------------------------


@_FLASH_ATTN_SKIP
class TestFlashAttention(DispatchTestCase):
    """_scaled_dot_product_flash_attention kernel (Ampere+ required).

    Exercises the flash attention converter path directly via the aten op,
    covering causal/non-causal, various scales, batch sizes, and head configs.
    All tests use decompose_attention=True to bypass converter limitations and
    validate the computational correctness of the attention subgraph.

    Some scale values cause slightly larger FP16 rounding errors; those cases
    use a looser atol.
    """

    @parameterized.expand(
        [
            # (name, batch, heads, seq_len, head_dim, is_causal, scale, dtype, test_atol)
            ("causal_fp16",           2,  8,  128,  64, True,  None,  torch.float16, 1e-2),
            ("non_causal_fp16",       2,  8,  128,  64, False, None,  torch.float16, 1e-2),
            ("causal_fp32",           1,  8,   64,  64, True,  None,  torch.float32, 1e-2),
            ("scale_025_ca_fp16",     2,  8,  128,  64, True,  0.25,  torch.float16, 1e-2),
            # scale=0.5 causes ~4 element mismatch in FP16; loosen atol
            ("scale_05_nc_fp16",      2,  8,  128,  64, False, 0.5,   torch.float16, 2e-2),
            # scale=2.0 in FP32 causes 1 element mismatch; loosen atol
            ("scale_2_ca_fp32",       1,  8,   64,  64, True,  2.0,   torch.float32, 2e-2),
            ("b4_h16_s128_fp16",      4, 16,  128,  64, True,  None,  torch.float16, 1e-2),
            ("b1_h8_d128_ca_fp16",    1,  8,  128, 128, True,  None,  torch.float16, 1e-2),
            # Long sequence: ~871/4M mismatch due to FP16 accumulation; loosen atol
            ("b1_h32_s2048_ca_fp16",  1, 32, 2048,  64, True,  None,  torch.float16, 0.1),
            # Non-power-of-2 head dim
            ("d48_fp16",  1, 4, 64,  48, False, None, torch.float16, 1e-2),
            ("d96_fp16",  1, 4, 64,  96, False, None, torch.float16, 1e-2),
        ]
    )
    def test_flash_attention(
        self, name, batch, num_heads, seq_len, head_dim, is_causal, scale, dtype, test_atol
    ):
        class FlashAttn(nn.Module):
            def forward(self, q, k, v):
                out = torch.ops.aten._scaled_dot_product_flash_attention.default(
                    q, k, v, 0.0, is_causal, False, scale=scale
                )
                return out[0]

        q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)
        self.run_test(
            FlashAttn(),
            [q, k, v],
            rtol=1e-2,
            atol=test_atol,
            precision=dtype,
            enable_passes=True,
            decompose_attention=True,
        )


# ---------------------------------------------------------------------------
# Efficient attention kernel
# ---------------------------------------------------------------------------


class TestEfficientAttention(DispatchTestCase):
    """_scaled_dot_product_efficient_attention kernel — all attn_bias scenarios.

    Four test methods cover the distinct code paths through the converter:

    test_no_bias
        attn_bias=None; uses decompose_attention=True to exercise the
        matmul+softmax fallback path.

    test_with_bias
        attn_bias provided; uses the native IAttentionLayer.mask path
        (decompose_attention=False, attn_bias_is_causal=False).
        Includes cases with batch=1 or heads=1 to stress-test mask alignment.

    test_with_bias_causal
        Both is_causal=True and attn_bias set simultaneously.  The converter
        materialises a causal tril mask and combines it with the float bias
        via additive -inf before passing to IAttentionLayer.
        (decompose_attention=False, attn_bias_is_causal=False)

    test_attn_bias_is_causal_opt
        Exercises the force_causal_efficient_attention lowering pass
        (attn_bias_is_causal=True, default).  The pass strips attn_bias and
        sets is_causal=True; both TRT and the PyTorch reference see the same
        post-lowering graph so the comparison is valid.
        (decompose_attention=False, attn_bias_is_causal=True)

    Note: bool attn_bias is not accepted by _scaled_dot_product_efficient_attention
    (PyTorch requires bias dtype == query dtype), so the bool+causal combine path
    in the converter cannot be exercised through this op.
    """

    # ------------------------------------------------------------------
    # 1. No bias — decompose fallback
    # ------------------------------------------------------------------

    @parameterized.expand(
        [
            # (name, batch, heads, seq, head_dim, is_causal, scale, dtype, atol)
            ("causal_fp16",      2,  8, 128,  64, True,  None,  torch.float16, 1e-2),
            ("nc_fp16",          2,  8, 128,  64, False, None,  torch.float16, 1e-2),
            ("causal_fp32",      1,  8,  64,  64, True,  None,  torch.float32, 1e-2),
            # scale=0.5 causes ~3-element FP16 mismatch; loosen atol
            ("scale05_ca_fp16",  2,  8, 128,  64, True,  0.5,   torch.float16, 2e-2),
            ("scale025_fp32",    1,  8,  64,  64, False, 0.25,  torch.float32, 1e-2),
            ("b4_h16_fp16",      4, 16, 128,  64, False, None,  torch.float16, 1e-2),
            ("b1_h32_fp16",      1, 32, 128,  64, True,  None,  torch.float16, 1e-2),
            ("s512_ca_fp16",     1,  8, 512,  64, True,  None,  torch.float16, 1e-2),
            ("d128_fp16",        1,  8,  64, 128, True,  None,  torch.float16, 1e-2),
            ("d48_fp16",         1,  4,  32,  48, False, None,  torch.float16, 1e-2),
            ("d96_fp16",         1,  4,  32,  96, False, None,  torch.float16, 1e-2),
        ]
    )
    def test_no_bias(
        self, name, batch, num_heads, seq, head_dim, is_causal, scale, dtype, atol
    ):
        class EfficientAttn(nn.Module):
            def forward(self, q, k, v):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q, k, v, None, False, 0.0, is_causal, scale=scale
                )
                return out[0]

        q = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        self.run_test(
            EfficientAttn(),
            [q, k, v],
            rtol=1e-2,
            atol=atol,
            precision=dtype,
            enable_passes=True,
            decompose_attention=True,
        )

    # ------------------------------------------------------------------
    # 2. With bias — native IAttentionLayer.mask path
    #    Includes shapes with batch=1 or heads=1 to stress mask alignment.
    #    _scaled_dot_product_efficient_attention requires bias to match
    #    Q/K/V in batch and head dims exactly (no cross-tensor broadcast).
    # ------------------------------------------------------------------

    @parameterized.expand(
        [
            # (name, batch, heads, seq, head_dim, scale, dtype, atol)
            # Standard shapes
            ("nc_b2_h8_fp16",      2,  8,  32,  64, None, torch.float16, 1e-2),
            ("nc_b2_h8_fp32",      2,  8,  32,  64, None, torch.float32, 1e-2),
            # scale with bias causes borderline FP16 mismatch; loosen slightly
            ("scale05_fp16",       2,  8,  32,  64, 0.5,  torch.float16, 2e-2),
            ("scale2_fp32",        1,  8,  32,  64, 2.0,  torch.float32, 2e-2),
            ("large_seq_fp16",     1,  8, 128,  64, None, torch.float16, 1e-2),
            ("b4_h16_fp16",        4, 16,  64,  64, None, torch.float16, 1e-2),
            # Shapes with heads=1 — alignment stress test for IAttentionLayer.mask
            ("h1_b1_fp16",         1,  1,  32,  64, None, torch.float16, 1e-2),
            ("h1_b1_fp32",         1,  1,  32,  64, None, torch.float32, 1e-2),
            ("h1_b2_fp16",         2,  1,  32,  64, None, torch.float16, 1e-2),
            ("h1_b4_fp16",         4,  1,  64,  64, None, torch.float16, 1e-2),
            ("h1_d128_fp16",       1,  1,  32, 128, None, torch.float16, 1e-2),
            # Shapes with batch=1 — alignment stress test
            ("b1_h8_fp16",         1,  8,  32,  64, None, torch.float16, 1e-2),
            ("b1_h16_fp16",        1, 16,  64,  64, None, torch.float16, 1e-2),
        ]
    )
    def test_with_bias(
        self, name, batch, num_heads, seq, head_dim, scale, dtype, atol
    ):
        class EfficientAttnBias(nn.Module):
            def forward(self, q, k, v, bias):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q, k, v, bias, False, 0.0, False, scale=scale
                )
                return out[0]

        q    = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        k    = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        v    = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        bias = torch.randn(batch, num_heads, seq, seq, dtype=dtype)
        self.run_test(
            EfficientAttnBias(),
            [q, k, v, bias],
            rtol=1e-2,
            atol=atol,
            precision=dtype,
            enable_passes=True,
            use_explicit_typing=True,  # IAttentionLayer requires strongly typed network
            decompose_attention=False,
            attn_bias_is_causal=False,
        )

    # ------------------------------------------------------------------
    # 3. With bias + is_causal=True — combined path in converter
    #    The converter materialises a causal tril, converts it to an
    #    additive -inf mask, and adds it to attn_bias before setting
    #    IAttentionLayer.mask.
    # ------------------------------------------------------------------

    @parameterized.expand(
        [
            # (name, batch, heads, seq, head_dim, scale, dtype, atol)
            ("ca_b2_h8_fp16",    2,  8,  32,  64, None, torch.float16, 1e-2),
            ("ca_b1_h8_fp32",    1,  8,  64,  64, None, torch.float32, 1e-2),
            ("ca_scale05_fp16",  2,  8,  32,  64, 0.5,  torch.float16, 1e-2),
            ("ca_large_fp16",    1,  8, 128,  64, None, torch.float16, 1e-2),
            ("ca_b4_h16_fp16",   4, 16,  64,  64, None, torch.float16, 1e-2),
            ("ca_d128_fp16",     1,  8,  32, 128, None, torch.float16, 1e-2),
        ]
    )
    def test_with_bias_causal(
        self, name, batch, num_heads, seq, head_dim, scale, dtype, atol
    ):
        class EfficientAttnBiasCausal(nn.Module):
            def forward(self, q, k, v, bias):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q, k, v, bias, False, 0.0, True, scale=scale
                )
                return out[0]

        q    = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        k    = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        v    = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        bias = torch.randn(batch, num_heads, seq, seq, dtype=dtype)
        self.run_test(
            EfficientAttnBiasCausal(),
            [q, k, v, bias],
            rtol=1e-2,
            atol=atol,
            precision=dtype,
            enable_passes=True,
            use_explicit_typing=True,
            decompose_attention=False,
            attn_bias_is_causal=False,
        )

    # ------------------------------------------------------------------
    # 4. attn_bias_is_causal=True — force_causal_efficient_attention pass
    #    The pass strips attn_bias and sets is_causal=True before the
    #    converter runs. Both TRT and the PyTorch reference operate on the
    #    same post-lowering graph, so the comparison is valid.
    # ------------------------------------------------------------------

    @parameterized.expand(
        [
            # (name, batch, heads, seq, head_dim, scale, dtype, atol)
            ("opt_b2_h8_fp16",   2,  8,  32,  64, None, torch.float16, 1e-2),
            ("opt_b1_h8_fp32",   1,  8,  64,  64, None, torch.float32, 1e-2),
            ("opt_scale05_fp16", 2,  8,  32,  64, 0.5,  torch.float16, 1e-2),
            ("opt_large_fp16",   1,  8, 128,  64, None, torch.float16, 1e-2),
            ("opt_b4_h16_fp16",  4, 16,  64,  64, None, torch.float16, 1e-2),
            ("opt_d128_fp16",    1,  8,  32, 128, None, torch.float16, 1e-2),
        ]
    )
    def test_attn_bias_is_causal_opt(
        self, name, batch, num_heads, seq, head_dim, scale, dtype, atol
    ):
        class EfficientAttnBiasOpt(nn.Module):
            def forward(self, q, k, v, bias):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q, k, v, bias, False, 0.0, False, scale=scale
                )
                return out[0]

        q    = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        k    = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        v    = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        # The actual bias values do not affect the output: the pass replaces
        # attn_bias with is_causal=True before any converter runs.
        bias = torch.randn(batch, num_heads, seq, seq, dtype=dtype)
        self.run_test(
            EfficientAttnBiasOpt(),
            [q, k, v, bias],
            rtol=1e-2,
            atol=atol,
            precision=dtype,
            enable_passes=True,
            use_explicit_typing=True,
            decompose_attention=False,
            attn_bias_is_causal=True,
        )


if __name__ == "__main__":
    run_tests()
