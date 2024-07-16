import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flash_attn_jax import flash_mha
import flash_attn_jax_lib.flash_api as flash_api
from .test_flash import check, ref_mha, pretty

def ref_sympow(q, k, v, p=2, eps=1e-6):
    [n, l, h, d] = q.shape
    D = jnp.einsum('nlhd,nLhd->nhlL', q, k)
    print(D)
    # D /= d**0.5
    log_C = p * jnp.log(jnp.abs(D) + eps)
    log_C = jnp.where(jnp.tril(jnp.ones(log_C.shape)), log_C, -jnp.inf)
    log_C -= log_C.max(axis=-1, keepdims=True)
    
    B = jnp.exp(log_C)
    A = B / (B.sum(axis=-1, keepdims=True) + eps)
    Y = jnp.einsum('nhlL,nLhd->nlhd', A, v)
    return Y


# @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
# @pytest.mark.parametrize("local", ['local',''])
# @pytest.mark.parametrize("d", [59, 32])
# @pytest.mark.parametrize("h", [1, 4])
# @pytest.mark.parametrize("seqlen", [97, 128])
# @pytest.mark.parametrize("n", [1, 4])
# @pytest.mark.parametrize("p", [1, 2, 4, 8])



@pytest.mark.parametrize("dtype", [jnp.float16])
@pytest.mark.parametrize("local", ['local'])
@pytest.mark.parametrize("d", [4])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("seqlen", [8])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("p", [2])
def test_sympow_fwd(p, n, seqlen, h, d, local, dtype):
    # window_size = (10, 0) if local else (-1, -1)
    window_size = (-1, -1)
    
    # q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen, h, d], dtype=jnp.float32)
    # k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=jnp.float32)
    # v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=jnp.float32)
    
    q = jnp.tile(jnp.stack([jnp.ones(shape=[d])*i for i in range(1, seqlen+1)]), (n, h, 1, 1)).transpose(0, 2, 1, 3)

    k = q
    v = q
    
    ref_out_softmax = ref_mha(q, k, v, is_causal=True, window_size=window_size)
    ref_out = ref_sympow(q, k, v, p)
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    
    jax_out = ref_sympow(q, k, v, p)
    jax_softmax_out = ref_mha(q, k, v, is_causal=True, window_size=window_size)
    # flash_out, lse, p = flash_mha(q, k, v, is_causal=True, window_size=window_size, similarity=flash_api.sympower, deg=p)
    flash_out_softmax, lse_softmax, p_softmax = flash_mha(q, k, v, is_causal=True, window_size=window_size)
    pretty(p_softmax)
    import pdb; pdb.set_trace()
    # check(ref_out, jax_out, flash_out)
    check(ref_out_softmax, jax_softmax_out, flash_out_softmax)
    