import jax
import jax.numpy as jnp
import numpy as np
import pytest
from functools import partial
import time

from flash_attn_jax import flash_mha
import flash_attn_jax_lib.flash_api as flash_api


def pretty(tensor):
    shape = tensor.shape
    mx = jnp.max(tensor)
    mn = jnp.min(tensor)
    mean = jnp.mean(tensor)
    std = jnp.std(tensor)
    return f'[{shape}: {mn:.3g} | {mean:.3g}±{std:.3g} | {mx:.3g}]'


def check(ref_out, jax_out, out):
    def check1(ref_out, jax_out, out):
        max_err = jnp.max(jnp.abs(out - ref_out)).item()
        jax_max_err = jnp.max(jnp.abs(jax_out - ref_out)).item()
        assert (
            jnp.max(jnp.abs(out - ref_out)).item()
            <= 2 * jnp.max(jnp.abs(jax_out - ref_out)).item()
        ), (
            'max_err',
            max_err,
            'jax_max_err',
            jax_max_err,
            pretty(jnp.abs(out - ref_out)),
            'vs',
            pretty(jnp.abs(jax_out - ref_out)),
        )

    jax.tree_map(check1, ref_out, jax_out, out)


def ref_sympow(q, k, v, p=2, eps=1e-6):
    [n, l, h, d] = q.shape
    D = jnp.einsum('nlhd,nLhd->nhlL', q, k)
    D /= d**0.5
    log_C = p * jnp.log(jnp.abs(D) + eps)
    log_C = jnp.where(jnp.tril(jnp.ones(log_C.shape)), log_C, -jnp.inf)
    log_C -= log_C.max(axis=-1, keepdims=True)

    B = jnp.exp(log_C)
    A = B / (B.sum(axis=-1, keepdims=True) + eps)
    Y = jnp.einsum('nhlL,nLhd->nlhd', A, v)
    return Y


def loss(fn):
    def loss_fn(q, k, v):
        return jnp.sum(fn(q, k, v))

    return loss_fn


@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize('h', [2])
@pytest.mark.parametrize('seqlen', [64, 128, 256, 512, 1024, 8196, 16384])
@pytest.mark.parametrize('n', [1])
@pytest.mark.parametrize('p', [4])
def test_sympow_fwd(p, n, seqlen, h, d, dtype):
    window_size = (-1, -1)

    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen, h, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=jnp.float32)

    ref_out = ref_sympow(q, k, v, p)
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)

    jax_out = ref_sympow(q, k, v, p)
    flash_out = flash_mha(
        q,
        k,
        v,
        is_causal=True,
        window_size=window_size,
        similarity=flash_api.sympower,
        deg=p,
    )
    check(ref_out, jax_out, flash_out)


@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize('h', [2])
@pytest.mark.parametrize('seqlen', [1024])
@pytest.mark.parametrize('n', [1])
@pytest.mark.parametrize('p', [4])
def test_sympow_bwd(p, n, seqlen, h, d, dtype):
    window_size = (-1, -1)

    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen, h, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=jnp.float32)

    ref_out = ref_sympow(q, k, v, p)
    ref_loss = loss(partial(ref_sympow, p=p))(q, k, v)
    ref_grad = jax.grad(loss(partial(ref_sympow, p=p)), argnums=(0, 1, 2))(q, k, v)

    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)

    jax_out = ref_sympow(q, k, v, p)
    jax_loss = loss(partial(ref_sympow, p=p))(q, k, v)
    jax_grad = jax.grad(loss(partial(ref_sympow, p=p)), argnums=(0,1,2))(q, k, v)

    flash_out = flash_mha(
        q,
        k,
        v,
        is_causal=True,
        window_size=window_size,
        similarity=flash_api.sympower,
        deg=p,
    )
    flash_loss = loss(
        partial(
            flash_mha,
            softmax_scale=1.0,
            is_causal=True,
            window_size=window_size,
            similarity=flash_api.sympower,
            deg=p,
        )
    )(q, k, v)
    flash_grad = jax.grad(
        loss(
            partial(
                flash_mha,
                softmax_scale=1.0,
                is_causal=True,
                window_size=window_size,
                similarity=flash_api.sympower,
                deg=p,
            )
        ),
        argnums=(0, 1, 2),
    )(q, k, v)
    check(ref_grad, jax_grad, flash_grad)


def timer(fn, warmup=10, number=50):
    for _ in range(warmup):
        fn().block_until_ready()

    start = time.time()
    for _ in range(number):
        res = fn().block_until_ready()

    return (time.time() - start) / number, res


def benchmark_fwd():
    print('Benchmarking fwd')
    for dtype in [jnp.float16, jnp.bfloat16]:
        for d in [32, 64, 128]:
            for h in [6]:
                for seqlen in [128, 1024, 4096, 8192]:
                    for n in [1, 4]:
                        for p in [2, 4, 8]:
                            window_size = (-1, -1)

                            q = jax.random.normal(
                                jax.random.PRNGKey(0),
                                [n, seqlen, h, d],
                                dtype=jnp.float32,
                            )
                            k = jax.random.normal(
                                jax.random.PRNGKey(1),
                                [n, seqlen, h, d],
                                dtype=jnp.float32,
                            )
                            v = jax.random.normal(
                                jax.random.PRNGKey(2),
                                [n, seqlen, h, d],
                                dtype=jnp.float32,
                            )

                            q = q.astype(dtype)
                            k = k.astype(dtype)
                            v = v.astype(dtype)

                            def flash_fn():
                                return flash_mha(
                                    q,
                                    k,
                                    v,
                                    is_causal=True,
                                    window_size=window_size,
                                    similarity=flash_api.sympower,
                                    deg=p,
                                )

                            def jax_fn():
                                return ref_sympow(q, k, v, p)

                            def ref_fn(q, k, v):
                                q, k, v = (
                                    q.astype(jnp.float32),
                                    k.astype(jnp.float32),
                                    v.astype(jnp.float32),
                                )
                                return ref_sympow(q, k, v, p)

                            ref_out = ref_fn(q, k, v)

                            jax_time, jax_out = timer(jax.jit(jax_fn))
                            flash_time, flash_out = timer(jax.jit(flash_fn))
                            speedup = (jax_time / flash_time - 1) * 100
                            dtype_str = str(dtype).split('.')[-1].strip("><'")
                            print(
                                f'dtype={dtype_str}, d={d}, h={h}, seqlen={seqlen}, n={n}, p={p}, jax_time={jax_time:.4f}, flash_time={flash_time:.4f}, speedup={speedup:.2f}%'
                            )


def benchmark_bwd():
    print('Benchmarking bwd')
    for dtype in [jnp.float16, jnp.bfloat16]:
        for d in [32, 64, 128]:
            for h in [6]:
                for seqlen in [128, 1024, 4096, 8192]:
                    for n in [1, 4]:
                        for p in [2, 4, 8]:
                            window_size = (-1, -1)

                            q = jax.random.normal(
                                jax.random.PRNGKey(0),
                                [n, seqlen, h, d],
                                dtype=jnp.float32,
                            )
                            k = jax.random.normal(
                                jax.random.PRNGKey(1),
                                [n, seqlen, h, d],
                                dtype=jnp.float32,
                            )
                            v = jax.random.normal(
                                jax.random.PRNGKey(2),
                                [n, seqlen, h, d],
                                dtype=jnp.float32,
                            )

                            q = q.astype(dtype)
                            k = k.astype(dtype)
                            v = v.astype(dtype)

                            def flash_fn(q, k, v):
                                out = flash_mha(
                                    q,
                                    k,
                                    v,
                                    is_causal=True,
                                    window_size=window_size,
                                    similarity=flash_api.sympower,
                                    deg=p,
                                )
                                return jnp.sum(out**0.2)

                            def jax_fn(q, k, v):
                                out = ref_sympow(q, k, v, p)
                                return jnp.sum(out**0.2)

                            flash_grad_fn = partial(jax.grad(flash_fn), q, k, v)
                            jax_grad_fn = partial(jax.grad(jax_fn), q, k, v)

                            jax_time, jax_out = timer(jax.jit(jax_grad_fn))
                            flash_time, flash_out = timer(jax.jit(flash_grad_fn))
                            speedup = (jax_time / flash_time - 1) * 100
                            dtype_str = str(dtype).split('.')[-1].strip("><'")
                            print(
                                f'dtype={dtype_str}, d={d}, h={h}, seqlen={seqlen}, n={n}, p={p}, jax_time={jax_time:.4f}, flash_time={flash_time:.4f}, speedup={speedup:.2f}%'
                            )


if __name__ == '__main__':
    # benchmark_fwd()
    benchmark_bwd()
    # pytest.main([__file__])