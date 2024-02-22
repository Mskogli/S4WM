import jax
import jax.numpy as jnp

from jax.numpy.linalg import eigh, inv, matrix_power
from typing import Callable


def log_step_initializer(dt_min: float = 0.001, dt_max: float = 0.1) -> Callable:
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)

    return init


def scan_SSM(
    Ab: jnp.ndarray, Bb: jnp.ndarray, Cb: jnp.ndarray, u: jnp.ndarray, x0: jnp.ndarray
) -> jnp.ndarray:

    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def causal_convolution(u: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
    ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
    Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
    out = ud * Kd
    return jnp.fft.irfft(out)[: u.shape[0]]


def make_HiPPO(N: jnp.ndarray) -> jnp.ndarray:
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A


@jax.jit
def cauchy(v: jnp.ndarray, omega: jnp.ndarray, lambd: jnp.ndarray) -> jnp.ndarray:
    """Cauchy matrix multiplication: (n), (l), (n) -> (l)"""
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)


def kernel_DPLR(
    Lambda: jnp.ndarray,
    P: jnp.ndarray,
    Q: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    step: float,
    L: int,
) -> jnp.ndarray:
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = jnp.exp((-2j * jnp.pi) * (jnp.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = jnp.fft.ifft(atRoots, L).reshape(L)
    return out.real


def discrete_DPLR(
    Lambda: jnp.ndarray,
    P: jnp.ndarray,
    Q: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    step: float,
    L: int,
) -> jnp.ndarray:
    # Convert parameters to matrices
    B = B[:, jnp.newaxis]
    Ct = C[jnp.newaxis, :]

    N = Lambda.shape[0]
    A = jnp.diag(Lambda) - P[:, jnp.newaxis] @ Q[:, jnp.newaxis].conj().T
    I = jnp.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = jnp.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()


def make_NPLR_HiPPO(N: int) -> jnp.ndarray:
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = jnp.sqrt(jnp.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return nhippo, P, B


def make_DPLR_HiPPO(N: int) -> jnp.ndarray:
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]

    # Check skew symmetry
    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V


def init(x):
    def _init(key, shape):
        assert shape == x.shape
        return x

    return _init


def hippo_initializer(N: int) -> jnp.ndarray:
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    return init(Lambda.real), init(Lambda.imag), init(P), init(B)
