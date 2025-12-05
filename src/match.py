import cupy as cp

def convolution(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
    assert A.shape == B.shape and len(A.shape) == 1
    n = len(A) + len(B) - 1
    extA = cp.zeros(n)
    extA[:len(A)] = A
    extB = cp.zeros(n)
    extB[:len(A)] = B
    tA = cp.fft.fft(extA)
    tB = cp.fft.fft(extB)
    return cp.real(cp.fft.ifft(tA * tB))

def slide_dot(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
    assert A.shape == B.shape and len(A.shape) == 1
    return convolution(A, B[::-1])[len(B) - 1:]

def _pad_all_dims_to_shape(arr, target_shape):
    padded_arr = cp.zeros(target_shape, dtype=arr.dtype)
    indices = [slice(0, arr.shape[dim]) for dim in range(arr.ndim)]
    padded_arr[tuple(indices)] = arr
    return padded_arr

def match(A: cp.ndarray, B: cp.ndarray, P: cp.ndarray) -> cp.ndarray:
    assert B.shape == P.shape and len(A.shape) == len(B.shape)
    for i in range(len(A.shape)):
        assert A.shape[i] >= B.shape[i]
    A2 = A ** 2
    B = _pad_all_dims_to_shape(B, A.shape)
    P = _pad_all_dims_to_shape(P, A.shape)
    BP = B * P
    B2PSUM = cp.sum((B ** 2) * P)
    ANS = (slide_dot(A2.flatten(), P.flatten()) - 2 * slide_dot(A.flatten(), BP.flatten()) + B2PSUM).reshape(A.shape)
    return ANS

if __name__ == "__main__":
    A = cp.array([
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
    ])
    B = cp.array([
        [1, 0],
        [0, 1],
    ])
    C = cp.array([
        [1, 0],
        [1, 1],
    ])
    print(match(A, B, C))
