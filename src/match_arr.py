import cupy as cp

def convolution(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
    assert A.shape == B.shape and len(A.shape) == 1
    n = len(A) + len(B) - 1
    extA = cp.zeros(n, dtype=A.dtype)
    extA[:len(A)] = A
    extB = cp.zeros(n, dtype=B.dtype)
    extB[:len(A)] = B
    tA = cp.fft.fft(extA)
    tB = cp.fft.fft(extB)
    return cp.real(cp.fft.ifft(tA * tB))

def slide_dot(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
    assert A.shape == B.shape and len(A.shape) == 1
    return convolution(A, B[::-1])[len(B) - 1:]

def _pad_all_dims_to_shape(arr: cp.ndarray, target_shape:tuple[int]):
    padded_arr = cp.zeros(target_shape, dtype=arr.dtype)
    indices = [slice(0, arr.shape[dim]) for dim in range(arr.ndim)]
    padded_arr[tuple(indices)] = arr
    return padded_arr

def crop_array_to_shape(arr: cp.ndarray, target_shape:tuple[int]):
    slices = tuple([slice(0, target_shape[dim]) for dim in range(arr.ndim)])
    cropped_arr = arr[slices]
    return cropped_arr

def match_arr(A: cp.ndarray, B: cp.ndarray, P: cp.ndarray) -> cp.ndarray:
    assert B.shape == P.shape and len(A.shape) == len(B.shape)
    for i in range(len(A.shape)):
        assert A.shape[i] >= B.shape[i]
    A2 = A ** 2
    extB = _pad_all_dims_to_shape(B, A.shape)
    extP = _pad_all_dims_to_shape(P, A.shape)
    BP = extB * extP
    B2PSUM = cp.sum((extB ** 2) * extP)
    ANS = (slide_dot(A2.flatten(), extP.flatten()) - 2 * slide_dot(A.flatten(), BP.flatten()) + B2PSUM).reshape(A.shape)
    return crop_array_to_shape(ANS, tuple([A.shape[i] - B.shape[i] + 1 for i in range(len(A.shape))]))

if __name__ == "__main__":
    A = cp.array([[
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
    ]] * 2)
    B = cp.array([[
        [1, 0],
        [0, 1],
    ]] * 2)
    C = cp.array([[
        [1, 0],
        [1, 1],
    ]] * 2)
    print(match_arr(A, B, C))
