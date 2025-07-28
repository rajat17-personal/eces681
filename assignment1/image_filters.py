import cv2
import numpy as np

def apply_averaging_kernels(img: np.ndarray, sizes=(3,5)) -> dict:
    outputs = {}
    for k in sizes:
        kernel = np.ones((k, k), dtype=np.float32) / (k*k)
        outputs[f"average_{k}x{k}"] = cv2.filter2D(img, -1, kernel)
    return outputs

def apply_gaussian_kernels(img: np.ndarray, sigmas=(1,2,3)) -> dict:
    outputs = {}
    for sigma in sigmas:
        outputs[f"gaussian_sigma{sigma}"] = cv2.GaussianBlur(
            img,
            (0, 0),   
            sigma      
        )
    return outputs

def apply_roberts(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kx = np.array([[1, 0],
                   [0,-1]], dtype=np.float32)
    ky = np.array([[0, 1],
                   [-1,0]], dtype=np.float32)
    rx = cv2.filter2D(gray, cv2.CV_32F, kx)
    ry = cv2.filter2D(gray, cv2.CV_32F, ky)
    edges = cv2.magnitude(rx, ry)
    return cv2.convertScaleAbs(edges)

def apply_sobel(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(sx, sy)
    return cv2.convertScaleAbs(edges)

def apply_prewitt(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[ 1,  1,  1],
                   [ 0,  0,  0],
                   [-1, -1, -1]], dtype=np.float32)
    px = cv2.filter2D(gray, cv2.CV_32F, kx)
    py = cv2.filter2D(gray, cv2.CV_32F, ky)
    edges = cv2.magnitude(px, py)
    return cv2.convertScaleAbs(edges)

def GaussianPyramids(img: np.ndarray, levels: int) -> list:
    gp = [img.copy()]
    for i in range(1, levels):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp

def LaplacianPyramids(img: np.ndarray, levels: int) -> list:
    gp = GaussianPyramids(img, levels)
    lp = []
    for i in range(levels - 1):
        # expand the next Gaussian level back to current size
        expanded = cv2.pyrUp(gp[i+1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lap = cv2.subtract(gp[i], expanded)
        lp.append(lap)
    lp.append(gp[-1])  # smallest Gaussian at the top
    return lp

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(
        description="Problem 2: filtering & pyramids"
    )
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("-l", "--levels", type=int, default=4,
                        help="Number of pyramid levels (default: 4)")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Cannot read {args.input}")

    base, _ = os.path.splitext(os.path.basename(args.input))

    # 1) Spatial filters
    avgs   = apply_averaging_kernels(img)
    gauss  = apply_gaussian_kernels(img)
    roberts= apply_roberts(img)
    sobel  = apply_sobel(img)
    prewitt= apply_prewitt(img)

    # save results
    for name, out in {**avgs, **gauss}.items():
        cv2.imwrite(f"{base}_{name}.png", out)
    cv2.imwrite(f"{base}_roberts.png",  roberts)
    cv2.imwrite(f"{base}_sobel.png",    sobel)
    cv2.imwrite(f"{base}_prewitt.png",  prewitt)

    print("Saved: averaging, gaussian, Roberts, Sobel, Prewitt outputs")

    # 2) Gaussian pyramid
    gp = GaussianPyramids(img, args.levels)
    for i, level in enumerate(gp):
        cv2.imwrite(f"{base}_gp_level{i}.png", level)
    print(f"Saved {args.levels} levels of Gaussian pyramid")

    # 3) Laplacian pyramid
    lp = LaplacianPyramids(img, args.levels)
    for i, level in enumerate(lp):
        cv2.imwrite(f"{base}_lp_level{i}.png", level)
    print(f"Saved {args.levels} levels of Laplacian pyramid")
