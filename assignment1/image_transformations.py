import numpy as np
import cv2

def transform(image: np.ndarray, T: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    T_inv = np.linalg.inv(T)
    output = np.zeros_like(image)

    for y_out in range(h):
        for x_out in range(w):
            p_out = np.array([x_out, y_out, 1.0])
            p_src = T_inv @ p_out
            x_src, y_src = p_src[0] / p_src[2], p_src[1] / p_src[2]

            xn = int(round(x_src))
            yn = int(round(y_src))
            if 0 <= xn < w and 0 <= yn < h:
                output[y_out, x_out] = image[yn, xn]

    return output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Apply translation and rotation to an image"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input image")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Could not read {args.input}")

    h, w = img.shape[:2]

    # TRANSLATION
    #shift right by 150px, down by 50px
    tx, ty = 150, 50
    T_translate = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)

    translated = transform(img, T_translate)
    cv2.imwrite("translated.png", translated)
    print("→ Saved translated.png")

    # ROTATION about image center
    # rotate by 30 degrees counter-clockwise
    angle_deg = 30
    θ = np.deg2rad(angle_deg)
    cx, cy = w / 2.0, h / 2.0

    # 1) shift center to origin
    T1 = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0,   1]
    ], dtype=np.float64)
    # 2) rotate by θ
    R = np.array([
        [ np.cos(θ), -np.sin(θ), 0],
        [ np.sin(θ),  np.cos(θ), 0],
        [         0,          0, 1]
    ], dtype=np.float64)
    # 3) shift back
    T2 = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0,  1]
    ], dtype=np.float64)

    T_rotate = T2 @ R @ T1
    rotated = transform(img, T_rotate)
    cv2.imwrite("rotated.png", rotated)
    print("→ Saved rotated.png")