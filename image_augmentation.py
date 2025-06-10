"""
Task: Augmenting images from a source camera to match a target camera

Motivation:
We have a large annotated dataset to train a neural network. This dataset consists of images which were captured by a source camera.
Now we get a new camera and notice that our network does not perform well with the new images of this camera.
In order to improve results we want to augment our dataset. This means that we make the images from the source camera visually match 
the images of our new camera (let's call it the target camera) so we can retrain our model as if we had a dataset captured with our target camera.

Task:
Augment the given image captured by the source camera to match geometric and color characteristics of the target camera.
First, adjust intrinsics and distortions.
Additionally, we want to keep the resolution of the source image with an additional 128 pixels cropped from top and bottom (the order is: first scale, then crop).
Finally, adapt the color space of the image to match the color space of the target camera.
Fill in the blanks indicated by the comments and do not remove or modify code that is already there.

Asumptions:
We assume opencv fisheye distortions for both cameras.
The raw images have gamma applied after loading. img_bgr_linear = img_bgr ** gamma converts to linear colors.
The color correction matrix (ccm) maps the camera color space of the raw image to a common reference color space via bgr_common = ccm @ bgr_cam.
The color order is BGR throughout the entire code.
"""

import cv2
import numpy as np


def compute_remap(K_s, D_s, shape_s, K_t, D_t, shape_t, crop):
    h_s, w_s = shape_s[:2]
    h_t, w_t = shape_t[:2]

    scale_y = h_s / h_t
    # scale_y = (h_s - 2 * crop) / h_t  # Account for cropping
    scale_x = w_s / w_t
    scale = min(scale_x, scale_y)

    K_t_scaled = K_t.copy()
    K_t_scaled[0, 0] *= scale  # fx
    K_t_scaled[1, 1] *= scale  # fy
    K_t_scaled[0, 2] *= scale  # cx
    K_t_scaled[1, 2] *= scale  # cy

    new_size = (w_s, h_s)
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
        K_s, D_s, np.eye(3), K_t_scaled, new_size, cv2.CV_32FC1
    )

    mapx = mapx[crop : h_s - crop, :]
    mapy = mapy[crop : h_s - crop, :]

    return mapx, mapy


def correct_colors(img, ccm_s, gamma_s, ccm_t, gamma_t):
    # Source camera --> Linear space --> Gamma correction --> Reference space --> Target camera space --> Target Gamma space
    # Convert to linear color space
    img_linear = img.astype(np.float32) / 255.0
    img_linear = img_linear**gamma_s  # gamma correction
    h, w, c = img_linear.shape
    img_reshaped = img_linear.reshape(-1, 3).T
    # Transform to target camera color space
    common_space = ccm_s @ img_reshaped  # source camera to common
    ccm_t_inv = np.linalg.pinv(ccm_t)
    target_space = ccm_t_inv @ common_space  # common to target camera
    # ensure valid range [0,1]
    target_space = np.clip(target_space, 0, 1)
    target_space = target_space.T.reshape(h, w, c)
    res = target_space ** (1.0 / gamma_t)  # target gamma
    # Convert back to 8-bit
    res = (res * 255).astype(np.uint8)
    return res


def main():
    # source camera image which we want to map to...
    img_s = cv2.imread("source.png")  # image from source camera
    K_s = np.array(
        [
            [959.2911641873376, 0, 940.0755734722462],
            [0, 959.282939308865, 542.415127929082],
            [0, 0, 1],
        ]
    )  # camera matrix
    D_s = np.array(
        [
            -0.03573755121302301,
            -0.0036219160175428302,
            0.0006017312708962034,
            -0.0007929108890333689,
        ]
    )  # distortion coefficients
    ccm_s = np.array(
        [
            [0.78763107, 0.17527123, 0.15850111],
            [0.24557854, 0.78878407, 0.40866234],
            [0.23512331, 0.1418083, 1.22562264],
        ]
    )  # color correction matrix
    gamma_s = 2.2  # gamma

    # ... the target camera
    img_t = cv2.imread("target.png")
    K_t = np.array(
        [
            [3201.3907890945807, 0, 1884.892000429184],
            [0, 3201.529240818799, 1003.7764621284468],
            [0, 0, 1],
        ]
    )
    D_t = np.array(
        [
            -0.057152490839343144,
            -0.009121729078728442,
            0.006781644882334966,
            0.0980547019970438,
        ]
    )
    ccm_t = np.array(
        [
            [0.77917258, -0.20624031, 0.12478972],
            [-0.08190668, 0.47017028, -0.03888143],
            [0.01975545, -0.30940343, 1.67714349],
        ]
    )
    gamma_t = 3.0

    crop = 128

    mapx, mapy = compute_remap(K_s, D_s, img_s.shape, K_t, D_t, img_t.shape, crop)
    res = cv2.remap(
        img_s,
        mapx.astype(np.float32),
        mapy.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    res = correct_colors(np.copy(res), ccm_s, gamma_s, ccm_t, gamma_t)

    # cv2.imwrite('solution.png', res)

    # check solution
    solution = cv2.imread("solution.png")
    assert res.shape == (764, 1920, 3), f"solution has wrong dimensions ({res.shape})"
    assert res.dtype == np.uint8, "solution has wrong data type"
    error = (
        np.sum(np.abs(solution.astype(np.float32) - res.astype(np.float32)))
        / solution.size
    )
    assert (
        error < 0.01 * 256
    ), f"solution is incorrect (per pixel average error is: {error:.4f})"

    # pad result again for display
    res = cv2.copyMakeBorder(res, crop, crop, 0, 0, cv2.BORDER_CONSTANT, 0)
    # display images
    display_size = (1920 // 4, 1024 // 4)
    img_s = cv2.resize(img_s, display_size)
    img_t = cv2.resize(img_t, display_size)
    res = cv2.resize(res, display_size)
    images = cv2.hconcat([img_s, img_t, res])
    cv2.imshow("result", images)
    cv2.waitKey(-1)


if __name__ == "__main__":
    main()
