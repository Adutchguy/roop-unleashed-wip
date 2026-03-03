import cv2
import numpy as np

# VR Lense Distortion
# Taken from https://github.com/g0kuvonlange/vrswap


def get_perspective(img, FOV, THETA, PHI, height, width):
    #
    # THETA is left/right angle, PHI is up/down angle, both in degree
    #
    [orig_width, orig_height, _] = img.shape
    equ_h = orig_height
    equ_w = orig_width
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    wFOV = FOV
    hFOV = float(height) / width * wFOV

    w_len = np.tan(np.radians(wFOV / 2.0))
    h_len = np.tan(np.radians(hFOV / 2.0))

    x_map = np.ones([height, width], np.float32)
    y_map = np.tile(np.linspace(-w_len, w_len, width), [height, 1])
    z_map = -np.tile(np.linspace(-h_len, h_len, height), [width, 1]).T

    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.stack((x_map, y_map, z_map), axis=2) / np.repeat(
        D[:, :, np.newaxis], 3, axis=2
    )

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    xyz = xyz.reshape([height * width, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T
    lat = np.arcsin(xyz[:, 2])
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])

    lon = lon.reshape([height, width]) / np.pi * 180
    lat = -lat.reshape([height, width]) / np.pi * 180

    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90 * equ_cy + equ_cy

    persp = cv2.remap(
        img,
        lon.astype(np.float32),
        lat.astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    )
    return persp
