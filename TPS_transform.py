import os
import cv2
import glob
import time
import numpy as np


def get_tps_transform_matrix(dst_points, src_points):
    K = dst_points.shape[0]

    B = src_points.T
    expand_zeros = np.zeros((2, 3), dtype=np.float)
    B = np.concatenate([B, expand_zeros], axis=1)

    ones = np.ones((1, K), dtype=np.float)
    C = dst_points.T
    C = np.concatenate([ones, C], axis=0)
    delta_C = np.zeros((K + 3, K + 3), dtype=np.float)
    delta_C[:3, :K] = C
    delta_C[3:, K:] = C.T

    A = np.tile(np.expand_dims(dst_points, axis=0), [K, 1, 1])
    D = np.expand_dims(dst_points, axis=1)
    distance = np.sqrt(np.sum((A - D)**2, axis=2))
    C_hat = (distance**2) * np.log(distance + 1e-6)

    delta_C[3:, :K] = C_hat
    delta_C_inv = np.linalg.inv(delta_C)
    transform_matrix = np.matmul(B, delta_C_inv)

    return transform_matrix


def get_grid(height, width):
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.flatten(), yy.flatten()
    grid = np.vstack([xx, yy])

    return grid


def interplate(image, grid):
    height, width = image.shape[:2]
    floor_grid = np.floor(grid).astype(np.int)
    x0 = floor_grid[0, :]
    y0 = floor_grid[1, :]
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    t = np.expand_dims(x1 - grid[0], axis=1)
    s = np.expand_dims(y1 - grid[1], axis=1)

    R1 = t * image[y0, x0] + (1 - t) * image[y0, x1]
    R2 = t * image[y1, x0] + (1 - t) * image[y1, x1]
    P = s * R1 + (1 - s) * R2

    return P


def transform_image(image, matrix, dst_points, output_size):
    height, width = output_size
    dst_grid = get_grid(height, width)
    ones = np.ones((1, dst_grid.shape[1]), dtype=np.float)
    A = np.concatenate([ones, dst_grid], axis=0)
    B = np.expand_dims(dst_grid.T, axis=0)
    C = np.expand_dims(dst_points, axis=1)
    D = np.sqrt(np.sum((B - C)**2, axis=2).T)
    D = (D**2) * np.log(D + 1e-6)

    A = np.concatenate([A.T, D], axis=1)
    src_grid = np.matmul(matrix, A.T)
    interplate_value = interplate(image, src_grid)

    interplate_new_image = np.zeros((height, width, 3), dtype=np.uint8)
    interplate_new_image[dst_grid[1], dst_grid[0]] = interplate_value

    return interplate_new_image


def main():
    image = cv2.imread('test.jpg')
    height, width = image.shape[:2]
    dst_points = []
    src_points = []
    K = 20
    shift_height = [-1, -2, -3, -4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    interval = width // K
    for i in range(K):
        src_points.append([i * interval, 0])
        src_points.append([i * interval, height])
        dst_points.append([i * interval, shift_height[i]])
        dst_points.append([i * interval, height + shift_height[i]])

    dst_points = np.array(dst_points, dtype=np.float)
    src_points = np.array(src_points, dtype=np.float)
    start = time.time()
    matrix = get_tps_transform_matrix(dst_points, src_points)
    interplate_dst_image = transform_image(
        image, matrix, dst_points, (height, int(width)))
    print('Spend {}s use tps transform image.'.format(time.time() - start))
    cv2.imwrite('res.jpg', interplate_dst_image)
    cv2.imshow('src_image', image)
    cv2.imshow('interplate_dst_image', interplate_dst_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()