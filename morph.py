import cv2
import numpy as np


def warp_triangle(img, src_tri, dst_tri, out_img):
    # Bounding boxes
    src_rect = cv2.boundingRect(np.float32([src_tri]))
    dst_rect = cv2.boundingRect(np.float32([dst_tri]))

    src_offset = []
    dst_offset = []
    for i in range(3):
        src_offset.append(
            ((src_tri[i][0] - src_rect[0]), (src_tri[i][1] - src_rect[1]))
        )
        dst_offset.append(
            ((dst_tri[i][0] - dst_rect[0]), (dst_tri[i][1] - dst_rect[1]))
        )

    # Warp triangle
    M = cv2.getAffineTransform(np.float32(src_offset), np.float32(dst_offset))
    warped = cv2.warpAffine(
        img[
            src_rect[1] : src_rect[1] + src_rect[3],
            src_rect[0] : src_rect[0] + src_rect[2],
        ],
        M,
        (dst_rect[2], dst_rect[3]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Mask and blend into output image
    mask = np.zeros((dst_rect[3], dst_rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_offset), (1.0, 1.0, 1.0), 16, 0)
    out_img[
        dst_rect[1] : dst_rect[1] + dst_rect[3], dst_rect[0] : dst_rect[0] + dst_rect[2]
    ] = (
        out_img[
            dst_rect[1] : dst_rect[1] + dst_rect[3],
            dst_rect[0] : dst_rect[0] + dst_rect[2],
        ]
        * (1 - mask)
        + warped * mask
    )


def morph_images(img1, img2, pts1, pts2, triangles, alpha):
    pts = (1 - alpha) * np.array(pts1) + alpha * np.array(pts2)

    # Prepare two warped results
    warp1 = np.zeros_like(img1, dtype=np.float32)
    warp2 = np.zeros_like(img2, dtype=np.float32)

    for tri in triangles:
        x1, y1, z1 = pts1[tri], pts2[tri], pts[tri]
        warp_triangle(img1, x1, z1, warp1)
        warp_triangle(img2, y1, z1, warp2)

    # Cross-dissolve between the two warped images
    morphed = (1 - alpha) * warp1 + alpha * warp2
    return np.clip(morphed, 0, 255).astype(np.uint8)
