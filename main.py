import cv2
import numpy as np
from load_image import load_image
from morph import morph_images
from geometry import delaunay_triangulation
from points_picker import pick_corresponding_points_side_by_side

# algorithm, không dùng thư viện

# load 2 ảnh
# pick điểm tương ứng trên 2 ảnh
# -> pick_corresponding_points_side_by_side()

# gen delaunay triangle
# -> delaunay_triangulation()
# TODO: delaunay_triangulation viết tay ko dùng thư viện

# tìm ma trận biến đổi
# -> affine_transform()

# warp từng triangle từ ảnh 1 sang ảnh 2
# -> morph_images() dùng nhiều warp_triangle()

# cross dissolve ảnh
# -> morph_images(), dòng pts = (1 - alpha) * np.array(pts1) + alpha * np.array(pts2)


# Load images
img1 = load_image("images/barack_obama.jpg")
img2 = load_image("images/hillary_clinton.jpg")

# chọn điểm, ấn enter để confirm
pts1, pts2 = pick_corresponding_points_side_by_side(img1, img2)

# Compute Delaunay triangles
triangles = delaunay_triangulation(np.array(pts1))

# Morph sequence
while True:
    for alpha in np.linspace(0, 1, 11):
        morphed = morph_images(img1, img2, pts1, pts2, triangles, alpha)
        cv2.imshow("Morph", morphed)
        key = cv2.waitKey(30)
        if key == 27:  # ESC to exit
            break
    if key == 27:  # ESC to exit
        break
