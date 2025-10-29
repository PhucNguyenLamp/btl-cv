import cv2
import numpy as np


def pick_corresponding_points_side_by_side(img1, img2, radius=5):
    """
    Side-by-side interactive picker:
      Left  = image 1 (click to add points)
      Right = image 2 (drag to adjust)
    Returns (pts1, pts2).
    """

    # Ensure same height for tiling
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_height = max(h1, h2)
    scale1 = target_height / h1
    scale2 = target_height / h2
    img1 = cv2.resize(img1, (int(w1 * scale1), target_height))
    img2 = cv2.resize(img2, (int(w2 * scale2), target_height))

    # Combine side-by-side
    sep = 20
    combined = np.zeros(
        (target_height, img1.shape[1] + img2.shape[1] + sep, 3), dtype=np.uint8
    )
    combined[:, : img1.shape[1]] = img1
    combined[:, img1.shape[1] + sep :] = img2

    pts1, pts2 = [], []
    selected_idx = -1
    dragging = False
    offset_x = img1.shape[1] + sep  # start x of second image

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_idx, dragging

        # Left image click → add new point
        if event == cv2.EVENT_LBUTTONDOWN and x < img1.shape[1]:
            pts1.append((x, y))
            pts2.append((x + offset_x, y))  # initialize same y in right image

        # Right image drag
        elif event == cv2.EVENT_LBUTTONDOWN and x > offset_x:
            for i, (px, py) in enumerate(pts2):
                if abs(x - px) < 10 and abs(y - py) < 10:
                    selected_idx = i
                    dragging = True
                    break
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            pts2[selected_idx] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            selected_idx = -1

    cv2.namedWindow("Side-by-side Picker")
    cv2.setMouseCallback("Side-by-side Picker", mouse_callback)
    print("🖱️ Left = click to add points. Right = drag to match. Press ENTER when done.")

    while True:
        display = combined.copy()

        # Draw corresponding pairs
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1, pts2)):
            cv2.circle(
                display, (int(x1), int(y1)), radius, (0, 0, 255), -1
            )  # red = left
            cv2.circle(
                display, (int(x2), int(y2)), radius, (0, 255, 0), -1
            )  # green = right
            cv2.line(display, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
            cv2.putText(
                display,
                str(i + 1),
                (int(x1) + 5, int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                display,
                str(i + 1),
                (int(x2) + 5, int(y2) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        cv2.imshow("Side-by-side Picker", display)

        key = cv2.waitKey(20)
        if key == 13:  # ENTER
            break
        elif key == 27:  # ESC
            pts1, pts2 = [], []
            break

    cv2.destroyAllWindows()

    # Convert back to coordinate space relative to each image
    pts1 = np.array(pts1, np.float32)
    pts2 = np.array([(x - offset_x, y) for (x, y) in pts2], np.float32)

    return pts1, pts2
