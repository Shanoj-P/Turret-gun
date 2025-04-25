import cv2
import numpy as np

# === Load stereo calibration parameters ===
cv_file = cv2.FileStorage("stereo_params.xml", cv2.FILE_STORAGE_READ)

K1 = cv_file.getNode("K1").mat()
D1 = cv_file.getNode("D1").mat()
K2 = cv_file.getNode("K2").mat()
D2 = cv_file.getNode("D2").mat()
R = cv_file.getNode("R").mat()
T = cv_file.getNode("T").mat()
R1 = cv_file.getNode("R1").mat()
R2 = cv_file.getNode("R2").mat()
P1 = cv_file.getNode("P1").mat()
P2 = cv_file.getNode("P2").mat()
Q = cv_file.getNode("Q").mat()

cv_file.release()

# === Extract focal length and baseline ===
focal_length_px = P1[0, 0]
baseline_mm = abs(T[0][0])
print(f"Focal length (px): {focal_length_px}")
print(f"Baseline (mm): {baseline_mm}")

# === Open stereo cameras ===
capL = cv2.VideoCapture(2)  # Left camera
capR = cv2.VideoCapture(0)  # Right camera

# === Grab a test frame to get image size ===
retL, frameL = capL.read()
image_size = (frameL.shape[1], frameL.shape[0])

# === Create rectification maps ===
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

# === Stereo SGBM matcher ===
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,  # Must be divisible by 16
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("Error: Could not read frames from cameras.")
        break

    # === Rectify frames ===
    rectL = cv2.remap(frameL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, map2x, map2y, cv2.INTER_LINEAR)

    # === Convert to grayscale ===
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    # === Compute disparity map ===
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # === Reproject to 3D using Q matrix ===
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    # === Visualize disparity ===
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # === Display center pixel depth ===
    h, w = disparity.shape
    disp_center = disparity[h // 2, w // 2]

    if disp_center > 0:
        Z = points_3D[h // 2, w // 2, 2]
        print(f"Center disparity: {disp_center:.2f}, Depth (Z): {Z:.2f} mm")
    else:
        print("Invalid disparity at center pixel.")

    # === Visual overlay ===
    cv2.circle(disp_vis, (w // 2, h // 2), 5, (255, 0, 0), 2)

    # === Show windows ===
    cv2.imshow("Left Rectified", rectL)
    cv2.imshow("Right Rectified", rectR)
    cv2.imshow("Disparity", disp_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
capL.release()
capR.release()
cv2.destroyAllWindows()
