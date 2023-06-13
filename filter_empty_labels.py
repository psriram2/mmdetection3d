import os

LABEL_DIR = "/home/psriram2/mmdetection3d/data/kitti/training/label_2"
IMG_DIR = "/home/psriram2/mmdetection3d/data/kitti/training/image_2"
CALIB_DIR = "/home/psriram2/mmdetection3d/data/kitti/training/calib"
cnt = 0
for f in os.listdir(LABEL_DIR):
    curr_file = os.path.join(LABEL_DIR, f)

    if os.stat(curr_file).st_size == 0:
        print(f"f: {f}")

        # os.remove(os.path.join(IMG_DIR, f[:-4] + ".png"))
        # os.remove(os.path.join(CALIB_DIR, f[:-4] + ".png"))
        # os.remove(curr_file)
        cnt += 1

print(f"empty files: {cnt}/{len(os.listdir(LABEL_DIR))}")