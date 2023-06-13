import mmcv
import numpy as np
from mmengine import load

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import CameraInstance3DBoxes

# img = mmcv.imread('./data/kitti_orig/training/image_2/000000.png')
# img = mmcv.imconvert(img, 'bgr', 'rgb')


# print("img.shape: ", img.shape)

# 1/0

info_file = load('./data/kitti/kitti_infos_trainval.pkl')
# 000000
# print("info file keys: ", info_file['data_list'][0].keys())
# 1/0
cam2img = np.array(info_file['data_list'][0]['images']['CAM2']['cam2img'], dtype=np.float32)
bboxes_3d = []
for instance in info_file['data_list'][0]['instances']:
    bboxes_3d.append(instance['bbox_3d'])


print("KEYS: ", info_file['data_list'][0]['images']['CAM2'].keys())
print("bboxes 3d: ", bboxes_3d)



#############################################################################################


lidar2cam = np.array(info_file['data_list'][0]['images']['CAM2']['lidar2cam'], dtype=np.float32)
lidar2img = np.array(info_file['data_list'][0]['images']['CAM2']['lidar2img'], dtype=np.float32)

# cam2img = copy.deepcopy(input_meta['cam2img'])
# corners_3d = bboxes_3d.corners
# num_bbox = corners_3d.shape[0]
# points_3d = corners_3d.reshape(-1, 3)
# if not isinstance(cam2img, torch.Tensor):
#     cam2img = torch.from_numpy(np.array(cam2img))

# assert (cam2img.shape == torch.Size([3, 3])
#         or cam2img.shape == torch.Size([4, 4]))
# cam2img = cam2img.float().cpu()

# # print("cam2img: ", cam2img)
# # project to 2d to get image coords (uv)
# uv_origin = points_cam2img(points_3d, cam2img)
# uv_origin = (uv_origin - 1).round()
# imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()
# 1/0

cam2img = np.matmul(lidar2img, np.linalg.inv(lidar2cam))
print("cam2img: ", cam2img)

#############################################################################################



gt_bboxes_3d = np.array(bboxes_3d, dtype=np.float32)
gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d)
input_meta = {'cam2img': cam2img}

visualizer = Det3DLocalVisualizer()

img = mmcv.imread('./data/kitti/training/image_2/000000.png')
img = mmcv.imconvert(img, 'bgr', 'rgb')


print("img.shape: ", img.shape)



# FOR PROJECTED 3D BOUNDING BOXES
visualizer.set_image(img)

# project 3D bboxes to image
visualizer.draw_proj_bboxes_3d(gt_bboxes_3d, input_meta)


# FOR BIRD'S EYE VIEW
# visualizer.set_bev_image()
# # draw bev bboxes
# visualizer.draw_bev_bboxes(gt_bboxes_3d, edge_colors='orange')


new_img = visualizer.get_image()
mmcv.imwrite(new_img[..., ::-1], "test_viz_3d.png")
# visualizer.show()



