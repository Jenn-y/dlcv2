import cv2
import numpy as np

from depth_estimation.main import get_depth_map

scene_image = cv2.imread('./Frames/14.png')
depth_map = get_depth_map(scene_image)
grey_ball = cv2.imread('./grey_ball.png', cv2.IMREAD_UNCHANGED)

def place_object(background, overlay, x, y):
    h, w = overlay.shape[:2]
    roi = background[y:y+h, x:x+w]

    overlay_image = overlay[..., :3]  
    alpha_mask = overlay[..., 3] / 255.0 

    roi_combined = overlay_image * alpha_mask[..., np.newaxis] + roi * (1 - alpha_mask[..., np.newaxis])
    background[y:y+h, x:x+w] = roi_combined

    return background

x, y = 200, 200
result = place_object(scene_image, grey_ball, x, y)

cv2.imwrite('scene_with_ball.png', result)
cv2.imshow('Scene with Grey Ball', result)
cv2.waitKey(0)
cv2.destroyAllWindows()