import numpy as np
import cv2

# Exemple de paramètres intrinsèques et extrinsèques
# intrinsics = np.array([[focal_length_x, 0, optical_center_x],
#                        [0, focal_length_y, optical_center_y],
#                        [0, 0, 1]])

# extrinsics = np.array([[rotation_matrix], [translation_vector]])

cap = cv2.VideoCapture(0)
focal_length = cap.get(3)
center = (cap.get(3)/2, cap.get(4)/2)
intrinsics = np.array(
                [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype = "double"
                )

distorsion = np.zeros((4,1))

# Enregistrement des paramètres dans un fichier .npz
np.savez("Camera.npz", mtx=intrinsics, dist=distorsion)
