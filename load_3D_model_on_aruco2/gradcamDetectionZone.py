import cv2
import numpy as np

# retourne la liste des zones rouges (x, y ,intensity)
def detect_red_zone(image):
    gradcam_image = cv2.imread('./assets/'+image)

    image_row, image_cols, x = gradcam_image.shape
    image_size = image_row * image_cols
    reds_zones = []

    hsv = cv2.cvtColor(gradcam_image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    # red_areas = cv2.bitwise_and(gradcam_image, gradcam_image, mask=mask)

    # trouve les contours des zones rouges
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # calcule l'aire de chaque contour pour obtenir le nombre de pixels rouges dans la zone
        area = cv2.contourArea(contour)
        area = area / image_size

        # calcule le centre du contour pour obtenir la position du centre de la zone rouge
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print("x: ",cX," y: ",cY)
            # Normalisation des coordonnées par rapport à la moitié droite ou gauche de l'image @param image
            if cX <= image_cols / 2:
                cX = - cX / (image_cols / 2)
                
            else:
                cX = (cX - (image_cols / 2)) / (image_cols / 2)
            cY = cY / image_row
            reds_zones.append((cX, cY, area))
        
    return reds_zones

# res = detect_red_zone("cam_test.jpg")

# for zone in res:
#     print("x: ",zone[0]," y: ",zone[1]," size: ",zone[2])

    

    # cv2.drawContours(gradcam_image, contours, -1, (0, 255, 0), 2)
    # cv2.imshow('Red Areas Contours', gradcam_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
