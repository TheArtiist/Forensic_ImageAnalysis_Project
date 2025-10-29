from PIL import Image, ImageChops
import cv2

def ela_image(path, quality=90):
    original = Image.open(path)
    original.save("resaved.jpg", "JPEG", quality=quality)
    resaved = Image.open("resaved.jpg")
    ela = ImageChops.difference(original, resaved)
    ela = ela.point(lambda x: x * 10)
    ela.save("ela.jpg", "JPEG", quality=quality)
    ela.show()


#ela_image("christmas_tree_stallone_star_decoration.jpg")

def detect_duplicates(image_path):
    img = cv2.imread(image_path, 0)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des, des)
    matches = [m for m in matches if m.distance < 30 and m.queryIdx != m.trainIdx]
    result = cv2.drawMatches(img, kp, img, kp, matches, None)
    cv2.imshow("Detected Duplicates", result)
    cv2.waitKey(0)

detect_duplicates("christmas_tree_stallone_star_decoration.jpg")