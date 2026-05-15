import cv2
import os

class PanoramaStitcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def get_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(gray, None)

def main():
    image_folder = "images"
    valid_formats = ('.jpg', '.jpeg', '.png', '.bmp')

    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(valid_formats)
    ])

    imgs = []
    for f in image_files:
        full_path = os.path.join(image_folder, f)
        img = cv2.imread(full_path)
        if img is not None:
            imgs.append(img)

    stitcher = PanoramaStitcher()
    final_result = stitcher.stitch_multiple(imgs)
    cv2.imwrite("final_panorama_result.jpg", final_result)

if __name__ == "__main__":
    main()