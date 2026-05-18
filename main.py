import cv2
import numpy as np
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
    valid_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.svg')

    if not os.path.exists(image_folder):
        print(f"Error: The folder '{image_folder}' does not exist.")
        return

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
            print(f"Loaded: {f}")

    if len(imgs) < 2:
        print("Need at least 2 imgs in the folder to stitch.")
        return

    print(f"\nPanorama construction with {len(imgs)} imgs")
    stitcher = PanoramaStitcher()
    final_result = stitcher.stitch_multiple(imgs)

    if final_result is not None:
        cv2.imwrite("final_panorama_result.jpg", final_result)

        cv2.imshow("Multi-Part Panorama", final_result)
        print("\nResult saved as 'final_panorama_result.jpg'")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Stitching failed.")

if __name__ == "__main__":
    main()