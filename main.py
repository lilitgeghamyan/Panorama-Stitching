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

    def trim_black_borders(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            return image[y:y + h, x:x + w]
        return image

    def stitch_pair(self, base_img, next_img):
        kp1, des1 = self.get_features(base_img)
        kp2, des2 = self.get_features(next_img)

        if des1 is None or des2 is None:
            return base_img

        matches = self.flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:
            src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                return base_img

            h1, w1 = base_img.shape[:2]
            h2, w2 = next_img.shape[:2]

            result = cv2.warpPerspective(next_img, M, (w1 + w2, h1 + h2))

            result[0:h1, 0:w1] = base_img

            return self.trim_black_borders(result)
        else:
            print("Could not find enough matches between parts. Skipping one.")
            return base_img

    def stitch_multiple(self, images):
        if not images:
            return None

        panorama = images[0]

        for i in range(1, len(images)):
            print(f"Stitching image {i + 1} of {len(images)}...")
            panorama = self.stitch_pair(panorama, images[i])

        return panorama

def main():
    image_folder = "images2"
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