import cv2
import numpy as np
import pytesseract
from sklearn.cluster import KMeans

class ColorContrast:
    """
    Analyzes color contrast and accessibility in an image, focusing on text
    and UI elements, based on WCAG guidelines.
    """
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def get_foreground_and_background(self, region, n_colors=2):
        if region.shape[0] < 2 or region.shape[1] < 2:
            return None, None
        pixels = region.reshape((-1, 3)).astype(np.float32)
        kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=0)
        kmeans.fit(pixels)
        colors = [tuple(center.astype(int)) for center in kmeans.cluster_centers_]
        colors.sort(key=lambda c: self.calculate_luminance(c))
        return colors[0], colors[1]

    def calculate_luminance(self, bgr_color):
        rgb = [c / 255.0 for c in reversed(bgr_color)]
        
        for i, c in enumerate(rgb):
            if c <= 0.03928:
                rgb[i] = c / 12.92
            else:
                rgb[i] = ((c + 0.055) / 1.055) ** 2.4
                
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    def get_contrast_ratio(self, color1, color2):
        l1 = self.calculate_luminance(color1)
        l2 = self.calculate_luminance(color2)
        if l1 > l2:
            return (l1 + 0.05) / (l2 + 0.05)
        return (l2 + 0.05) / (l1 + 0.05)

    def analyze_text_contrast(self, min_confidence=50):
        try:
            data = pytesseract.image_to_data(self.image, output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractNotFoundError:
            print("Pytesseract is not installed or not in your PATH.")
            return []

        contrast_results = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > min_confidence and data['text'][i].strip():
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                padding = 5
                roi = self.image[y-padding:y+h+padding, x-padding:x+w+padding]

                if roi.size == 0: continue

                fg_color, bg_color = self.get_foreground_and_background(roi)
                if fg_color is None or bg_color is None: continue
                
                ratio = self.get_contrast_ratio(fg_color, bg_color)
                
                contrast_results.append({
                    'text': data['text'][i],
                    'box': (x, y, w, h),
                    'foreground_color': fg_color,
                    'background_color': bg_color,
                    'contrast_ratio': ratio,
                    'grade': self.grade_contrast(ratio)
                })
        return contrast_results

    def grade_contrast(self, ratio):
        if ratio >= 7.0:
            return "AAA"
        if ratio >= 4.5:
            return "AA"
        return "Fail"

    def visualize_contrast_issues(self, contrast_results):
        output = self.image.copy()
        for result in contrast_results:
            grade = result['grade']
            if grade == "Fail":
                x, y, w, h = result['box']
                ratio = result['contrast_ratio']
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
                label = f"{ratio:.2f}"
                cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return output