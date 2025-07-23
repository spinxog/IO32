import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from typing import Dict
from analysis.layout_analysis import LayoutAnalysis
from analysis.color_contrast import ColorContrast
from typing import Dict
from typing import List

def enhance_contrast_if_low(image: np.ndarray, issues: List[str], threshold: float = 4.5) -> np.ndarray:
    contrast_checker = ColorContrast(image)
    results = contrast_checker.analyze_text_contrast()

    low_contrast_regions = [res for res in results if res['grade'] == 'Fail']

    if low_contrast_regions:
        issues.append(f"{len(low_contrast_regions)} low-contrast regions found.")
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return enhanced_image
    else:
        issues.append("Contrast is acceptable.")
        return image

def straighten_off_grid(image: np.ndarray, off_grid_elements: list, grid_lines: Dict[str, np.ndarray], threshold: int = 10) -> np.ndarray:
    
    adjusted_image = image.copy()

    for (x, y, w, h) in off_grid_elements:
        new_x = find_closest_grid(x, grid_lines.get('vertical', []), threshold)
        new_y = find_closest_grid(y, grid_lines.get('horizontal', []), threshold)

        if new_x is None:
            new_x = x
        if new_y is None:
            new_y = y

        element_roi = adjusted_image[y:y+h, x:x+w]

        if 0 <= new_y < adjusted_image.shape[0] - h and 0 <= new_x < adjusted_image.shape[1] - w:
            adjusted_image[new_y:new_y+h, new_x:new_x+w] = element_roi

        cv2.rectangle(adjusted_image, (new_x, new_y), (new_x + w, new_y + h), (0, 255, 255), 2)

    return adjusted_image

def find_closest_grid(coord, grid_lines, threshold):
    if grid_lines is None or len(grid_lines) == 0:
        return None
    for line in grid_lines:
        if abs(coord - line[0]) < threshold:
            return int(line[0])
    return None

def reflow_elements(image, elements=None):
    """
    Basic layout reflow that tries to stack elements vertically with consistent spacing.
    """
    layout = LayoutAnalysis()
    if elements is None:
        elements = layout.detect_elements(image)

    if not elements:
        return image
    elements = sorted(elements, key=lambda el: el[1])
    output = image.copy()
    padding = 20
    current_y = padding

    for (x, y, w, h) in elements:
        new_x = (image.shape[1] - w) // 2
        cv2.rectangle(output, (new_x, current_y), (new_x + w, current_y + h), (255, 0, 0), 2)
        current_y += h + padding

    return output
