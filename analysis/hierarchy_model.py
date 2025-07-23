import cv2
import numpy as np

class HierarchyModel:
    """
    Analyzes visual hierarchy using both rule-based scoring and a classic
    saliency model (with optional local saliency support).
    """
    def __init__(self, weights={'size': 0.5, 'position': 0.3, 'center_bias': 0.2}, saliency_map=None):
        self.weights = weights
        self.saliency_map = saliency_map
        self.saliency_model = cv2.saliency.StaticSaliencySpectralResidual_create() if saliency_map is None else None #type: ignore

    def analyze_visual_hierarchy(self, elements, image_shape):
        if not elements:
            return {}

        scores = {}
        for element in elements:
            score = self._calculate_normalized_score(element, image_shape)
            if self.saliency_map is not None:
                sal_score = self._get_saliency_score(element)
                score = (score + sal_score) / 2  # Combine rule-based and saliency
            scores[element] = score
        return scores

    def _calculate_normalized_score(self, element, image_shape):
        x, y, w, h = element
        img_h, img_w = image_shape[:2]
        norm_size = (w * h) / (img_w * img_h)
        norm_position = 1.0 - (y / img_h)
        img_center = np.array([img_w / 2, img_h / 2])
        elem_center = np.array([x + w / 2, y + h / 2])
        dist_from_center = np.linalg.norm(elem_center - img_center)
        max_dist = np.linalg.norm(np.array([0, 0]) - img_center)
        norm_center_bias = 1.0 - (dist_from_center / max_dist)

        score = (
            self.weights['size'] * norm_size +
            self.weights['position'] * norm_position +
            self.weights['center_bias'] * norm_center_bias
        )
        return score

    def _get_saliency_score(self, element):
        x, y, w, h = element
        hmap = self.saliency_map[y:y+h, x:x+w] #type: ignore
        if hmap.size == 0:
            return 0
        return np.mean(hmap) / 255.0  

    def get_saliency_map(self, image):
        success, saliency_map = self.saliency_model.computeSaliency(image) #type: ignore
        if success:
            return (saliency_map * 255).astype("uint8")
        return np.zeros(image.shape[:2], dtype=np.uint8)
