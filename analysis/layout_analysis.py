import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import easyocr
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms
from models.salicon_model import Salicon


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
salicon_net = Salicon()
salicon_path = './models/salicon_model.pth'
assert os.path.exists(salicon_path), "Missing SALICON model"
salicon_net.load_state_dict(torch.load(salicon_path))
salicon_net.eval().cuda()

class LayoutAnalysis:
    def __init__(self, threshold=10, max_grid_lines=10):
        self.threshold = threshold
        self.max_grid_lines = max_grid_lines

    def detect_elements(self, image):
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elements = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 100]
        return elements

    def cluster_edges(self, elements, axis='x'):
        if not elements:
            return np.array([])

        edges = []
        if axis == 'x':
            for x, y, w, h in elements:
                edges.extend([x, x + w])
        else: 
            for x, y, w, h in elements:
                edges.extend([y, y + h])

        if not edges:
            return np.array([])
            
        unique_edges = np.array(list(set(edges))).reshape(-1, 1)
        n_clusters = min(self.max_grid_lines, len(unique_edges))
        if n_clusters < 1:
            return np.array([])

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(unique_edges)
        return np.sort(kmeans.cluster_centers_, axis=0)

    def detect_visual_grid(self, elements):
        grid_lines = {
            'vertical': self.cluster_edges(elements, axis='x'),
            'horizontal': self.cluster_edges(elements, axis='y')
        }
        
        off_grid = [el for el in elements if not self.is_on_grid(el, grid_lines)]
        return off_grid, grid_lines

    def is_on_grid(self, element, grid_lines):
        x, y, w, h = element
        
        for line in grid_lines['vertical']:
            if abs(x - line[0]) < self.threshold or abs((x + w) - line[0]) < self.threshold:
                return True
        for line in grid_lines['horizontal']:
            if abs(y - line[0]) < self.threshold or abs((y + h) - line[0]) < self.threshold:
                return True
                
        return False

    def analyze_visual_balance(self, image, elements=None):
        height, width = image.shape[:2]
        mid_x, mid_y = width / 2, height / 2
        
        quadrants = {
            'top_left': [], 'top_right': [],
            'bottom_left': [], 'bottom_right': []
        }

        if elements is None:
            elements = self.detect_elements(image)

        for x, y, w, h in elements:
            cx, cy = x + w / 2, y + h / 2
            if cx < mid_x and cy < mid_y:
                quadrants['top_left'].append((w, h))
            elif cx >= mid_x and cy < mid_y:
                quadrants['top_right'].append((w, h))
            elif cx < mid_x and cy >= mid_y:
                quadrants['bottom_left'].append((w, h))
            else:
                quadrants['bottom_right'].append((w, h))

        return self.calculate_balance(quadrants)

    def calculate_balance(self, quadrants):
        visual_weights = {quad: sum(w * h for w, h in elems) for quad, elems in quadrants.items()}
        
        total_weight = sum(visual_weights.values())
        if total_weight == 0:
            return False, {k: 0 for k in visual_weights}

        normalized = {k: v / total_weight for k, v in visual_weights.items()}
        imbalance = (max(normalized.values()) - min(normalized.values())) > 0.25
        
        return imbalance, normalized

    def analyze_whitespace(self, elements, image):
        height, width = image.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.uint8)

        for x, y, w, h in elements:
            cv2.rectangle(heatmap, (x, y), (x + w, y + h), 255, -1)

        whitespace_map = cv2.bitwise_not(heatmap)
        total_pixels = width * height
        whitespace_pixels = cv2.countNonZero(whitespace_map)
        density = whitespace_pixels / total_pixels if total_pixels > 0 else 0

        contours, _ = cv2.findContours(whitespace_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        whitespace_regions = [cv2.boundingRect(cnt) for cnt in contours]

        return density, whitespace_regions

    def draw_layout_overlay(self, image, elements, grid_lines=None, off_grid_elements=None):
        overlay = image.copy()
        
        for x, y, w, h in elements:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if off_grid_elements:
            for x, y, w, h in off_grid_elements:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if grid_lines:
            for line in grid_lines.get('vertical', []):
                x = int(line[0])
                cv2.line(overlay, (x, 0), (x, image.shape[0]), (255, 0, 0), 1)
            for line in grid_lines.get('horizontal', []):
                y = int(line[0])
                cv2.line(overlay, (0, y), (image.shape[1], y), (255, 0, 0), 1)
                
        return overlay

    def generate_report(self, image):
        elements = self.detect_elements(image)
        off_grid, grid_lines = self.detect_visual_grid(elements)
        imbalance_flag, balance_scores = self.analyze_visual_balance(image, elements)
        whitespace_density, whitespace_regions = self.analyze_whitespace(elements, image)

        vertical_lines = grid_lines['vertical'].flatten().tolist() if grid_lines['vertical'].size > 0 else []
        horizontal_lines = grid_lines['horizontal'].flatten().tolist() if grid_lines['horizontal'].size > 0 else []

        return {
            "total_elements": len(elements),
            "off_grid_elements": off_grid,
            "grid_lines": {
                "vertical": vertical_lines,
                "horizontal": horizontal_lines
            },
            "balance": {
                "is_imbalanced": imbalance_flag,
                "scores": balance_scores
            },
            "whitespace": {
                "density": whitespace_density,
                "regions": whitespace_regions
            }
        }
    
def generate_text_heatmap(image_path, output_size):
    result = reader.readtext(image_path)
    heatmap = np.zeros(output_size[::-1], dtype=np.float32)
    for (bbox, text, conf) in result:
        if conf < 0.3:  #type: ignore 
            continue
        pts = np.array(bbox).astype(np.int32)
        rect = cv2.boundingRect(pts)
        _, _, w, h = rect
        area = w * h
        size_weight = max(area / (output_size[0] * output_size[1]), 0.2)
        cv2.fillPoly(heatmap, [pts], color=size_weight)  #type: ignore 
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), sigmaX=10)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    return (heatmap * 255).astype(np.uint8)

def detect_shapes(image_path, output_size):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, output_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    heatmap = dilated.astype(np.float32) / 255.0
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    return (heatmap * 255).astype(np.uint8)

def predict_saliency(image_path, model, device, output_path=None):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    preprocess_coarse = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor()
    ])
    preprocess_fine = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor()
    ])
    coarse = preprocess_coarse(image).unsqueeze(0).to(device) #type: ignore
    fine = preprocess_fine(image).unsqueeze(0).to(device) #type: ignore
    with torch.no_grad():
        saliency = salicon_net(fine, coarse)

    saliency = saliency.squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
    saliency_img = (saliency * 255).astype(np.uint8)
    saliency_resized = cv2.resize(saliency_img, original_size, interpolation=cv2.INTER_CUBIC)

    if output_path:
        cv2.imwrite(output_path, saliency_resized)

    return saliency_resized, image

def fuse_all_maps(saliency_map, text_heatmap, shape_heatmap, alpha=0.5, beta=0.3, gamma=0.2):
    fused = (saliency_map.astype(np.float32) * alpha +
             text_heatmap.astype(np.float32) * beta +
             shape_heatmap.astype(np.float32) * gamma)
    return np.clip(fused, 0, 255).astype(np.uint8)