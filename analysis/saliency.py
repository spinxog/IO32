import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import torch
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from models.salicon_model import Salicon
from integration.express_api_client import ExpressAPIClient
from integration.project_retriever import ProjectRetriever
from analysis.layout_analysis import (
    generate_text_heatmap, detect_shapes, fuse_all_maps, predict_saliency, LayoutAnalysis
)
from analysis.hierarchy_model import HierarchyModel
from auto_fix.fix_algorithms import enhance_contrast_if_low, straighten_off_grid, reflow_elements
from auto_fix.slider_controls import contrast_slider_window, grid_snap_slider_window


# TODO: replace with real OAuthCallback  handling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
client = ExpressAPIClient()
retriever = ProjectRetriever(client)
auth_url, _ = client.get_auth_url()
print(" Authenticated by visiting:", auth_url)
#client.fetch_token('https://yourapp.com/callback?...code=...')
retriever.client.session = client.session
net = Salicon().to(device)
net.load_state_dict(torch.load('models/salicon_model.pth', map_location=device))
net.eval()

def get_latest_image(folder='express_exported', extensions=('*.jpg', '*.png', '*.jpeg')):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    if not files:
        raise FileNotFoundError("No image found in the specified folder.")
    return max(files, key=os.path.getctime)

def analyze_and_fix(image_path):
    
    sal_map, orig_img = predict_saliency(image_path, net, device)
    text_map = generate_text_heatmap(image_path, sal_map.shape[::-1])
    shape_map = detect_shapes(image_path, sal_map.shape[::-1])
    fused_map = fuse_all_maps(sal_map, text_map, shape_map)
    orig_img_np = np.array(orig_img)
    layout = LayoutAnalysis()
    elements = layout.detect_elements(orig_img)
    hierarchy = HierarchyModel(saliency_map=fused_map)
    scores = hierarchy.analyze_visual_hierarchy(elements, orig_img.shape) #type:ignore
    orig_img_np = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR)
    off_grid, grid_lines = layout.detect_visual_grid(elements)
    
    contrast_issues = []  


    img_fixed = enhance_contrast_if_low(orig_img_np, contrast_issues)
    img_fixed = straighten_off_grid(img_fixed, off_grid, grid_lines)
    img_fixed_positions = reflow_elements(elements, img_fixed.shape) #type:ignore 


    contrast_slider_window(orig_img, contrast_issues)
    grid_snap_slider_window(img_fixed, off_grid, grid_lines)

    overlay = layout.draw_layout_overlay(img_fixed, elements, grid_lines, off_grid)
    for box, sc in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"{box}: score {sc:.3f}")
    return overlay

def show_overlay(saliency_map, original_image):
    heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    orig_np = np.array(original_image)
    heatmap = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]), interpolation=cv2.INTER_CUBIC)
    overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    img_path = get_latest_image(folder='express_exported')
    print("Using latest image:", img_path)
    overlay = analyze_and_fix(img_path)
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/final_overlay.png", overlay)
    try:
        cv2.imshow("Review Fixes", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("Unable to display image window. Check the output folder.")