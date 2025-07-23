import sys
import os
from typing import Optional, Dict, List, Any
import cv2
import torch
import numpy as np
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.layout_analysis import LayoutAnalysis, predict_saliency
from analysis.color_contrast import ColorContrast
from models.salicon_model import Salicon



class FeedbackGenerator:
    def __init__(self, model_name: str = "llama3", max_tokens: int = 300):
        self.model_name = model_name
        self.max_tokens = max_tokens

    def generate_feedback(
        self,
        layout_report: Dict[str, Any],
        contrast_results: List[Dict[str, Any]],
        saliency_map: Optional[Any] = None,
    ) -> str:
        prompt = self.compose_prompt(layout_report, contrast_results, saliency_map)

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": self.max_tokens}
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.RequestException as e:
            return f"Error generating feedback from LLM API: {e}"

    def compose_prompt(
        self,
        layout_report: Dict[str, Any],
        contrast_results: List[Dict[str, Any]],
        saliency_map: Optional[Any] = None,
    ) -> str:
        prompt = (
            "You are an AI assistant helping users improve their Adobe Express designs.\n"
            "Analyze the following layout and color contrast issues and provide clear actionable feedback:\n\n"
        )
        prompt += self._compose_layout_summary(layout_report)
        prompt += self._compose_contrast_summary(contrast_results)

        if saliency_map is not None:
            prompt += "\nNote: The design includes visual focus areas identified by saliency mapping.\n"

        prompt += "\nBased on these issues, suggest specific improvements to the user.\n"

        return prompt

    def _compose_layout_summary(self, layout_report: Dict[str, Any]) -> str:
        total_elements = layout_report.get("total_elements", 0)
        off_grid_elements = layout_report.get("off_grid_elements", [])
        off_grid_count = len(off_grid_elements) if isinstance(off_grid_elements, list) else 0

        imbalance = layout_report.get("balance", {}).get("is_imbalanced", False)
        whitespace_density = layout_report.get("whitespace", {}).get("density", 0.0)

        return (
            f"- Total visual elements detected: {total_elements}\n"
            f"- Elements off grid alignment: {off_grid_count}\n"
            f"- Visual balance problem detected: {'Yes' if imbalance else 'No'}\n"
            f"- Whitespace density: {whitespace_density:.2f}\n"
        )

    def _compose_contrast_summary(self, contrast_results: List[Dict[str, Any]]) -> str:
        failed_contrast = [r for r in contrast_results if r.get("grade") == "Fail"]
        return f"- Text regions with low contrast: {len(failed_contrast)}\n"


def run_full_analysis(image_path: str, device: torch.device):
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise FileNotFoundError(f"Image not found or failed to load: {image_path}")

    net = Salicon().to(device)
    model_path = "models/salicon_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saliency model not found at {model_path}")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    saliency_map, pil_image = predict_saliency(image_path, net, device)
    orig_img_processed = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    layout = LayoutAnalysis()
    elements = layout.detect_elements(orig_img_processed)
    off_grid_elements, grid_lines = layout.detect_visual_grid(elements)
    imbalance_flag, balance_scores = layout.analyze_visual_balance(orig_img_processed, elements)
    whitespace_density, whitespace_regions = layout.analyze_whitespace(elements, orig_img_processed)

    layout_report = {
        "total_elements": len(elements),
        "off_grid_elements": off_grid_elements,
        "grid_lines": {
            "vertical": grid_lines['vertical'].flatten().tolist() if grid_lines['vertical'].size > 0 else [],
            "horizontal": grid_lines['horizontal'].flatten().tolist() if grid_lines['horizontal'].size > 0 else []
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

    contrast_checker = ColorContrast(orig_img_processed)
    contrast_results = contrast_checker.analyze_text_contrast()

    return orig_img_processed, layout_report, contrast_results, saliency_map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run design analysis and generate AI feedback.")
    parser.add_argument("--image_path", required=True, help="Path to design image to analyze.")
    parser.add_argument("--model_name", default="llama3", help="Name of the local Ollama model (e.g., llama3, mistral).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_img, layout_report, contrast_results, saliency_map = run_full_analysis(args.image_path, device)

    generator = FeedbackGenerator(model_name=args.model_name)
    feedback = generator.generate_feedback(layout_report, contrast_results, saliency_map)

    print("\n=== AI Feedback ===\n")
    print(feedback)