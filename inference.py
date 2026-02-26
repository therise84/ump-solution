import torch
import cv2
import json
import numpy as np
import segmentation_models_pytorch as smp
from skimage.morphology import skeletonize
import networkx as nx
from utils import snap_to_nearest_road, build_graph_from_mask, simplify_path

def load_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

def predict_mask(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    tensor = torch.tensor(img.transpose(2,0,1), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor)
    mask = (pred.squeeze().numpy() > 0.5).astype(np.uint8)
    return mask

def find_path(mask, start, goal):
    skeleton = skeletonize(mask)
    start = snap_to_nearest_road(skeleton, start)
    goal = snap_to_nearest_road(skeleton, goal)

    if start is None or goal is None:
        return []

    G = build_graph_from_mask(skeleton)
    path = nx.astar_path(
        G,
        start,
        goal,
        heuristic=lambda a,b: np.linalg.norm(np.array(a)-np.array(b)),
        weight='weight'
    )
    path = simplify_path(path)
    return path

def run_inference(image_path, start, goal, output_file):
    model = load_model()
    mask = predict_mask(model, image_path)
    path = find_path(mask, start, goal)

    result = {
        "id": "test_case",
        "path": path
    }

    with open(output_file, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    run_inference("test/sats/test_001.tif", (523,1847), (3201,456), "submission.json")