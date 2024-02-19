# export_predictions.py
# Creates patch-wise predictions for each region in the test set
# and saves them to disk. Also saves manually calculated metrics.

import os
import numpy as np
import torch
import glob
import cv2
import pandas as pd
from tqdm import tqdm
import models as m
from pprint import pprint
import math

def parse_model(fname):
    return {
            "composition": fname.split('-')[-1].split('.')[0],
            "loss": fname.split('-')[0].split('/')[-1]
        }

def load_model(fname):
    parsed = parse_model(fname)

    if parsed['loss'] == "FocalLoss_GammaDot5":
        if parsed['composition'] == "All+NDVI":
            return m.DeepLabV3Plus_FocalLoss_GammaDot5_8ch.load_from_checkpoint(fname)
        else:
            return m.DeepLabV3Plus_FocalLoss_GammaDot5_4ch.load_from_checkpoint(fname)
    elif parsed['loss'] == "FocalLoss_Gamma2":
        if parsed['composition'] == "All+NDVI":
            return m.DeepLabV3Plus_FocalLoss_Gamma2_8ch.load_from_checkpoint(fname)
        else:
            return m.DeepLabV3Plus_FocalLoss_Gamma2_4ch.load_from_checkpoint(fname)
    elif parsed['loss'] == "Baseline":
        if parsed['composition'] == "All+NDVI":
            return m.DeepLabV3Plus_Baseline_8ch.load_from_checkpoint(fname)
        else:
            return m.DeepLabV3Plus_Baseline_4ch.load_from_checkpoint(fname)
    elif "TverskyLoss" in parsed['loss']:
        if parsed['composition'] == "All+NDVI":
            return m.DeepLabV3Plus_TverskyLoss_8ch.load_from_checkpoint(fname)
        else:
            return m.DeepLabV3Plus_TverskyLoss_4ch.load_from_checkpoint(fname)
    
    raise ValueError(f"Model {fname} not recognized")

def patchify(array, patch_size, stride):
    height, width, _ = array.shape
    patches = []
    patch_counts = []
    for x in range(0, height, stride):
        for y in range(0, width, stride):
            # Crop the patch from the input image
            patch = array[x:x + patch_size[0], y:y + patch_size[1], :]
            if patch.shape[0] != patch_size[0] or patch.shape[1] != patch_size[1]:
                # print(f'Padding patch at {x}, {y} with shape {patch.shape}')
                bottompad = patch_size[0] - patch.shape[0]
                rightpad = patch_size[1] - patch.shape[1]
                patch = np.pad(patch, ((0, bottompad), (0, rightpad), (0, 0)), mode='reflect')
            patches.append(patch)
            patch_counts.append((x, y))
    return {"patches": patches, "patch_counts": patch_counts}

patch_size = (256, 256)
stride = 256 

if __name__ == "__main__":
    # Get models
    models = glob.glob('./models/*')

    # Check if predictions folder exists
    os.makedirs('predictions', exist_ok=True)

    # Create a dataframe for the metrics
    results_metrics = pd.DataFrame(columns=["Region", "Composition", "Loss", "Precision", "Recall", "F1", "Accuracy", "IoU"])

               
    for modelpath in tqdm(models):
        for region in ["x01", "x03"]:
            image = np.load(f'data/scenes_allbands_ndvi/allbands_ndvi_{region}.npy')
            # Normalize image
            image = (image - np.min(image)) / (np.max(image) - np.min(image))

            truth = np.load(f'data/truth_masks/truth_{region}.npy')
            # Adjust to binary segmentation
            truth = np.where(truth == 2, 0, 1)
            height, width, _ = image.shape
            width = math.ceil(width / stride) * stride
            height = math.ceil(height / stride) * stride
    
            info = parse_model(modelpath)
            composition = info['composition']
            loss = info['loss']

            bands = [int(i) for i in composition] if composition != "All+NDVI" else list(range(1, 9))
            bands = [i - 1 for i in bands]
            image = image[:, :, bands]

            # Load model
            model = load_model(modelpath)
            model = model.to('cuda')
            model.eval()

            # Patchify image
            patchified_image = patchify(image, patch_size, stride)
            image_patches = patchified_image["patches"]
            image_patchcounts = patchified_image["patch_counts"]
            # Patchify truth
            patchified_truth = patchify(truth, patch_size, stride)
            truth_patches = patchified_truth["patches"]
            truth_patchcounts = patchified_truth["patch_counts"]
            assert len(image_patches) == len(truth_patches), "Image and truth patches don't match"

            # Iterate through patches and perform predictions
            predicted_masks = []
            for patch, (x, y) in zip(image_patches, image_patchcounts):
                patch = torch.tensor(patch, device='cuda').permute(2, 0, 1).unsqueeze(0).float()
                with torch.no_grad():
                    prediction = model(patch)
                    prediction = torch.sigmoid(prediction)
                    prediction = (prediction > 0.5)
                predicted_masks.append((prediction, (x, y)))
                # cv2.imwrite(f'predictions/{region}_{composition}_{loss}_{x}_{y}.png', prediction.squeeze().cpu().numpy() * 255)
            
            # Stitch patches together
            stitched_mask = np.zeros((height, width), dtype=np.uint8)
            for mask, (x, y) in predicted_masks:
                mask = mask.squeeze().cpu().numpy()
                stitched_mask[x:x + patch_size[0], y:y + patch_size[1]] = mask
            
            stitched_truth = np.zeros((height, width), dtype=np.uint8)
            for truthpatch, (x, y) in zip(truth_patches, truth_patchcounts):
                truthpatch = truthpatch.squeeze()
                stitched_truth[x:x + patch_size[0], y:y + patch_size[1]] = truthpatch
            
            # Build confusion mask
            confusion_mask = np.zeros((height, width, 3))
            true_positive_color = (1, 1, 1)
            false_positive_color = (0, 0, 1)
            false_negative_color = (1, 0, 0)
            true_negative_color = (0, 0, 0)

            # print(np.unique(stitched_mask), stitched_mask.shape)
            # print(np.unique(truth), truth.shape)

            true_positives = np.logical_and(stitched_mask == 1, stitched_truth == 1)
            false_positives = np.logical_and(stitched_mask == 1, stitched_truth == 0)
            false_negatives = np.logical_and(stitched_mask == 0, stitched_truth == 1)
            true_negatives = np.logical_and(stitched_mask == 0, stitched_truth == 0)
            confusion_mask[true_positives] = true_positive_color
            confusion_mask[false_positives] = false_positive_color
            confusion_mask[false_negatives] = false_negative_color
            confusion_mask[true_negatives] = true_negative_color

            # Calculate metrics for whole image
            true_positives = np.sum(true_positives)
            false_positives = np.sum(false_positives)
            false_negatives = np.sum(false_negatives)
            true_negatives = np.sum(true_negatives)
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1 = 2 * (precision * recall) / (precision + recall)
            accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)
            iou = true_positives / (true_positives + false_positives + false_negatives)

            # Extend lower part of confusion mask for writing text
            confusion_mask = np.pad(confusion_mask, ((0, 100), (0, 0), (0, 0)), mode='constant', constant_values=0)
            confusion_mask = (confusion_mask * 255).astype(np.uint8)

            # Write metrics on image
            metrics_str = f"Precision: {precision :.2f}\nRecall: {recall :.2f}\nF1: {f1 :.2f}\nAccuracy: {accuracy :.2f}\nIoU: {iou :.2f}"
            cv2.putText(confusion_mask, metrics_str, (0, height + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            confusion_mask = cv2.cvtColor(confusion_mask, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'predictions/{region}_{composition}_{loss}.png', confusion_mask)
            
            # Save metrics to dataframe
            new_row = pd.DataFrame.from_records([{
                "Region": region,
                "Composition": composition,
                "Loss": loss,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Accuracy": accuracy,
                "IoU": iou
            }])
            results_metrics = pd.concat([results_metrics, new_row], ignore_index=True)

    results_metrics = results_metrics.sort_values(by=["Region", "IoU"], ascending=False)
    results_metrics.to_csv(f'predictions/metrics.csv', index=False)
    results_metrics.to_excel(f'predictions/metrics.xlsx', index=False)

