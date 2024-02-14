import os
import glob
from copy import deepcopy

import cv2
import numpy as np

import torch

from typing import Sequence

import nibabel
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results, deprecation_warn, Boxes
from ultralytics.utils.plotting import Annotator, colors

class Visual(Results):
    def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None) -> None:
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.path = path
        self.names = names
        self.boxes = boxes
        self.masks = masks
        self.probs = probs
        self.keypoints = keypoints
        self.save_dir = None
        self._keys = ('boxes', 'masks', 'probs', 'keypoints') 
        self.speed = {'preprocess': None, 'inference': None, 'postprocess': None}

    def plot(
            self,
            conf=True,
            pixdim=None,
            line_width=None,
            font_size=None,
            font='Arial.ttf',
            pil=False,
            img=None,
            im_gpu=None,
            kpt_radius=5,
            kpt_line=True,
            labels=True,
            boxes=True,
            masks=True,
            probs=True,
            **kwargs  # deprecated args TODO: remove support in 8.2
    ):
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).cpu().contiguous() * 255).to(torch.uint8).numpy()

        # Deprecation warn TODO: remove in 8.2
        if 'show_conf' in kwargs:
            deprecation_warn('show_conf', 'conf')
            conf = kwargs['show_conf']
            assert isinstance(conf, bool), '`show_conf` should be of boolean type, i.e, show_conf=True/False'

        if 'line_thickness' in kwargs:
            deprecation_warn('line_thickness', 'line_width')
            line_width = kwargs['line_thickness']
            assert isinstance(line_width, int), '`line_width` should be of int type, i.e, line_width=3'

        names = self.names
        pred_boxes, show_boxes = self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names)

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device).permute(
                    2, 0, 1).flip(0).contiguous() / 255
            idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect results
        if pred_boxes and show_boxes:
            for d in reversed(pred_boxes):
                w, h = d.xywh[:, 2:].cpu().tolist()[0]
                size = max(w*pixdim[1], h*pixdim[2])
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ('' if id is None else f'id:{id} ') + names[c]
                # label = (f'{name} {conf:.2f}' if conf else name) if labels else None
                label = f'{name}({size*0.1:.2f}cm)'
                annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = ',\n'.join(f'{names[j] if names else j} {pred_probs.data[j]:.2f}' for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        return annotator.result()


if __name__ == "__main__":
    data_path = "preprocessing/nii2video/*"
    nii_path = "RawData/MSD/imagesVa"
    label_path = "RawData/MSD/labelsVa"
    videos = glob.glob(data_path)
    nii_images = os.listdir(nii_path)
    
    os.makedirs("good_results", exist_ok=True)  # Saving good result frame
    os.makedirs("results", exist_ok=True) # Saving all frame
    
    alpha = 0.8
    beta = 0.7
    label_color = (255, 0, 0)
    
    model = YOLO("runs/detect/cyst/weights/best.pt")
    label_num = 2
    
    for video in videos:
        nii_name = f"pancreas_{int(os.path.basename(video).split('.')[0].split('_')[1]):03d}.nii.gz"
        if nii_name in nii_images:
            cap = cv2.VideoCapture(video)
            nii = nibabel.load(os.path.join(nii_path, nii_name))
            pixdim = nii.header['pixdim']
            labels = nibabel.load(os.path.join(label_path, nii_name))
            labels = labels.get_fdata()
            labels = (labels == label_num).astype(np.uint8)
            
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                output: Sequence[Results] = model(frame)
                output: Results = output[0]
                output = Visual(
                    orig_img=output.orig_img,
                    path=output.path,
                    names=output.names,
                    boxes=output.boxes,
                    masks=output.masks,
                    probs=output.probs,
                    keypoints=output.keypoints,
                )
                result = output.plot(conf=True, pixdim=pixdim)
                label = labels[:, :, i]
                
                if len(np.unique(label)) > 1:
                    # label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
                    mask = np.zeros_like(result)
                    mask[label == label_num] = label_color
                    result = cv2.addWeighted(result, alpha, mask, beta, 0.0)
                    cv2.imwrite(os.path.join("good_results", os.path.basename(video).split(".")[0] + f"_{i}.jpg"), result)

                cv2.imwrite(os.path.join("results", os.path.basename(video).split(".")[0]+f"_{i}.jpg"), result)
                
                i += 1