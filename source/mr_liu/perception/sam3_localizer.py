"""SAM3 concept detection supplies masks directly to the fast tracking loop."""
import torch
from find_and_track.types import Detection


class Sam3Localizer:
    def __init__(self, checkpoint, conf=.4):
        from ultralytics.models.sam import SAM3SemanticPredictor
        self.predictor = SAM3SemanticPredictor(overrides={
            'model': str(checkpoint), 'device': '0', 'task': 'segment', 'mode': 'predict',
            'conf': conf, 'quantize': 16, 'save': False, 'verbose': False,
        })
        self.predictor.setup_model(model=None, verbose=False)

    def locate(self, bgr, labels):
        # Calling with source refreshes image features; never reuse an old camera frame.
        with torch.autocast('cuda', dtype=torch.float16):
            results = self.predictor(source=bgr, text=list(labels), stream=False)
        found = []
        for result in results:
            if result.masks is None:
                continue
            masks = result.masks.data.cpu().numpy().astype(bool)
            for box, mask in zip(result.boxes, masks):
                label = result.names[int(box.cls.item())]
                detection = Detection(box.xyxy[0].float().cpu().numpy(), label,
                                      float(box.conf.item()), source='slow')
                found.append((detection, mask))
        return found
