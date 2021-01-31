from PIL import Image
import albumentations as aug
from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
import os

weights_path = os.getcwd()+"/pneumonia/weights/Pneumonia.pth"

pneumonia= ['Normal', 'Pneumonia']
def predict_pneumonia(file,weights=weights_path,labs=pneumonia):
      model=torch.load(weights)
      model.cpu()
      model.eval()
      image = Image.open(file).convert("RGB")
      image = np.array(image)
      transforms = aug.Compose([
            aug.Resize(224,224),
            aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225),max_pixel_value=255.0,always_apply=True),
            ])
      image = transforms(image=image)["image"]
      image = np.transpose(image, (2, 0, 1)).astype(np.float32)
      image = torch.tensor([image], dtype=torch.float)
      preds = model(image)
      probs  = preds.detach().numpy()[0]
      probs = np.exp(probs)/np.sum(np.exp(probs))
      probs = ["%.10f" % x for x in probs]
      outs={}
      for i in range(len(labs)):
            outs[labs[i]]=probs[i]
      return outs   
