# deepvision

Image classification with convolutional neural networks.
Transfer learning on top of VGG16 using Keras + TensorFlow.

## Requirements
- Python 3.5+
- TensorFlow 1.2
- Keras 2.0

## Quick Start
```bash
pip install -r requirements.txt
python train.py --data ./data/train --epochs 25
python predict.py --image cat.jpg
```
