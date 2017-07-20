# deepvision

Image classification with convolutional neural networks.
Transfer learning on top of VGG16 using Keras + TensorFlow.

## Results
- 94.2% validation accuracy on a 5-class flower dataset (25 epochs)
- Data augmentation: rotation, shift, flip

## Requirements
- Python 3.5+
- TensorFlow 1.2
- Keras 2.0

## Usage
```bash
pip install -r requirements.txt

# Train
python train.py --data ./data/train --epochs 25

# Predict
python predict.py --image cat.jpg

# Evaluate
python evaluate.py --model model.h5 --data ./data/test
```

## Architecture
```
VGG16 (frozen) → Flatten → Dense(256) → Dropout(0.5) → Softmax
```

## License
MIT
