# Bridge Corrosion Detection

A U-Net model finds corrosion in bridge photos.
It marks each pixel as corrosion or not corrosion.

## How it works

1. OpenCV sharpens each photo with an unsharp mask.
2. The photo is resized to 256 × 256 pixels.
3. A U-Net in PyTorch predicts a corrosion mask.
4. Training uses binary cross-entropy loss and the Adam optimizer (learning rate 1e-4).
5. The model trains for 15 epochs with a batch size of 4, on a GPU in Google Colab.

## Dataset

The dataset has 800 photos with semantic masks.
It was exported from Roboflow.
The dataset is not in this repository.

## Run

1. Put the dataset zip in your Google Drive.
2. Open `Bridge_Corrosion_Detection.ipynb` in Google Colab.
3. Set `DRIVE_ZIP_PATH` to the path of your zip.
4. Run the cells in order.
5. In the last cell, upload a photo to see its corrosion mask.

## Stack

Python · PyTorch · OpenCV · NumPy · Google Colab
