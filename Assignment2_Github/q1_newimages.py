#Jayneel Pratap Jain    
#101983237
#jpjain@myseneca.ca

import os
import glob
import numpy as np
import joblib
from skimage.io import imread
from skimage.transform import resize

IMG_SIZE = (128, 128)

MODEL_PATH = "q1_best_cat_dog_model.joblib"

TEST_FOLDER = "new images" 
def preprocess_image(path):
    img = imread(path)


    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    img_resized = resize(img, IMG_SIZE, anti_aliasing=True)

    if img_resized.ndim == 3:
        img_gray = np.dot(img_resized[..., :3], [0.299, 0.587, 0.114])
    else:
        img_gray = img_resized

    return img_gray.ravel()

def main():
    # Load trained model
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")

    image_paths = []
    image_paths.extend(glob.glob(os.path.join(TEST_FOLDER, "*.jpg")))
    image_paths.extend(glob.glob(os.path.join(TEST_FOLDER, "*.jpeg")))
    image_paths.extend(glob.glob(os.path.join(TEST_FOLDER, "*.png")))

    if not image_paths:
        print("No images found in folder:", TEST_FOLDER)
        return

    X = []
    for path in image_paths:
        X.append(preprocess_image(path))

    X = np.array(X)

    preds = model.predict(X)

    print("\nPredictions on NEW IMAGES:\n")
    for path, pred in zip(image_paths, preds):
        print(f"{os.path.basename(path)} → {pred}")

if __name__ == "__main__":
    main()
