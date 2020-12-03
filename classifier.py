# IMPORT LIBRARIES
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

# IMPORT DATA
x = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")

# SPLIT THE DATA
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 2500,train_size = 7500,random_state = 9)

# MAKE EVERY DATASET IN ONE UNIT

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

# TRAIN THE MODEL

clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial').fit(X_train_scaled,y_train)

# PREDICT THE RESULT

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized,pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel,0,255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_predict = clf.predict(test_sample)
    print("test predict ", test_predict)
    return test_predict[0]
