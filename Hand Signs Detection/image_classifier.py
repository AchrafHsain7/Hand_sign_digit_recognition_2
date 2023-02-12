import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
from ConvNet import load_dataset


model = keras.models.load_model('model.tf')
_,_,_,_,classes = load_dataset()
images = ['0_1','0_2','1_1','2_1','3_1','4_1','5_1']

for i in range(len(images)):
    image = np.array(Image.open('testData/'+images[i]+'.jpg').resize((64, 64)))
    image = image.reshape((1,64,64,3))
    result = model.predict(image)
    print(result)
    image = image.reshape((64,64,3))
    plt.imshow(image)
    plt.title(result[0])
    plt.show()
