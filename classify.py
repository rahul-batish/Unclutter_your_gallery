IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
import cv2 
import os
import tensorflow as tf
import shutil
def prepare(path):
    img_size=128
    img_array=cv2.imread(path)
    new_array=cv2.resize(img_array,IMAGE_SIZE)
    return new_array.reshape(-1,IMAGE_HEIGHT,IMAGE_HEIGHT,3)/255


try:
    os.mkdir("textimages")
    os.mkdir("non_textimages")
except:
    pass
# print(prediction)
print(os.listdir("allimages"))
for filename in os.listdir("allimages"):
    if(filename[-3:]=='jpg'):
        ar=prepare(r"allimages/"+filename)
        model=tf.keras.models.load_model(r"model.model")
        prediction=model.predict([ar])
        # print(prediction)
        if 0.9<prediction[0][0]:
            shutil.copy(r"allimages/"+filename,r"non_textimages/"+filename)
            # print(os.listdir("D:/phone sent/textimages/"))
            # print(filename)
        elif prediction[0][1]>0.9:
            shutil.copy(r"allimages/"+filename,r"textimages/"+filename)
            # print(os.listdir("D:/phone sent/textimages/"))
        
        
