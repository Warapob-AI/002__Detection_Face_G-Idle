import sys
import os
sys.path.append(os.path.abspath('C:/Machine Learning/002__Detection_Face_G-Idle/Scripts'))

from Scripts.process_models import save_model, load_model
from Scripts.predict_new_image import predict_new_image
from Scripts.load_folder_image import load_folder_image
from Scripts.train_model import train_model, classification

# minnie_array = load_folder_image('Dataset/minnie_face')
# miyeon_array = load_folder_image('Dataset/miyeon_face')
# shuhua_array = load_folder_image('Dataset/shuhua_face')
# yuqi_array = load_folder_image('Dataset/yuqi_face')
# soyeon_array = load_folder_image('Dataset/soyeon_face')

# save_model(minnie_array)
# save_model(miyeon_array)
# save_model(shuhua_array)
# save_model(yuqi_array)
# save_model(soyeon_array)

minnie_array = load_model("002__Detection_Face_G-Idle/Model/minnie_array")
miyeon_array = load_model("002__Detection_Face_G-Idle/Model/miyeon_array")
shuhua_array = load_model("002__Detection_Face_G-Idle/Model/shuhua_array")
yuqi_array = load_model("002__Detection_Face_G-Idle/Model/yuqi_array")
soyeon_array = load_model("002__Detection_Face_G-Idle/Model/soyeon_array")

model, y_test, y_pred = train_model(minnie_array, miyeon_array, shuhua_array, yuqi_array, soyeon_array)
classification(model, y_test, y_pred)
predict_new_image('002__Detection_Face_G-Idle/test_image.png', model)