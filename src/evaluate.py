
import glob
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img


#############################################################################################
#############################################################################################
                                        #EVALUATION
#############################################################################################
#############################################################################################



model_selected ='pretrained_resnet50_pspnet' #@param  ['fcn_8','fcn_32','fcn_8_vgg','fcn_32_vgg','fcn_8_resnet50','fcn_32_resnet50','fcn_8_mobilenet','fcn_32_mobilenet','pspnet',	'vgg_pspnet','resnet50_pspnet','unet_mini','unet','vgg_unet','resnet50_unet','mobilenet_unet','segnet','vgg_segnet','resnet50_segnet','mobilenet_segnet', 'pretrained_resnet50_pspnet']

img_height =  1600#@param {type:"number"}
img_width =  2400 #@param {type:"number"}
classes = 24 #@param  {type:"number"}

if model_selected =='resnet50_pspnet':
    from keras_segmentation.models.pspnet import resnet50_pspnet
    model = resnet50_pspnet(24)
    model = resnet50_pspnet(n_classes=classes)
    model.load_weights('/content/drone_segmenter/model')

if model_selected =='pretrained_resnet50_pspnet':
    from keras_segmentation.models.pspnet import pspnet_50
    model = pspnet_50( n_classes=classes )
    model.load_weights('/content/drone_segmenter/model')

if model_selected =='mobilenet_segnet':
    from keras_segmentation.models.segnet import mobilenet_segnet
    model = mobilenet_segnet(n_classes=classes )
    model.load_weights('/content/drone_segmenter/model')

if model_selected =='resnet50_unet':
    from keras_segmentation.models.unet import resnet50_unet
    model = resnet50_unet(n_classes=classes )
    model.load_weights('/content/drone_segmenter/model')



val_image_dir = "data/data_processed/val/original_images" #@param {type:"raw"}
val_mask_dir = "data/data_processed/val/label_images_semantic" #@param {type:"raw"}

val_input_img_paths = glob.glob(val_image_dir+'/*.jpg')
val_target_img_paths = glob.glob(val_mask_dir+'/*.png')

val_input_img_paths.sort()
val_target_img_paths.sort()

print(f'Validation -- Number of images: {len(val_input_img_paths)}\nNumber of masks: {len(val_target_img_paths)}')





from time import time
print(f'Evaluation for {model_selected}' )


start_eval = time()

model_result = model.evaluate_segmentation( inp_images_dir=val_image_dir  , annotations_dir=val_mask_dir )

print(f'Results on eval data : {model_result}')
end_eval = time() - start_eval
print(f'Evaluation time : {end_eval}')

import json
eval_metrics={'model_selected':model_selected,
         'n_classes':classes,
         'input_height':img_height,
         'input_width':img_width,
         'eval_time':end_eval,
         'frequency_weighted_IU':model_result["frequency_weighted_IU"],
         'mean_IU':model_result["mean_IU"],
         'class_wise_IU_0':model_result["class_wise_IU"][0],
         'class_wise_IU_1':model_result["class_wise_IU"][1],
         'class_wise_IU_2':model_result["class_wise_IU"][2],
         'class_wise_IU_3':model_result["class_wise_IU"][3],
         'class_wise_IU_4':model_result["class_wise_IU"][4],
         'class_wise_IU_5':model_result["class_wise_IU"][5],
         'class_wise_IU_6':model_result["class_wise_IU"][6],
         'class_wise_IU_7':model_result["class_wise_IU"][7],
         'class_wise_IU_8':model_result["class_wise_IU"][8],
         'class_wise_IU_9':model_result["class_wise_IU"][9],
         'class_wise_IU_10':model_result["class_wise_IU"][10],
         'class_wise_IU_11':model_result["class_wise_IU"][11],
         'class_wise_IU_12':model_result["class_wise_IU"][12],
         'class_wise_IU_13':model_result["class_wise_IU"][13],
         'class_wise_IU_14':model_result["class_wise_IU"][14],
         'class_wise_IU_15':model_result["class_wise_IU"][15],
         'class_wise_IU_16':model_result["class_wise_IU"][16],
         'class_wise_IU_17':model_result["class_wise_IU"][17],
         'class_wise_IU_18':model_result["class_wise_IU"][18],
         'class_wise_IU_19':model_result["class_wise_IU"][19],
         'class_wise_IU_20':model_result["class_wise_IU"][20],
         'class_wise_IU_21':model_result["class_wise_IU"][21],
         'class_wise_IU_22':model_result["class_wise_IU"][22],
         'class_wise_IU_23':model_result["class_wise_IU"][23]
}


with open('metrics/eval_metrics.json', 'w') as f:
    json.dump(eval_metrics, f)


#############################################################################################
#############################################################################################
                            #SAVE OUTPUT
#############################################################################################
#############################################################################################

import cv2
img_width =int(img_width)
img_height =int(img_height)

def resize_dataset(out):
    img = out
    resized_img = cv2.resize(img, dsize=(img_width, img_height))
    return resized_img

for i in range(10,15):
    sample_image = image.img_to_array(image.load_img(f'{val_input_img_paths[i]}'))/255.
    sample_mask = image.img_to_array(image.load_img(f'{val_target_img_paths[i]}', color_mode = "grayscale"))
    sample_mask = np.squeeze(sample_mask)
    out = model.predict_segmentation(inp=val_input_img_paths[i])
    out = out.astype('float32')
    out= resize_dataset(out)  

    fig =plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title('Image')
    ax.imshow(sample_image)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('True mask')
    ax2.imshow(sample_mask)

    ax1 = fig.add_subplot(1, 3, 3)
    ax1.set_title('predicted_Mask')
    ax1.imshow(out)
    plt.savefig("/content/drone_segmenter/output/predict_"+str(i)+".jpg")