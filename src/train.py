
#############################################################################################
#############################################################################################
                                        #IMPORTS
#############################################################################################
#############################################################################################

import json
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from imgaug import augmenters as iaa
from time import time



#############################################################################################
#############################################################################################
                                        #AUGMENTATION
#############################################################################################
#############################################################################################

def custom_augmentation():
    return  iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Crop(px=(0, 16)),
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5), # horizontally flip 50% of all images
            iaa.LinearContrast((0.75, 1.5)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, rotate=(-25, 25), shear=(-8, 8)),
        ])
    
#############################################################################################
#############################################################################################
                                        #PARAMETERS
#############################################################################################
#############################################################################################

model_selected ='mobilenet_segnet' #@param  ['fcn_8','fcn_32','fcn_8_vgg','fcn_32_vgg','fcn_8_resnet50','fcn_32_resnet50','fcn_8_mobilenet','fcn_32_mobilenet','pspnet',	'vgg_pspnet','resnet50_pspnet','unet_mini','unet','vgg_unet','resnet50_unet','mobilenet_unet','segnet','vgg_segnet','resnet50_segnet','mobilenet_segnet', 'pretrained_resnet50_pspnet']

train_image_dir = "data/data_processed/train/original_images"  #@param {type:"string"}
train_mask_dir = "data/data_processed/train/label_images_semantic" #@param {type:"string"}

val_image_dir = "data/data_processed/val/original_images" #@param {type:"string"}
val_mask_dir = "data/data_processed/val/label_images_semantic" #@param {type:"string"}

classes = 24 #@param {type:"number"}
epochs =   20#@param {type:"number"}

input_height = 400 #@param {type:"number"}
input_width =  600 #@param {type:"number"}

input_width =int(input_width)
input_height =int(input_height)

do_augment = True #@param {type:"boolean"}
validation = True #@param {type:"boolean"}


save_model_path = 'model' #@param {type:"string"}

checkpointspath =  '/content/checkspoints/' #@param {type:"string"}
logs_dir = 'metrics/training_logs' #@param {type:"string"}

model_name= str(model_selected)+'_augmentation='+str(do_augment)+'_'+str(input_height)+'*'+str(input_width)+'_'+str(epochs)+'epochs' 

if model_selected =='fcn_8':

    from keras_segmentation.models.fcn import fcn_8
    model = fcn_8(n_classes=classes )

if model_selected =='resnet50_pspnet':

    from keras_segmentation.models.pspnet import resnet50_pspnet
    model = resnet50_pspnet(n_classes=classes )

if model_selected =='mobilenet_segnet':

    from keras_segmentation.models.segnet import mobilenet_segnet
    model = mobilenet_segnet(n_classes=classes )

if model_selected =='resnet50_unet':

    from keras_segmentation.models.unet import resnet50_unet
    model = resnet50_unet(n_classes=classes )


if model_selected =='pretrained_resnet50_pspnet':
    from keras_segmentation.models.model_utils import transfer_weights
    from keras_segmentation.pretrained import pspnet_50_ADE_20K
    from keras_segmentation.models.pspnet import pspnet_50

    pretrained_model = pspnet_50_ADE_20K()
    model = pspnet_50( n_classes=24 )
    transfer_weights( pretrained_model , model  ) # transfer weights from pre-trained model to your model


callbacks = [   
    tf.keras.callbacks.ModelCheckpoint(checkpointspath, save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir=logs_dir),
    EarlyStopping(monitor="val_loss",patience=5)
]


#############################################################################################
#############################################################################################
                                   #TRAINING
#############################################################################################
#############################################################################################

print(f'Training for {model_name}' )
start = time()

history = model.train(
    train_images = train_image_dir,
    train_annotations = train_mask_dir,
    validate=validation,
    input_width=input_width,
    input_height=input_height,
    val_images=val_image_dir,
    val_annotations=val_mask_dir,  
    epochs=epochs,
    do_augment=do_augment, # enable augmentation 
    custom_augmentation=custom_augmentation,
    # sets the augmention function to use 
    callbacks=callbacks        
)

end = time() - start
print(f'Training time : {end}')



model.save(save_model_path)   # Saving model



train_info={'train_time':end,
            'model':model_selected,
            'n_classes':classes,
            'input_height':input_height,
            'input_width':input_width,
            'do_augment':do_augment,
            'earlystop':True,
            'epochs':epochs
}


with open('metrics/training_info.json', 'w') as f:
    json.dump(train_info, f)