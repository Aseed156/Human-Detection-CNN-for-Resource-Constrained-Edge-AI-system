#import the libraries

import tensorflow as tf
from tensorflow.keras import layers,models
import numpy as np
import pandas as pd
import os,random,shutil
from pycocotools.coco import COCO
from google.colab import drive

drive.mount('/content/drive')

#download coco datset

!wget -q http://images.cocodataset.org/zips/train2017.zip
!wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip

!unzip -q train2017.zip
!unzip -q annotations_trainval2017.zip

print("Images:", os.path.exists("/content/train2017"))
print("Annotations:", os.path.exists("/content/annotations/instances_train2017.json"))

shutil.rmtree("/content/ds", ignore_errors=True)

os.makedirs("/content/ds/persons", exist_ok=True)
os.makedirs("/content/ds/non_persons", exist_ok=True)

sort_images(person_ids, "persons")
sort_images(non_person_ids, "non_persons")

print(" Done")

print("Persons:", len(os.listdir("/content/ds/persons")))
print("Non-persons:", len(os.listdir("/content/ds/non_persons")))

coco = COCO('/content/annotations/instances_train2017.json')

print("COCO loaded")

#get person category from datset

person_category_id=coco.getCatIds(catNms=['person'])

# Keep only images where person occupies >10% area

person_ids=coco.getImgIds(catIds=person_category_id)

clean_person_ids = []

for img_id in person_ids:

    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=person_category_id)
    anns = coco.loadAnns(ann_ids)
    img_info = coco.loadImgs(img_id)[0]

    img_area = img_info['width'] * img_info['height']

    for ann in anns:
        if ann['area'] / img_area > 0.1:
            clean_person_ids.append(img_id)
            break

non_person_ids = []

for img_id in coco.getImgIds():

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    has_person = any(ann['category_id'] in person_category_id for ann in anns)

    if not has_person:
        non_person_ids.append(img_id)

#same dataset

person_ids=random.sample(clean_person_ids,3000)
non_person_ids=random.sample(non_person_ids,3000)

print(f"Person images: {len(person_ids)} | Non-person images: {len(non_person_ids)}")

#delete any corrupted folders

shutil.rmtree("/content/ds/persons")
shutil.rmtree("/content/ds/non_persons")

print("Deleted!")

# make two folders for person and non person

os.makedirs("/content/ds/persons" , exist_ok=True)
os.makedirs("/content/ds/non_persons",exist_ok=True)

#copy the images into folders now

def sort_images (Img_ids,label):

    copied=0

    for Img_id in Img_ids:

        img_information=coco.loadImgs(Img_id)[0]

        src="/content/train2017/" + img_information['file_name']

        dst= f"/content/ds/{label}/" + img_information['file_name']

        if os.path.exists(src):

            shutil.copy(src,dst)
            copied +=1

    print (f"copied {copied} images to '{label}' folder")

sort_images(person_ids,"persons")
sort_images(non_person_ids,"non_persons")

# train the data by tensorflow

train_data=tf.keras.preprocessing.image_dataset_from_directory
(
    "/content/ds",
    image_size=(64,64),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=42
)

validate_data=tf.keras.preprocessing.image_dataset_from_directory
(
    "/content/ds",
    image_size=(64,64),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=42
)

print ("Class names:", train_data.class_names)

tf.keras.backend.clear_session()

#buildig cnn now

model = models.Sequential([

    layers.Input(shape=(64,64,3)),

    layers.Rescaling(1./255),

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),

    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

model.summary()

#compile the model

model.compile
(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),

    loss='binary_crossentropy',

    metrics=['accuracy']
)

#early stopping

early_stop = tf.keras.callbacks.EarlyStopping
(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

#train the model

model.fit
(
    train_data,

    validation_data=validate_data,

    epochs=25,

    callbacks=[early_stop]
)

model.save("human_detection_model.keras")

print("model saved")

#evaluate the model to give best accuracy

loss,accuracy=model.evaluate(validate_data)

print(f"validation accuracy :{accuracy *100:.2f}%")
