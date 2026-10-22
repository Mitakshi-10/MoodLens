import time, numpy as np, keras
from keras import layers, models
from src.data import build_dataset
from src import config

train = build_dataset("train","cnn",training=True).take(125)   # 8000 images
val   = build_dataset("validation","cnn",training=False).take(8)

def make(order, aug=False):
    inp = layers.Input((48,48,1))
    x = inp
    if aug:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.2, fill_mode="reflect")(x)
        x = layers.Rescaling(1.0)(x)
    x = layers.Rescaling(1.0/255.0)(x)
    for f,bn in [(32,True),(64,True),(128,False)]:
        x = layers.Conv2D(f,3,padding="same")(x)
        if order=="bn_first":
            if bn: x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        else:
            x = layers.ReLU()(x)
            if bn: x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        if f<128: x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x); x = layers.Dense(128,activation="relu")(x)
    x = layers.Dropout(0.5)(x); x = layers.Dense(7,activation="softmax")(x)
    m = models.Model(inp,x)
    m.compile(keras.optimizers.Adam(1e-3),"categorical_crossentropy",metrics=["accuracy"])
    return m

for order in ["relu_first","bn_first"]:
    for cw in [None, "cw"]:
        keras.utils.set_random_seed(42)
        m = make(order)
        t=time.time()
        h = m.fit(train, validation_data=val, epochs=3, verbose=0,
                  class_weight=(config.class_weights() if cw=="cw" else None))
        print(f"{order:11s} {cw:3s} train_acc {h.history['accuracy'][-1]:.4f} "
              f"val_acc {h.history['val_accuracy'][-1]:.4f} loss {h.history['loss'][-1]:.4f} "
              f"({time.time()-t:.0f}s)", flush=True)
