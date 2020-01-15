from pathlib import Path

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Flatten

from datetime import datetime

from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus

from self_supervised_3d_tasks.algorithms.contrastive_predictive_coding import network_cpc
import keras

from self_supervised_3d_tasks.models.cnn_baseline import KaggleGenerator
from self_supervised_3d_tasks.custom_preprocessing.cpc_preprocess import preprocess

aquire_free_gpus()

epochs = 10,
code_size = 128,
lr = 1e-3,
terms = 3,
predict_terms = 3,
image_size = 46,
batch_size = 8
n_channels = 3
crop_size = 186
split_per_side = 7
img_shape = (image_size, image_size, n_channels)

model, enc_model = network_cpc(image_shape=img_shape, terms=terms,
                               predict_terms=predict_terms, code_size=code_size, learning_rate=lr)

model.load_weights('/home/Julius.Severin/workspace/self-supervised-transfer-learning/cpc_retina/'
                   'weights-improvement-1000-0.48.hdf5')

# we need so many models
layer_in = [keras.Input(img_shape)] * (split_per_side * split_per_side)
layer_out = [enc_model(x) for x in layer_in]

x = Flatten()(layer_out)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
x = Dense(5, activation="sigmoid")(x)

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.summary()

f_train = lambda x, y: preprocess(x, crop_size, split_per_side)
f_val = lambda x, y: preprocess(x, crop_size, split_per_side, is_training=False)

gen = KaggleGenerator(batch_size=64, split=0.66, shuffle=False, pre_proc_func_train=f_train,
                      pre_proc_func_val=f_val)


output = (
        Path(f"~/workspace/cpc_test/run_{datetime.now()}/").expanduser().resolve()
    )
output.mkdir(parents=True, exist_ok=True)
output = output / f"model.hdf5"

checkp = ModelCheckpoint(
    str(output.with_name("intermediate_{epoch:04d}_{acc:.2f}_" + output.name)),
    monitor="accuracy",
    period=1,
    mode="max",
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=10, min_lr=0.001
)

model.fit_generator(
    generator=gen,
    epochs=500,
    callbacks=[checkp, reduce_lr],
    validation_data=gen.get_val_data(),
)
model.save(output)