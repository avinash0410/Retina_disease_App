"""
vision_agent.py

Loads the fine-tuned EfficientNetV2S model once and exposes
predict_and_explain().
"""
import tensorflow as tf

from pathlib import Path
from dotenv import load_dotenv
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
load_dotenv()
import streamlit as st

@st.cache_resource
def load_model():
    model = build_efficientnetv2s_model()

    model.load_weights(WEIGHTS_PATH)

    return model

model = load_model()
from config import (
    MODELS_DIR,
    IMAGE_SIZE,
    CLASS_NAMES,
    CLINICAL_NOTES,
)


INPUT_SHAPE = (384, 384, 3)
NUM_CLASSES = len(CLASS_NAMES)


def build_efficientnetv2s_model():

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                factor=0.03,
                fill_mode="nearest",
                seed=42,
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.03,
                width_factor=0.03,
                fill_mode="nearest",
                seed=42,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.05, 0.05),
                width_factor=(-0.05, 0.05),
                fill_mode="nearest",
                seed=42,
            ),
        ],
        name="data_augmentation",
    )

    backbone = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
        pooling=None,
        name="efficientnetv2-s",
    )

    backbone.trainable = True

    for layer in backbone.layers[:-40]:
        layer.trainable = False

    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inputs = tf.keras.Input(
        shape=INPUT_SHAPE,
        name="input_image",
    )

    x = data_augmentation(inputs)

    x = backbone(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="gap"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="bn"
    )(x)

    x = tf.keras.layers.Dropout(
        0.30,
        name="dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="retinal_oct_efficientnetv2s_baseline",
    )

    return model



from huggingface_hub import hf_hub_download

WEIGHTS_PATH = hf_hub_download(
    repo_id="Avinash0410/retina_APP",
    filename="efficientnetv2s_finetuned_best.weights.h5",
)

print("=" * 60)
print("Building EfficientNetV2S...")
model = build_efficientnetv2s_model()

print("Loading weights...")
model.load_weights(WEIGHTS_PATH)

print("Weights loaded successfully!")
print("=" * 60)



backbone = model.get_layer("efficientnetv2-s")

def find_last_conv_layer(backbone_model):
    for layer in reversed(backbone_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in backbone")


last_conv_layer_name = find_last_conv_layer(backbone)
last_conv_layer = backbone.get_layer(last_conv_layer_name)

feature_extractor = tf.keras.Model(
    inputs=backbone.input,
    outputs=backbone.output,
)
classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
x = model.get_layer("gap")(classifier_input)
x = model.get_layer("bn")(x, training=False)
x = model.get_layer("dropout")(x, training=False)
classifier_output = model.get_layer("predictions")(x)
classifier_model = tf.keras.Model(classifier_input, classifier_output)



def load_image_for_display(image_path):
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def load_image_for_model(image_path, image_size=IMAGE_SIZE):
    img = tf.keras.utils.load_img(image_path, target_size=image_size, color_mode="rgb")
    arr = tf.keras.utils.img_to_array(img).astype("float32")
    return arr


def preprocess_for_backbone(img_array):
    x = tf.convert_to_tensor(img_array[None, ...], dtype=tf.float32)
    return x



def make_gradcam_heatmap(img_array, pred_index=None):
    x = preprocess_for_backbone(img_array)
    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(x, training=False)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), int(pred_index.numpy() if hasattr(pred_index, "numpy") else pred_index)


def make_gradcam_pp_heatmap(img_array, pred_index=None):
    x = preprocess_for_backbone(img_array)

    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_outputs = feature_extractor(x, training=False)
                tape1.watch(conv_outputs)
                tape2.watch(conv_outputs)
                tape3.watch(conv_outputs)
                predictions = classifier_model(conv_outputs, training=False)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]
            first_grads = tape3.gradient(class_channel, conv_outputs)
        second_grads = tape2.gradient(first_grads, conv_outputs)
    third_grads = tape1.gradient(second_grads, conv_outputs)

    conv_outputs = conv_outputs[0]
    first_grads = first_grads[0]
    second_grads = second_grads[0]
    third_grads = third_grads[0]

    global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1), keepdims=True)
    alpha_num = second_grads
    alpha_denom = 2.0 * second_grads + third_grads * global_sum
    alpha_denom = tf.where(tf.abs(alpha_denom) > 1e-8, alpha_denom, tf.ones_like(alpha_denom))
    alphas = alpha_num / (alpha_denom + 1e-8)

    weights = tf.reduce_sum(alphas * tf.nn.relu(first_grads), axis=(0, 1))
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), int(pred_index.numpy() if hasattr(pred_index, "numpy") else pred_index)


def make_eigencam_heatmap(img_array):
    x = preprocess_for_backbone(img_array)
    conv_outputs = feature_extractor(x, training=False)[0]
    conv_outputs = tf.cast(conv_outputs, tf.float32).numpy()

    h, w, c = conv_outputs.shape
    features = conv_outputs.reshape(-1, c).astype(np.float32)
    features = features - features.mean(axis=0, keepdims=True)

    _, _, vt = np.linalg.svd(features, full_matrices=False)
    principal_component = vt[0].astype(np.float32)

    cam = features @ principal_component
    cam = cam.reshape(h, w)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    preds = classifier_model(feature_extractor(x, training=False), training=False).numpy()[0]
    pred_index = int(np.argmax(preds))
    return cam, pred_index


def resize_heatmap(heatmap, target_size):
    heatmap = tf.image.resize(heatmap[..., np.newaxis], target_size)
    return tf.squeeze(heatmap).numpy()


def overlay_heatmap_on_image(image, heatmap, alpha=0.35):
    heatmap_uint8 = np.uint8(255 * heatmap)
    cmap = plt.get_cmap("jet")
    heatmap_color = cmap(heatmap_uint8)[:, :, :3]
    heatmap_color = np.uint8(255 * heatmap_color)
    overlay = np.clip((1 - alpha) * image + alpha * heatmap_color, 0, 255).astype("uint8")
    return overlay



def predict_and_explain(image_path):
    """
    Runs prediction + all three XAI heatmaps on a single image and returns
    everything as data (arrays + primitives) — no plotting, no file I/O,
    so app.py can render it with st.image() / st.write().
    """
    img_for_model = load_image_for_model(image_path)
    img_for_display = load_image_for_display(image_path)

    x = preprocess_for_backbone(img_for_model)
    features = feature_extractor(x, training=False)
    preds = classifier_model(features, training=False).numpy()[0]

    pred_idx = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    pred_conf = float(preds[pred_idx])

    sorted_idx = np.argsort(preds)[::-1]
    top3 = [(CLASS_NAMES[i], float(preds[i])) for i in sorted_idx[:3]]

    gradcam, _ = make_gradcam_heatmap(img_for_model, pred_index=pred_idx)
    gradcam_pp, _ = make_gradcam_pp_heatmap(img_for_model, pred_index=pred_idx)
    eigencam, _ = make_eigencam_heatmap(img_for_model)

    gradcam_resized = resize_heatmap(gradcam, img_for_display.shape[:2])
    gradcam_pp_resized = resize_heatmap(gradcam_pp, img_for_display.shape[:2])
    eigencam_resized = resize_heatmap(eigencam, img_for_display.shape[:2])

    gradcam_overlay = overlay_heatmap_on_image(img_for_display, gradcam_resized)
    gradcam_pp_overlay = overlay_heatmap_on_image(img_for_display, gradcam_pp_resized)
    eigencam_overlay = overlay_heatmap_on_image(img_for_display, eigencam_resized)

    mask = gradcam_pp_resized >= np.quantile(gradcam_pp_resized, 0.85)
    ys, xs = np.where(mask)
    h, w = gradcam_pp_resized.shape

    if len(xs) > 0 and len(ys) > 0:
        cx = float(xs.mean() / w)
        cy = float(ys.mean() / h)
        horiz = "left" if cx < 0.33 else ("central" if cx < 0.66 else "right")
        vert = "upper" if cy < 0.33 else ("middle" if cy < 0.66 else "lower")
        region = f"{vert}-{horiz}"
        spread = (float(xs.std() / w) + float(ys.std() / h)) / 2
        localization = (
            "highly localized" if spread < 0.10
            else "moderately spread" if spread < 0.20
            else "broadly distributed"
        )
        activation_pct = float(mask.mean() * 100)
    else:
        region = "central"
        localization = "minimal"
        activation_pct = 0.0

    clinical_note = CLINICAL_NOTES.get(pred_class, "No clinical notes available.")

    return {
        "predicted_class": pred_class,
        "confidence": pred_conf,
        "top3": top3,
        "region": region,
        "localization": localization,
        "activation_pct": activation_pct,
        "clinical_note": clinical_note,
        "original_image": img_for_display,
        "images": {
            "gradcam_overlay": gradcam_overlay,
            "gradcam_pp_overlay": gradcam_pp_overlay,
            "eigencam_overlay": eigencam_overlay,
        },
    }