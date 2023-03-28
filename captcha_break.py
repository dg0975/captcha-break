from tensorflow import keras

import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras import layers


KERAS_MODEL = keras.models.load_model(f'{sys.path[0]}/captchaV2') # path for model
image = "captcha.jpg"


def get_predicted_captcha(image):
     ###########################################################
     characters = ['0', '2', '4', '7', '1', '5', '8', '6', '3', '9']

     char_to_num = layers.experimental.preprocessing.StringLookup(
      vocabulary=list(characters), num_oov_indices=0, mask_token=None
     )

     # Mapping integers back to original characters
     num_to_char = layers.experimental.preprocessing.StringLookup(
      vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
     )

     imgPath = f"{sys.path[0]}/" + image
     img = tf.io.read_file(imgPath)

     # 2. Decode and convert to grayscale
     img = tf.io.decode_png(img, channels=1)
     img = tf.image.convert_image_dtype(img, tf.float32)

     # 4. Resize to the desired size
     img = tf.image.resize(img, [50, 200])

     # 5. Transpose the image because we want the time
     # dimension to correspond to the width of the image.
     img = tf.transpose(img, perm=[1, 0, 2])
     img = tf.expand_dims(img, 0)

     preds = KERAS_MODEL.predict([img])
     pred_texts = decode_batch_predictions(preds, num_to_char)
     return pred_texts[0][:6]


def decode_batch_predictions(pred, num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search

    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :10]

    # Iterate over the results and get back the text
    output_text = []
    for res in results:
         res = tf.strings.reduce_join(num_to_char(res + 1)).numpy().decode("utf-8")
         output_text.append(res)
    return output_text


if __name__ == "__main__":
    get_predicted_captcha(image)