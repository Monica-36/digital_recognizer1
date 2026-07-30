import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Kichwa cha App
st.title("Handwritten Digit Recognizer AI (Real)")
st.write(
    "Chora namba yoyote (0 hadi 9) katikati ya kisanduku kisha ubofye Tabiri"
    " Namba!"
)


# Tunapakia dataset ya MNIST na kufundisha model halisi ya Deep Learning
@st.cache_resource
def load_and_train_real_model():
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # Tunapunguza ukubwa wa picha ziwe kati ya 0 na 1
  x_train, x_test = x_train / 255.0, x_test / 255.0

  # Tunatengeneza mtandao wa Neural Network wenye nguvu kubwa ya utambuzi
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation="softmax"),
  ])

  model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"],
  )

  # Tunafanya mafunzo kwa haraka (epochs 3) ili app iwe tayari
  model.fit(x_train, y_train, epochs=3, verbose=0)
  return model


# Ujumbe wa kupakia AI Model
with st.spinner(
    "Inapakia na kuisanidi AI Model halisi... Tafadhali subiri kidogo."
):
  model = load_and_train_real_model()

# 1. Eneo la kuchoria (Canvas)
st.subheader("Chora namba yako hapa:")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=22,
    stroke_color="white",
    background_color="black",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

# 2. Kitufe cha kutabiri
if st.button("Tabiri Namba (Predict)"):
  if canvas_result.image_data is not None:
    img_data = canvas_result.image_data

    if np.sum(img_data) == 0:
      st.warning("⚠️ Tafadhali chora namba kwanza kwenye kisanduku!")
    else:
      # Badilisha picha kuwa Grayscale
      gray = cv2.cvtColor(img_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)

      # Tafuta mipaka ya mchoro (Bounding Box) ili tuikate vizuri
      contours, _ = cv2.findContours(
          gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
      )

      if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        if w > 5 and h > 5:
          # Kata eneo la namba pekee (Crop)
          roi = gray[y : y + h, x : x + w]

          # Ibadili iwe saizi inayoeleweka na MNIST (28x28 pixels)
          resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

          # Sahihisha uwiano wa rangi na muundo
          normalized = resized / 255.0
          reshaped = np.reshape(normalized, (1, 28, 28))

          # Pata majibu sahihi kutoka kwenye TensorFlow
          predictions = model.predict(reshaped)
          predicted_digit = np.argmax(predictions[0])
          confidence = np.max(predictions[0]) * 100

          # Onyesha jibu la hakika kabisa
          st.success(
              f"🎉 Namba sahihi uliyochora ni: **{predicted_digit}** (kwa"
              f" uhakika wa {confidence:.1f}%)"
          )

          # Onyesha uchambuzi wa asilimia zote za namba 0 hadi 9
          st.write("### Uchambuzi wa Asilimia za Kila Namba:")
          for i, prob in enumerate(predictions[0]):
            st.progress(float(prob))
            st.write(f"Namba **{i}**: {prob * 100:.1f}%")
        else:
          st.error("Mchoro ni mdogo sana, tafadhali chora kwa uwazi zaidi.")
      else:
        st.error("Hatujaona mchoro wowote.")
