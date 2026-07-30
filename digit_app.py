import cv2
import numpy as np
import sklearn
import streamlit as st
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from streamlit_drawable_canvas import st_canvas

# Weka kichwa cha App
st.title("Handwritten Digit Recognizer AI")
st.write(
    "Chora namba yoyote (0 hadi 9) kwenye kisanduku hapa chini. AI itaichunguza"
    " na kuonyesha asilimia zake!"
)


# Tunatengeneza na kufundisha model hapo hapo
@st.cache_resource
def train_model():
  digits = load_digits()
  X, y = digits.data, digits.target
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  model = SVC(probability=True, gamma=0.001)
  model.fit(X_train, y_train)
  return model


model = train_model()

# 1. Eneo la kuchoria (Canvas)
st.subheader("Chora namba yako hapa:")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
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

    # Angalia kama mtumiaji amechora kweli (kama hajakacha tupu)
    if np.sum(img_data) == 0:
      st.warning(
          "⚠️ Tafadhali chora namba kwanza kabla ya kubonyeza kitufe cha"
          " kutabiri!"
      )
    else:
      # Badilisha picha kuwa grayscale na ubadili ukubwa uwe 8x8 pixels
      gray = cv2.cvtColor(img_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)
      resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)

      # Geuza iwe rangi inayolingana na model ya digits
      flattened = resized.flatten()
      scaled_image = np.array([16 - (flattened / 255.0) * 16])

      # Pata utabiri na asilimia zake
      probabilities = model.predict_proba(scaled_image)
      predicted_digit = np.argmax(probabilities)
      confidence = np.max(probabilities) * 100

      # Onyesha moja kwa bila kugoma yoyote
      st.success(
          f"🎉 Namba sahihi ni: **{predicted_digit}** (kwa asilimia"
          f" {confidence:.1f}%)"
      )

      # Onyesha uchambuzi wa asilimia zote za namba 0 hadi 9
      st.write("### Uchambuzi wa Asilimia za Kila Namba:")
      for i, prob in enumerate(probabilities[0]):
        st.progress(float(prob))
        st.write(f"Namba **{i}**: {prob * 100:.1f}%")
