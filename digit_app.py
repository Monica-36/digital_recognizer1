import cv2
import numpy as np
import streamlit as st
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from streamlit_drawable_canvas import st_canvas

st.title("Handwritten Digit Recognizer AI (SVC)")
st.write("Chora namba yoyote (0 hadi 9) katikati ya kisanduku kisha ubofye Tabiri!")


@st.cache_resource
def get_trained_svc_model():
  digits = load_digits()
  X, y = digits.data, digits.target
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  # Tunatumia SVC yenye uwezo mkubwa wa kutofautisha maumbo
  model = SVC(probability=True, kernel="rbf", gamma=0.001, C=10.0)
  model.fit(X_train, y_train)
  return model


model = get_trained_svc_model()

# Eneo la kuchoria
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=25,
    stroke_color="white",
    background_color="black",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Tabiri Namba (Predict)"):
  if canvas_result.image_data is not None:
    img_data = canvas_result.image_data

    if np.sum(img_data) == 0:
      st.warning("⚠️ Tafadhali chora namba kwanza kwenye kisanduku!")
    else:
      gray = cv2.cvtColor(img_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)

      # Kukata eneo lililochorwa (Crop) ili kuondoa nafasi tupu za pembeni
      contours, _ = cv2.findContours(
          gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
      )

      if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        if w > 5 and h > 5:
          roi = gray[y : y + h, x : x + w]

          # Kubadilisha kuwa ukubwa wa 8x8 unaoeleweka na mfumo
          resized = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_AREA)

          flattened = resized.flatten()
          scaled_image = np.array([16 - (flattened / 255.0) * 16])

          # Kupata matokeo ya hakika kutoka kwenye Model
          probabilities = model.predict_proba(scaled_image)
          predicted_digit = np.argmax(probabilities)
          confidence = np.max(probabilities) * 100

          st.success(
              f"🎉 Namba sahihi uliyochora ni: **{predicted_digit}** (kwa"
              f" uhakika wa {confidence:.1f}%)"
          )

          st.write("### Uchambuzi wa Asilimia za Kila Namba:")
          for i, prob in enumerate(probabilities[0]):
            st.progress(float(prob))
            st.write(f"Namba **{i}**: {prob * 100:.1f}%")
        else:
          st.error("Mchoro ni mdogo sana. Tafadhali chora kwa uwazi.")
      else:
        st.error("Hatujaona mchoro wowote kwenye kisanduku.")
