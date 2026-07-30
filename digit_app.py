import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Weka kichwa cha App
st.title("Handwritten Digit Recognizer AI")
st.write(
    "Chora namba yoyote (0 hadi 9) kwenye kisanduku hapa chini. AI itaichunguza"
    " na kuonyesha asilimia zake!"
)

# 1. Eneo la kuchoria (Canvas)
st.subheader("Chora namba yako hapa:")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
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
    # Hapa tunachukua picha iliyochorwa na kuifanyia mabadiliko
    img_data = canvas_result.image_data

    # Angalia kama mtumiaji amechora kweli (kama hajakacha tupu)
    if np.sum(img_data) == 0:
      st.warning(
          "⚠️ Tafadhali chora namba kwanza kabla ya kubonyeza kitufe cha"
          " kutabiri!"
      )
    else:
      # Hapa utaunganisha na model yako halisi ya Machine Learning (Mfano Scikit-Learn au mlinganisho wako)
      # Mfano wa kuiga mantiki ya probabilities (Ikiwa model yako imeshapakiwa):
      # probabilities = model.predict_proba(processed_image)

      # --- SEHEMU YA MANTIKI (LOGIC) YA KUTABIRI ---
      # Kwa mfano wa majaribio, hapa tunaweka mfano wa matokeo:
      # (Badilisha sehemu hii kuunganisha na model yako halisi ya ML uliyoweka)

      # Tunafanya simulation ya kupata namba na asilimia kubwa zaidi
      # Mfano tukipata matokeo ya probabilities kutoka kwenye model yako:
      # predicted_digit = np.argmax(probabilities)
      # confidence = np.max(probabilities) * 100

      # Mfano wa kuweka Kikomo (Threshold) cha kugoma vitu visivyo sahihi:
      kikomo_cha_asilimia = 50.0  # Chini ya 50% inagoma

      # Tuseme tumepata matokeo (Unaweza kubadilisha hapa ukishaiunganisha na model yako)
      # Hapa chini ni mfano tu wa jinsi ya kuandika hiyo code:

      # if confidence < kikomo_cha_asilimia:
      #     st.error("❌ Hii haionekani kuwa namba sahihi! Tafadhali chora vizuri zaidi.")
      # else:
      #     st.success(f"🎉 Namba sahihi ni: **{predicted_digit}** (kwa asilimia {confidence:.1f}%)")
      #
      #     # Kuonyesha asilimia zote za namba 0 hadi 9
      #     st.write("### Uchambuzi wa Asilimia za Kila Namba:")
      #     for i, prob in enumerate(probabilities[0]):
      #         st.progress(float(prob))
      #         st.write(f"Namba **{i}**: {prob * 100:.1f}%")

      st.info(
          "Ubunifu wa kuchora umepokelewa! Unganisha tu na model yako ya"
          " predict_proba hapa ili kutoa matokeo halisi."
      )
