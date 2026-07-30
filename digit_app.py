import streamlit as st
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title("Handwritten Digit Recognition App")
st.write("Hii ni app inayotambua namba zilizoandikwa kwa mkono kwa kutumia Machine Learning.")

@st.cache_resource
def load_and_train_model():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

with st.spinner("Inapakua na kuandaa Model... Tafadhali subiri kidogo."):
    model, X_test, y_test = load_and_train_model()

st.success("Model iko tayari!")

# Kitufe cha kuchagua picha ya mfano
index = st.slider("Chagua namba ya picha kwenye mfumo (Test Index):", 0, len(X_test)-1, 0)

sample_image = X_test[index]
actual_label = y_test[index] if not hasattr(y_test, 'iloc') else y_test.iloc[index]
predicted_label = model.predict([sample_image])[0]

# Onyesha picha kwenye Streamlit
fig, ax = plt.subplots()
ax.imshow(sample_image.reshape(28, 28), cmap='gray')
ax.axis('off')

st.pyplot(fig)
st.write(f"### Namba halisi (Actual): **{actual_label}**")
st.write(f"### AI imetabiri (Predicted): **{predicted_label}**")