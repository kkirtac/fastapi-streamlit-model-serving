import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import numpy as np
import cv2
import pandas as pd
import plotly.express as px

import streamlit as st

# interact with FastAPI endpoint
api_url = "http://backend:8000/predict"


def process(image, server_url: str):

    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )

    return r


# construct UI layout
st.title("ID card visibility classification")

st.write(
    """This streamlit example uses a FastAPI service as backend.
    Visit this URL at `localhost:8000/docs` for FastAPI documentation."""
)  # description and instructions

uploaded_image = st.file_uploader("upload image")  # image upload widget

if uploaded_image:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # display the image
    st.image(opencv_image, caption='Image', channels="BGR")

if st.button("Predict ID card visibility"):

    if uploaded_image:
        response = process(uploaded_image, api_url)
        st.json(response.json())
        df = pd.DataFrame({k:[v] for k,v in response.json().items()})
        st.table(df)
        fig = px.bar(x=df.loc[0], y=df.columns, orientation='h')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    else:
        # handle case with no image
        st.write("Insert an image!")
