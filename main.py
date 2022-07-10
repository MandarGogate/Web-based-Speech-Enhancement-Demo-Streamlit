import io
import os
from hashlib import md5
from os.path import join
from time import localtime

import numpy as np
import soundfile as sf
import librosa
import streamlit as st
from asteroid.models import BaseModel

def add_prefix(filename):
    prefix = md5(str(localtime()).encode('utf-8')).hexdigest()
    return f"{prefix}_{filename}"

# load your SE model
model_se = BaseModel.from_pretrained("cankeles/ConvTasNet_WHAMR_enhsingle_16k")

st.set_page_config(layout="wide", page_title="Speech Enhancement Demo", page_icon="🎙")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            #the-title {text-align: center}
            .stButton>button{
                display: block;
                width: 100%;
                text-align: left;
                padding-top: 25px;
                padding-bottom: 25px;
                padding-left: 50px;
                background-color: rgb(38, 39, 48);
                border: #fff;
                }
                .stForm{
                border: 0;
                }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Speech Enhancement Demo</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; padding-bottom:25px;'>This demo can be used to remove background noise from your audio/video files</p>",
    unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a audio/video file", type=["mp3", "wav", "mp4", "mov", "webm"])

if uploaded_file is not None:
    g = io.BytesIO(uploaded_file.read())
    file_name = uploaded_file.name

    with open(file_name, 'wb') as out:
        out.write(g.read())
    out.close()
    ext = file_name.split(".")[-1]
    prefix_filename = add_prefix(file_name)[:-len(ext)] + "wav"
    new_filename = join(prefix_filename)
    new_filename_enhanced = join("enhanced_" + prefix_filename)

    data, sr = librosa.load(file_name, sr=16000)
    sf.write(new_filename, data, sr)

    with open(new_filename, 'rb') as f:
        st.write("Original File")
        st.audio(f)

    with st.spinner('Processing your audio'):
        enhanced = model_se.separate(data[np.newaxis, ...])[0][0]
        print(enhanced.shape)
        sf.write(new_filename_enhanced, enhanced, sr)
        st.write("Enhanced File")
        with open(new_filename_enhanced, 'rb') as f:
            st.audio(f)

    os.remove(new_filename)
    os.remove(new_filename_enhanced)
