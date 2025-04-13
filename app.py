import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_auth_token=True
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

st.title("تولید عکس با پرامپت")

prompt = st.text_input("پرامپت:", "a beautiful sunset over the mountains")

if st.button("تولید"):
    with st.spinner("در حال پردازش..."):
        pipe = load_model()
        image = pipe(prompt).images[0]
        st.image(image)
