import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from engine import OilGasGeminiBot

load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="Capacitador Oil & Gas (Gemini)", page_icon="🛢️")
st.title("Capacitador Oil & Gas (Gemini)")

if not API_KEY:
    st.warning(
        "No se encontró una clave de Gemini. Por favor define GOOGLE_API_KEY o GEMINI_API_KEY en el entorno."
    )

language = st.radio("Idioma", ["Español", "Inglés"], index=0)
uploaded_files = st.file_uploader("Subir PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)
question = st.text_area("Tu pregunta técnica", height=150)

if st.button("Enviar Consulta"):
    if not question.strip():
        st.error("Escribe una pregunta técnica antes de enviar.")
    elif not API_KEY:
        st.error(
            "No se puede procesar la consulta: falta GOOGLE_API_KEY o GEMINI_API_KEY."
        )
    else:
        bot = OilGasGeminiBot(language=language, api_key=API_KEY)
        try:
            if uploaded_files:
                with tempfile.TemporaryDirectory() as tmpdir:
                    paths = []
                    for uploaded_file in uploaded_files:
                        suffix = os.path.splitext(uploaded_file.name)[1]
                        tmp_path = os.path.join(tmpdir, uploaded_file.name)
                        with open(tmp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        paths.append(tmp_path)
                    bot.ingest_files(paths)
            response = bot.ask(question)
            st.success("Respuesta generada")
            st.write(response)
        except Exception as e:
            st.error(f"Error técnico: {e}")
