import os
from dotenv import load_dotenv
import gradio as gr
from engine import OilGasGeminiBot

load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def chat_handler(language, api_key, query, files):
    effective_key = api_key.strip() or API_KEY
    if not effective_key:
        return (
            "Error técnico: No se encontró la clave de Gemini. "
            "Define GOOGLE_API_KEY o GEMINI_API_KEY en el entorno o ingresa la API key aquí."
        )

    try:
        bot_instance = OilGasGeminiBot(api_key=effective_key)
        bot_instance.language = language
        if files:
            bot_instance.ingest_files([f.name for f in files])

        return bot_instance.ask(query)
    except Exception as e:
        return f"Error técnico: {str(e)}"

# UI con estilo Azul Rey y Naranja
with gr.Blocks() as demo:
    gr.HTML("<h1 style='color: #002366; text-align: center;'>Capacitador Oil & Gas (Gemini)</h1>")
    
    with gr.Row():
        with gr.Column(scale=1):
            lang_opt = gr.Radio(["Español", "Inglés"], label="Idioma", value="Español")
            api_key_input = gr.Textbox(
                label="API Key de Gemini (opcional, se usa si no está en secrets)",
                type="password",
                lines=1,
                placeholder="Ingresa tu API key aquí o configura GOOGLE_API_KEY/GEMINI_API_KEY",
            )
            docs = gr.File(label="Subir PDF/TXT", file_count="multiple")
            
        with gr.Column(scale=2):
            msg = gr.Textbox(label="Tu pregunta técnica")
            ans = gr.Textbox(label="Respuesta del Ingeniero (Max 300 caracteres)")
            btn = gr.Button("Enviar Consulta", variant="primary")

    btn.click(chat_handler, [lang_opt, api_key_input, msg, docs], ans)

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Default(primary_hue="orange", secondary_hue="blue"),
        share=True,
        inbrowser=True,
    )