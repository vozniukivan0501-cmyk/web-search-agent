"""Hugging Face Spaces and cloud entrypoint."""
import os
import uvicorn
from app.server import app

# Mount Gradio if available (so Hugging Face Gradio SDK recognizes the app)
try:
    import gradio as gr
    demo = gr.Blocks()
    app = gr.mount_gradio_app(app, demo, path="/_gradio")
except Exception:
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
