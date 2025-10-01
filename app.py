
import gradio as gr
from transformers import pipeline

# Load models
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def meeting_minutes(file):
    transcript = asr(file)["text"]
    prompt = f'''
    Summarize the following meeting transcript into:
    - Key Decisions
    - Action Items
    - Important Notes

    Transcript: {transcript}
    '''
    summary = summarizer(prompt, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]
    return transcript, summary

demo = gr.Interface(
    fn=meeting_minutes,
    inputs=gr.Audio(sources=["upload"], type="filepath"),
    outputs=[gr.Textbox(label="Transcript"), gr.Textbox(label="Meeting Minutes")],
    title="Meeting Minutes Generator",
    description="Upload a meeting recording (.wav) and get auto-generated meeting notes."
)

demo.launch()
