import os
import gradio as gr
from groq import Groq
from pypdf import PdfReader

# ---------------- API KEY ----------------

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not set! Set it in environment or .env file.")

client = Groq(api_key=API_KEY)

# ---------------- PDF Extraction ----------------
def extract_text(pdf_path, max_chars=4000):
    """Extract and truncate text from PDF to avoid token overload"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            part = page.extract_text()
            if part:
                text += part + "\n"
        # Clean empty lines
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        return text[:max_chars]
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""

# ---------------- AI Logic ----------------
def analyze_pdf(file):
    if file is None:
        return "<p style='color:red;'>Please upload a PDF.</p>"

    file_path = file.name if hasattr(file, "name") else file
    if not os.path.exists(file_path):
        return "<p style='color:red;'>Error: file not found. Re-upload.</p>"

    text = extract_text(file_path)
    if not text:
        return "<p>No text extracted. PDF might be scanned (image-only).</p>"

    prompt = f"""
You are <b>Research Buddy</b>, a professional AI assistant. 
Analyze the following research paper and return the output <b>as HTML</b> for clear readability.
<b>Important:</b> 
- Each section must be unique. <b>Avoid repeating words, sentences, or phrases across sections.</b>
- Use <h2> for headings, <b> for key terms, <ul>/<li> for bullets, and numbered steps for methodology.
- Explain formulas in simple words.
- Make the output professional, concise, and readable.
Sections to include:
<h2>Summary</h2>
<h2>Keywords</h2>
<h2>Methodology</h2>
<h2>Future Implications</h2>
<h2>Improvements</h2>
<h2>Research Gaps</h2>
<h2>Limitations</h2>
Paper Text:
{text}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"<p style='color:red;'>Groq API Error:<br>{e}</p>"

# ---------------- Lavender CSS ----------------
lavender_css = """
body{
  background: linear-gradient(180deg,#f8f5ff,#e8dfff);
  font-family: 'Inter', sans-serif;
}
h1,h2{
  color:#5a2bc8;
}
.card{
  background:white;
  border:1px solid #d4bbff;
  border-radius:20px;
  padding:25px;
  box-shadow:0 8px 25px rgba(160,70,240,0.15);
  margin-bottom:25px;
}
#analyze_btn{
  background: linear-gradient(90deg,#9f7aea,#7c3aed);
  color:white;
  font-weight:bold;
  border-radius:15px;
  padding:15px;
  border:none;
  font-size:16px;
  cursor:pointer;
}
#output_box{
  font-family: 'Inter', sans-serif;
  line-height:1.6;
}
"""

# ---------------- UI ----------------
with gr.Blocks() as demo:  # removed theme= argument
    gr.HTML(f"<style>{lavender_css}</style>")  # inject CSS

    gr.HTML("""
<div style='text-align:center; margin-bottom:20px;'>
    <h1>📘 Research Buddy</h1>
    <p style='color:#555;'>AI Assistant for Scientific & Technical Research Papers</p>
</div>
""")

    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML("<div class='card'>")
            pdf_in = gr.File(label="Upload PDF", file_types=[".pdf"])
            analyze_btn = gr.Button("Analyze Paper", elem_id="analyze_btn")
            gr.HTML("</div>")

        with gr.Column(scale=5):
            gr.HTML("<div class='card'>")
            output = gr.HTML(
                "<p>Upload a PDF and click <b>Analyze Paper</b> to begin.</p>",
                elem_id="output_box"
            )
            gr.HTML("</div>")

    analyze_btn.click(analyze_pdf, inputs=pdf_in, outputs=output)

if __name__ == "__main__":
    demo.launch()
