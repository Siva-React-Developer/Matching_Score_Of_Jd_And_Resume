from flask import Flask, render_template,request,session,redirect,url_for
import os
import fitz
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key="siva"
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def summarize_text(text):
    genai.configure(api_key="AIzaSyDjdsEO_lCAgjfcOcfRKX5qZmJJnHaN6yg")
    model= genai.GenerativeModel("gemini-1.5-flash")
    temperature=0.0
    response = model.generate_content([f"please extract the programming skills and its frameworks from this text,don't give any info or about for programming skills and frameworks and make a list:{text}"])
    return response.text
    
def summarize_Jd_text(text):
    genai.configure(api_key="AIzaSyDjdsEO_lCAgjfcOcfRKX5qZmJJnHaN6yg")
    model= genai.GenerativeModel("gemini-1.5-flash")
    temperature=0.0
    response = model.generate_content([f"please extract the programming skills and its frameworks from this text,don't give any info or about for programming skills and frameworks and make a list:{text}"])
    return response.text

def get_embedding(text, model="models/text-embedding-004"):
        response = genai.embed_content(model=model, content=text)
        embedding = response['embedding']
        return np.array(embedding, dtype="float32")

def calculate_similarity(Resume_skills, Jd_skills, model="models/text-embedding-004"):
        embedding1 = get_embedding(Resume_skills, model=model)
        embedding2 = get_embedding(Jd_skills, model=model)
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity

@app.route('/', methods=['POST','GET'])
def home():
    if request.method == 'POST':
        if "resume" not in request.files or "txtarea" not in request.form:
            return redirect(url_for('error'))
        
        jd_text=request.form['txtarea']
        file = request.files["resume"]

        if file.filename == "" or jd_text == "":
            return redirect(url_for('error'))

        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            extracted_text = extract_text_from_pdf(file_path)
            resume_text=extracted_text
            Resume_skills = summarize_text(resume_text)
            Jd_skills=summarize_Jd_text(jd_text)
            ats_score = calculate_similarity(Resume_skills, Jd_skills)

            num = np.float32(ats_score*100)  # NumPy float32
            converted_num = round(float(num),2)
            session['score']=converted_num

            return redirect(url_for('score'))
        
    return render_template('index.html')


@app.route('/score')
def score():
    if 'score' in session:
        match_score=session['score']
        return render_template('score.html',match_score=match_score)
    else:
        return redirect(url_for('home'))

@app.route('/error')
def error():
     return render_template("error_page.html")

if __name__ == '__main__':
    app.run(debug=True)

# jd_text=""!We are seeking a talented and motivated Front-End Developer to join our team. The ideal candidate will be responsible for creating responsive, user-friendly web interfaces using HTML, CSS, and JavaScript, while collaborating with designers and backend developers to deliver seamless user experiences. Proficiency in modern frameworks like React, Vue.js, or Angular,java,bootstrap,sql along with a strong understanding of responsive design and version control (Git), is required. If you are passionate about building visually appealing, high-performance web applications and staying updated with the latest front-end technologies, we’d love to hear from you