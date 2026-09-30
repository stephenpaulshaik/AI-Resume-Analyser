import os
import re

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from pypdf import PdfReader
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --------------------------------------------------
# SKILLS DATABASE
# --------------------------------------------------

SKILLS = [
    "python",
    "java",
    "c",
    "c++",
    "javascript",
    "typescript",
    "html",
    "css",
    "react",
    "angular",
    "vue",
    "node.js",
    "express",
    "flask",
    "django",
    "fastapi",
    "spring boot",

    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "oracle",
    "firebase",
    "database",
    "dbms",

    "rest api",
    "api",
    "api development",

    "git",
    "github",
    "docker",
    "kubernetes",

    "aws",
    "azure",
    "google cloud",

    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",

    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",

    "data structures",
    "algorithms",
    "oops",
    "object oriented programming",

    "operating systems",
    "computer networks",

    "power bi",
    "tableau",

    "agile",

    "communication",
    "leadership",
    "problem solving"
]


# --------------------------------------------------
# BASIC FUNCTIONS
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------
# FILE TEXT EXTRACTION
# --------------------------------------------------

def extract_pdf_text(filepath):
    text = ""

    reader = PdfReader(filepath)

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_docx_text(filepath):
    document = Document(filepath)

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    return "\n".join(paragraphs)


def extract_resume_text(filepath):
    extension = filepath.rsplit(".", 1)[1].lower()

    if extension == "pdf":
        return extract_pdf_text(filepath)

    if extension == "docx":
        return extract_docx_text(filepath)

    return ""


# --------------------------------------------------
# SKILL EXTRACTION
# --------------------------------------------------

def contains_skill(text, skill):
    text = clean_text(text)
    skill = clean_text(skill)

    escaped_skill = re.escape(skill)

    pattern = r"(?<![a-z0-9])" + escaped_skill + r"(?![a-z0-9])"

    return re.search(pattern, text) is not None


def extract_skills(text):
    found_skills = []

    for skill in SKILLS:
        if contains_skill(text, skill):
            found_skills.append(skill)

    return found_skills


# --------------------------------------------------
# JOB DESCRIPTION SIMILARITY
# --------------------------------------------------

def calculate_similarity(resume_text, job_description):

    resume_text = clean_text(resume_text)
    job_description = clean_text(job_description)

    if not resume_text or not job_description:
        return 0

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        vectors = vectorizer.fit_transform(
            [resume_text, job_description]
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return round(similarity * 100, 2)

    except Exception:
        return 0


# --------------------------------------------------
# SKILL MATCH SCORE
# --------------------------------------------------

def calculate_skill_score(matched_skills, job_skills):

    if not job_skills:
        return 0

    score = (len(matched_skills) / len(job_skills)) * 100

    return round(min(score, 100), 2)


# --------------------------------------------------
# RESUME QUALITY SCORE
# --------------------------------------------------

def calculate_resume_quality(resume_text):

    text = clean_text(resume_text)

    score = 0

    # Contact information
    if re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                resume_text,
                re.IGNORECASE):
        score += 10

    if re.search(r"\b\d{10}\b", resume_text):
        score += 10

    # Sections
    if "education" in text:
        score += 8

    if "experience" in text or "internship" in text:
        score += 8

    if "project" in text or "projects" in text:
        score += 8

    if "skills" in text:
        score += 8

    if "certification" in text or "certifications" in text:
        score += 8

    # Numbers / achievements
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", resume_text)

    if len(numbers) >= 3:
        score += 10

    # Action verbs
    action_verbs = [
        "developed",
        "created",
        "designed",
        "implemented",
        "built",
        "managed",
        "developed",
        "optimized",
        "analyzed",
        "tested",
        "deployed"
    ]

    action_count = 0

    for verb in action_verbs:
        if verb in text:
            action_count += 1

    if action_count >= 3:
        score += 12

    return min(score, 100)


# --------------------------------------------------
# LOCAL AI REVIEW ENGINE
# --------------------------------------------------

def generate_local_ai_review(
    resume_text,
    job_description,
    matched_skills,
    missing_skills,
    skill_score,
    similarity_score,
    resume_quality,
    overall_score
):

    resume_lower = clean_text(resume_text)

    strengths = []
    weaknesses = []
    suggestions = []

    # ----------------------------------------------
    # STRENGTHS
    # ----------------------------------------------

    if skill_score >= 80:
        strengths.append(
            "Strong skill match with the target job description."
        )

    elif skill_score >= 60:
        strengths.append(
            "Good number of relevant skills are present."
        )

    else:
        strengths.append(
            "The resume contains some skills relevant to the target role."
        )

    if similarity_score >= 70:
        strengths.append(
            "Resume content has strong alignment with the job description."
        )

    elif similarity_score >= 40:
        strengths.append(
            "Several terms and concepts overlap with the target job."
        )

    if "project" in resume_lower:
        strengths.append(
            "Projects are included to demonstrate practical experience."
        )

    if "internship" in resume_lower:
        strengths.append(
            "Internship experience is included."
        )

    if "github" in resume_lower:
        strengths.append(
            "GitHub information is included."
        )

    if "linkedin" in resume_lower:
        strengths.append(
            "LinkedIn information is included."
        )

    # ----------------------------------------------
    # WEAKNESSES
    # ----------------------------------------------

    if missing_skills:
        weaknesses.append(
            "Some skills requested in the job description "
            "are missing from the resume."
        )

    if similarity_score < 40:
        weaknesses.append(
            "The resume has relatively low textual alignment "
            "with the target job description."
        )

    if resume_quality < 60:
        weaknesses.append(
            "The resume could be improved in structure, completeness "
            "and measurable achievements."
        )

    if "experience" not in resume_lower and "internship" not in resume_lower:
        weaknesses.append(
            "Professional or internship experience is not clearly shown."
        )

    if "certification" not in resume_lower:
        weaknesses.append(
            "Certifications are not clearly mentioned."
        )

    # ----------------------------------------------
    # SUGGESTIONS
    # ----------------------------------------------

    if missing_skills:

        skills_text = ", ".join(missing_skills[:5])

        suggestions.append(
            "If you genuinely have these skills, consider adding them: "
            + skills_text
        )

    if similarity_score < 50:
        suggestions.append(
            "Tailor the resume to the job description using "
            "relevant keywords that accurately describe your experience."
        )

    if "project" not in resume_lower:
        suggestions.append(
            "Add 2–3 relevant projects with technologies used "
            "and measurable results."
        )

    if "internship" not in resume_lower and "experience" not in resume_lower:
        suggestions.append(
            "Add internship, training, freelance or practical experience "
            "if applicable."
        )

    if "github" not in resume_lower:
        suggestions.append(
            "Add your GitHub profile and relevant project repositories."
        )

    if "certification" not in resume_lower:
        suggestions.append(
            "Add relevant certifications or completed courses."
        )

    # ----------------------------------------------
    # DEFAULT VALUES
    # ----------------------------------------------

    if not strengths:
        strengths.append(
            "The resume contains information relevant to the target role."
        )

    if not weaknesses:
        weaknesses.append(
            "No major issues were detected by the local analysis engine."
        )

    if not suggestions:
        suggestions.append(
            "Continue tailoring the resume for each job description."
        )

    # ----------------------------------------------
    # EXECUTIVE SUMMARY
    # ----------------------------------------------

    if overall_score >= 80:

        summary = (
            f"The resume shows a strong overall match for the target role, "
            f"with an analysis score of {overall_score}/100. "
            f"Focus on refining keywords and presenting achievements clearly."
        )

    elif overall_score >= 60:

        summary = (
            f"The resume shows a moderate match for the target role, "
            f"with an analysis score of {overall_score}/100. "
            f"Improving missing skills, job-specific keywords and measurable "
            f"achievements could strengthen the resume."
        )

    else:

        summary = (
            f"The resume currently has an analysis score of "
            f"{overall_score}/100 for this job description. "
            f"More alignment with the required skills and stronger "
            f"job-specific content could improve the match."
        )

    return {
        "summary": summary,
        "strengths": strengths[:5],
        "weaknesses": weaknesses[:5],
        "suggestions": suggestions[:6]
    }


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# ANALYZE RESUME
# --------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    resume_file = request.files.get("resume")

    job_description = request.form.get(
        "job_description",
        ""
    ).strip()

    # ----------------------------------------------
    # VALIDATION
    # ----------------------------------------------

    if not resume_file:

        return render_template(
            "result.html",
            error="Please upload your resume."
        )

    if resume_file.filename == "":

        return render_template(
            "result.html",
            error="Please select a resume file."
        )

    if not allowed_file(resume_file.filename):

        return render_template(
            "result.html",
            error="Only PDF and DOCX files are supported."
        )

    if not job_description:

        return render_template(
            "result.html",
            error="Please enter the job description."
        )

    # ----------------------------------------------
    # SAVE FILE
    # ----------------------------------------------

    filename = secure_filename(resume_file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        resume_file.save(filepath)

        # ------------------------------------------
        # EXTRACT RESUME TEXT
        # ------------------------------------------

        resume_text = extract_resume_text(filepath)

        if not resume_text.strip():

            return render_template(
                "result.html",
                error="Could not extract text from the resume."
            )

        # ------------------------------------------
        # SKILLS
        # ------------------------------------------

        resume_skills = extract_skills(resume_text)

        job_skills = extract_skills(job_description)

        matched_skills = [
            skill
            for skill in job_skills
            if skill in resume_skills
        ]

        missing_skills = [
            skill
            for skill in job_skills
            if skill not in resume_skills
        ]

        # ------------------------------------------
        # SCORES
        # ------------------------------------------

        skill_score = calculate_skill_score(
            matched_skills,
            job_skills
        )

        similarity_score = calculate_similarity(
            resume_text,
            job_description
        )

        resume_quality = calculate_resume_quality(
            resume_text
        )

        # ------------------------------------------
        # OVERALL SCORE
        # ------------------------------------------

        overall_score = round(
            skill_score * 0.50
            + similarity_score * 0.30
            + resume_quality * 0.20,
            2
        )

        # ------------------------------------------
        # LOCAL AI REVIEW
        # ------------------------------------------

        ai_review = generate_local_ai_review(
            resume_text=resume_text,
            job_description=job_description,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            skill_score=skill_score,
            similarity_score=similarity_score,
            resume_quality=resume_quality,
            overall_score=overall_score
        )

        # ------------------------------------------
        # RESULT PAGE
        # ------------------------------------------

        return render_template(
            "result.html",

            overall_score=overall_score,

            skill_score=skill_score,

            similarity_score=similarity_score,

            resume_quality=resume_quality,

            matched_skills=matched_skills,

            missing_skills=missing_skills,

            ai_review=ai_review,

            resume_filename=filename
        )

    except Exception as error:

        print("ANALYSIS ERROR:", error)

        return render_template(
            "result.html",
            error="Something went wrong while analyzing the resume."
        )

    finally:

        if os.path.exists(filepath):

            try:
                os.remove(filepath)

            except Exception:
                pass


# --------------------------------------------------
# FILE SIZE ERROR
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "result.html",
        error="File is too large. Maximum size is 5 MB."
    ), 413


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )