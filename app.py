"""Streamlit interface for the modern NLP resume screening system."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import SKILL_ALIASES
from utils.nlp_analyzer import (
    calculate_overall_fit,
    compute_tfidf_similarity,
    estimate_years_of_experience,
    extract_candidate_info,
    extract_education,
)
from utils.pdf_parser import ExtractionResult, extract_text_from_file
from utils.skill_extractor import extract_and_compare_skills
from utils.text_preprocessor import normalize_text


# Predefined job description templates for quick 1-click testing
SAMPLE_JOB_DESCRIPTIONS = {
    "AI / NLP Engineer": (
        "We are looking for a Senior NLP & Machine Learning Engineer with 3+ years of experience. "
        "Key skills: Python, PyTorch, TensorFlow, NLP, Scikit-learn, Pandas, NumPy, Docker, Git. "
        "Experience with FastApi or Flask for model serving, cloud deployment on AWS or GCP, "
        "and strong background in text preprocessing, TF-IDF, Transformer models, and deep learning."
    ),
    "Full-Stack Web Developer": (
        "Seeking a passionate Full-Stack Developer proficient in Python, JavaScript, TypeScript, React, "
        "and Node.js. Hands-on experience with FastAPI or Django, SQL, PostgreSQL, MongoDB, Docker, "
        "and Git version control. Minimum 2 years of building scalable web applications with REST APIs."
    ),
    "Data Scientist / Analyst": (
        "Looking for a Data Scientist skilled in Python, SQL, Machine Learning, Scikit-learn, "
        "Pandas, NumPy, Tableau, Power BI, and Excel. Must have strong statistical modeling, data visualization, "
        "and data extraction capabilities. 2+ years of relevant industry experience."
    ),
}

# Pre-packaged sample resumes for immediate preview/evaluation
SAMPLE_RESUMES = {
    "Alex_Rivers_NLP_Engineer.txt": (
        "Alex Rivers\nEmail: alex.rivers@example.com | Phone: +1-555-019-2831\n"
        "LinkedIn: linkedin.com/in/alex-rivers-nlp | GitHub: github.com/alexrivers\n\n"
        "Professional Summary:\n"
        "Passionate Machine Learning Engineer with 4 years of experience specializing in NLP, "
        "Natural Language Processing, PyTorch, TensorFlow, Scikit-learn, and Python.\n\n"
        "Education:\n"
        "Master of Science in Computer Science (M.S.), 2022\n\n"
        "Technical Skills:\n"
        "Python, NLP, PyTorch, TensorFlow, Scikit-learn, Pandas, NumPy, Docker, Git, FastAPI, AWS.\n\n"
        "Experience:\n"
        "NLP Engineer at TextVision AI (2022 - Present):\n"
        "- Developed text classification and semantic matching algorithms using TF-IDF and transformer architectures.\n"
        "- Containerized microservices with Docker and deployed to AWS cloud pipeline.\n"
        "- Optimized feature extraction routines with Pandas and Scikit-learn."
    ),
    "Jordan_Smith_FullStack.txt": (
        "Jordan Smith\nEmail: jordan.smith@techdomain.org | Phone: +1-555-482-9901\n"
        "LinkedIn: linkedin.com/in/jordansmith-dev | GitHub: github.com/jsmith-codes\n\n"
        "Professional Summary:\n"
        "Full-Stack Developer with 3 years experience building responsive web platforms with React and Python.\n\n"
        "Education:\n"
        "Bachelor of Technology in Computer Engineering (B.Tech), 2021\n\n"
        "Technical Skills:\n"
        "JavaScript, TypeScript, React, Node.js, Python, Django, PostgreSQL, SQL, Docker, Git, HTML, CSS.\n\n"
        "Experience:\n"
        "Software Engineer at CloudScale Labs (2021 - Present):\n"
        "- Built interactive dashboard frontends using React and modern CSS.\n"
        "- Designed backend REST APIs with Python Django and PostgreSQL databases."
    ),
    "Taylor_Morgan_Junior_Analyst.txt": (
        "Taylor Morgan\nEmail: tmorgan.analytics@gmail.com | Phone: +1-555-731-8842\n\n"
        "Professional Summary:\n"
        "Junior Data Analyst with 1 year of experience in statistical analysis, Excel modeling, and data visualization.\n\n"
        "Education:\n"
        "Bachelor of Science (B.Sc) in Statistics, 2023\n\n"
        "Technical Skills:\n"
        "Python, SQL, Excel, Tableau, Pandas, Power BI, HTML.\n\n"
        "Experience:\n"
        "Data Analyst Intern at InsightCorp (2023 - 2024):\n"
        "- Created business dashboards in Tableau and Power BI.\n"
        "- Cleaned relational datasets using SQL queries and basic Python scripts."
    ),
}


class MockUploadedFile:
    """Mock Streamlit UploadedFile to allow instant testing with built-in samples."""

    def __init__(self, name: str, text_content: str):
        self.name = name
        self._content = text_content.encode("utf-8")
        self._io = io.BytesIO(self._content)

    def read(self, *args: Any, **kwargs: Any) -> bytes:
        return self._io.read(*args, **kwargs)


def _load_custom_css() -> None:
    """Inject modern styling into the Streamlit session."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def _skill_match_score(comparison: dict[str, list[str]]) -> float:
    """Return the percentage of required skills found in a resume."""
    required = comparison.get("required_skills", [])
    if not required:
        return 0.0
    matched = comparison.get("matched_required_skills", [])
    return round(len(matched) / len(required) * 100, 1)


def _screen_resume(
    job_description: str,
    uploaded_file: Any,
    min_exp_req: float = 0.0,
) -> dict[str, Any]:
    """Extract, analyze, and screen one uploaded resume."""
    extraction: ExtractionResult = extract_text_from_file(uploaded_file)
    filename = str(getattr(uploaded_file, "name", "Unnamed file"))
    result: dict[str, Any] = {
        "filename": filename,
        "extraction": extraction,
    }

    if extraction.success:
        clean_text = extraction.text
        # 1. Skill Extraction & Alias Matching
        comparison = extract_and_compare_skills(
            job_description,
            clean_text,
            SKILL_ALIASES,
        )
        skill_score = _skill_match_score(comparison)

        # 2. NLP Semantic TF-IDF Cosine Similarity
        tfidf_score = compute_tfidf_similarity(job_description, clean_text)

        # 3. Information Extraction (Experience, Contact, Education)
        exp_years = estimate_years_of_experience(clean_text)
        candidate_info = extract_candidate_info(clean_text)
        education_degrees = extract_education(clean_text)

        # 4. Composite Fit Evaluation
        fit_eval = calculate_overall_fit(
            skill_match_pct=skill_score,
            tfidf_sim_pct=tfidf_score,
            experience_years=exp_years,
            required_min_experience=min_exp_req,
        )

        result.update({
            "comparison": comparison,
            "skill_score": skill_score,
            "tfidf_score": tfidf_score,
            "experience_years": exp_years,
            "candidate_info": candidate_info,
            "education": education_degrees,
            "score": fit_eval["composite_score"],
            "fit_eval": fit_eval,
        })

    return result


def _render_overview_charts(results: list[dict[str, Any]]) -> None:
    """Render interactive Plotly comparison charts."""
    valid_results = [r for r in results if r["extraction"].success]
    if not valid_results:
        return

    df = pd.DataFrame([
        {
            "Candidate": r["filename"].replace(".txt", "").replace(".pdf", ""),
            "Composite Score (%)": r["score"],
            "Skill Coverage (%)": r["skill_score"],
            "TF-IDF Similarity (%)": r["tfidf_score"],
            "Experience (Yrs)": r["experience_years"],
            "Status": r["fit_eval"]["status_tag"],
        }
        for r in valid_results
    ])

    tab1, tab2 = st.tabs(["📊 Candidate Comparison", "🎯 Metric Breakdown"])

    with tab1:
        fig = px.bar(
            df,
            x="Candidate",
            y="Composite Score (%)",
            color="Status",
            color_discrete_map={
                "Shortlist": "#10b981",
                "Review": "#f59e0b",
                "Reject": "#ef4444",
            },
            text="Composite Score (%)",
            title="Candidate Ranking (Overall NLP Fit Score)",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
            yaxis={"range": [0, 115]},
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Grouped comparison of Skill vs TF-IDF
        fig_multi = go.Figure()
        fig_multi.add_trace(go.Bar(
            name="Skill Match %",
            x=df["Candidate"],
            y=df["Skill Coverage (%)"],
            marker_color="#6366f1",
        ))
        fig_multi.add_trace(go.Bar(
            name="TF-IDF Semantic %",
            x=df["Candidate"],
            y=df["TF-IDF Similarity (%)"],
            marker_color="#06b6d4",
        ))
        fig_multi.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
            yaxis={"range": [0, 110]},
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
        )
        st.plotly_chart(fig_multi, use_container_width=True)


def _render_candidate_card(result: dict[str, Any], rank: int) -> None:
    """Render an individual candidate card with rich NLP metrics and skill badges."""
    filename = result["filename"]
    extraction: ExtractionResult = result["extraction"]

    if not extraction.success:
        st.error(f"❌ {filename}: {extraction.error}")
        return

    fit = result["fit_eval"]
    info = result["candidate_info"]
    comparison = result["comparison"]

    status_colors = {
        "Shortlist": ("#10b981", "rgba(16, 185, 129, 0.15)"),
        "Review": ("#f59e0b", "rgba(245, 158, 11, 0.15)"),
        "Reject": ("#ef4444", "rgba(239, 68, 68, 0.15)"),
    }
    tag_fg, tag_bg = status_colors.get(fit["status_tag"], ("#94a3b8", "rgba(148, 163, 184, 0.1)"))

    st.markdown(
        f"""
        <div class="candidate-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap;">
                <div>
                    <span style="background: rgba(99, 102, 241, 0.2); color: #818cf8; font-weight: 700; padding: 2px 8px; border-radius: 6px; font-size: 0.85rem; margin-right: 8px;">#{rank}</span>
                    <strong style="font-size: 1.25rem; color: #f8fafc;">{filename}</strong>
                </div>
                <div style="background: {tag_bg}; color: {tag_fg}; border: 1px solid {tag_fg}; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
                    {fit['status_tag'].upper()} • {fit['composite_score']}% Overall Fit
                </div>
            </div>
            <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 16px;">
                📧 <b>{info['email']}</b> &nbsp;|&nbsp; 📞 <b>{info['phone']}</b> &nbsp;|&nbsp; ⏳ <b>{result['experience_years']} yrs est. experience</b>
                {f" &nbsp;|&nbsp; 🎓 <b>{', '.join(result['education'])}</b>" if result['education'] else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Metric Columns
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Match", f"{fit['composite_score']}%")
    c2.metric("Skill Coverage", f"{result['skill_score']}%")
    c3.metric("TF-IDF Semantic", f"{result['tfidf_score']}%")
    c4.metric("Experience", f"{result['experience_years']} yrs")

    # Skills Display
    st.write("**Skill Breakdown:**")
    matched_html = "".join([f'<span class="skill-badge skill-matched">✓ {s}</span>' for s in comparison["matched_required_skills"]]) or "<span style='color:#64748b;'>None</span>"
    missing_html = "".join([f'<span class="skill-badge skill-missing">✗ {s}</span>' for s in comparison["missing_required_skills"]]) or "<span style='color:#64748b;'>None</span>"
    extra_html = "".join([f'<span class="skill-badge skill-extra">+ {s}</span>' for s in comparison["additional_candidate_skills"][:10]]) or "<span style='color:#64748b;'>None</span>"

    st.markdown(f"**Matched Required Skills:**<br>{matched_html}", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-top: 8px;'><b>Missing Required Skills:</b><br>{missing_html}</div>", unsafe_allow_html=True)
    if comparison["additional_candidate_skills"]:
        st.markdown(f"<div style='margin-top: 8px;'><b>Additional Candidate Skills:</b><br>{extra_html}</div>", unsafe_allow_html=True)

    with st.expander("📄 View Extracted Text & Details"):
        st.text_area("Extracted Resume Content", extraction.text, height=180, key=f"text_{filename}_{rank}")
    st.divider()


def main() -> None:
    """Run the interactive Streamlit Resume Screening application."""
    st.set_page_config(
        page_title="AI Resume Screening & NLP Matcher",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _load_custom_css()

    # Hero Banner
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">⚡ AI Resume Screening & Talent Matcher</div>
            <div class="hero-subtitle">
                Evaluate candidates using multi-factor NLP algorithms: <b>TF-IDF N-gram Cosine Similarity</b>, 
                <b>Alias-aware Skill Extraction</b>, and <b>Automated Experience Profiling</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar settings & sample loader
    with st.sidebar:
        st.header("⚙️ Screening Configuration")
        min_experience_needed = st.slider(
            "Minimum Required Experience (Years)",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Used to calibrate the candidate experience alignment score.",
        )

        st.markdown("---")
        st.subheader("💡 1-Click NLP Demos")
        st.write("Pick a template to auto-populate job description:")
        selected_template = st.selectbox("Job Role Templates", list(SAMPLE_JOB_DESCRIPTIONS.keys()))

        use_sample_resumes = st.checkbox(
            "Include 3 Pre-loaded Sample Resumes",
            value=False,
            help="Test the application immediately without uploading external files.",
        )

        st.markdown("---")
        st.markdown(
            """
            **NLP Pipeline Components:**
            - 📑 **Text Extraction**: PyMuPDF & UTF Decoder
            - 🔍 **Alias Matcher**: Contextual Regex
            - 🧠 **Semantic NLP**: TF-IDF (1,2-grams)
            - 📊 **Visual Analytics**: Plotly Interactive
            """
        )

    # Main input workspace
    col_jd, col_upload = st.columns([1, 1], gap="medium")

    with col_jd:
        st.subheader("1. Job Description")
        default_jd = SAMPLE_JOB_DESCRIPTIONS[selected_template] if st.button("Load Selected Template") else ""
        job_description = st.text_area(
            "Paste Job Description or Requirements",
            value=default_jd,
            height=240,
            placeholder="e.g. Seeking a Senior Machine Learning Engineer with Python, PyTorch, Docker, NLP...",
            help="The system extracts required technical competencies and measures semantic content overlap.",
        )

    with col_upload:
        st.subheader("2. Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Upload Resumes (.pdf or .txt)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Select one or more candidate resumes to compare against the job requirements.",
        )
        if use_sample_resumes:
            st.info(f"ℹ️ {len(SAMPLE_RESUMES)} sample resumes active (Alex Rivers, Jordan Smith, Taylor Morgan).")

    all_files_to_process = list(uploaded_files) if uploaded_files else []
    if use_sample_resumes:
        for name, text in SAMPLE_RESUMES.items():
            all_files_to_process.append(MockUploadedFile(name, text))

    # Action Trigger
    st.write("")
    screen_btn = st.button(
        "🚀 Run NLP Candidate Screening",
        type="primary",
        disabled=not job_description.strip() or not all_files_to_process,
        use_container_width=True,
    )

    if screen_btn:
        with st.spinner("Analyzing text representations and calculating semantic metrics..."):
            normalized_jd = normalize_text(job_description)
            results = [
                _screen_resume(normalized_jd, f, min_exp_req=float(min_experience_needed))
                for f in all_files_to_process
            ]

            successful = [r for r in results if r["extraction"].success]
            successful.sort(key=lambda x: x["score"], reverse=True)

            # High level metrics summary banner
            total_cand = len(successful)
            shortlisted = sum(1 for r in successful if r["fit_eval"]["status_tag"] == "Shortlist")
            review_needed = sum(1 for r in successful if r["fit_eval"]["status_tag"] == "Review")
            avg_score = round(sum(r["score"] for r in successful) / total_cand, 1) if total_cand else 0.0

            st.write("")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="stat-card"><div class="stat-label">Total Resumes</div><div class="stat-value">{total_cand}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="stat-card"><div class="stat-label">Shortlisted (>=75%)</div><div class="stat-value" style="color:#10b981;">{shortlisted}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="stat-card"><div class="stat-label">Review Needed</div><div class="stat-value" style="color:#f59e0b;">{review_needed}</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="stat-card"><div class="stat-label">Average Fit Score</div><div class="stat-value" style="color:#38bdf8;">{avg_score}%</div></div>', unsafe_allow_html=True)

            st.write("")
            # Plotly Charts
            _render_overview_charts(successful)

            # Exportable summary table
            summary_data = [
                {
                    "Rank": i + 1,
                    "Candidate": r["filename"],
                    "Status": r["fit_eval"]["status_tag"],
                    "Composite Score": f"{r['score']}%",
                    "Skill Match": f"{r['skill_score']}%",
                    "TF-IDF Semantic": f"{r['tfidf_score']}%",
                    "Est Experience": f"{r['experience_years']} yrs",
                    "Email": r["candidate_info"]["email"],
                    "Phone": r["candidate_info"]["phone"],
                }
                for i, r in enumerate(successful)
            ]
            summary_df = pd.DataFrame(summary_data)

            # Download CSV Button
            csv_data = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Screening Report (CSV)",
                data=csv_data,
                file_name="resume_screening_report.csv",
                mime="text/csv",
            )

            st.write("")
            st.subheader("📋 Detailed Candidate Profiles")
            for idx, r in enumerate(successful, start=1):
                _render_candidate_card(r, rank=idx)

            # Failed files
            failed = [r for r in results if not r["extraction"].success]
            if failed:
                st.subheader("⚠️ Unprocessable Documents")
                for r in failed:
                    st.error(f"**{r['filename']}**: {r['extraction'].error}")


if __name__ == "__main__":
    main()
