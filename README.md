# NLP-Based Resume Screening and Candidate Ranking System

This project is being implemented in stages. The current stage contains the
configuration, file extraction, text preprocessing, skill extraction core
modules, and a Streamlit interface for skill-coverage screening. Remaining
semantic, experience, and education scoring modules will be added in later
stages.

## Current development stage

- Configurable Sentence-BERT model and scoring settings
- PDF and TXT extraction using PyMuPDF for PDFs
- Conservative text normalization that preserves technical punctuation
- Canonical skill detection with aliases and safe phrase boundaries
- Streamlit screening of multiple resumes with per-file extraction errors
- Ranking by the percentage of required skills found in each resume

## Installation

```bash
cd resume_screening_system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Current module checks

From the project directory, run:

```bash
python -m compileall config.py utils
python -m unittest discover -s tests
```

The final application will be started with:

```bash
streamlit run app.py
```

The current interface accepts a job description and one or more PDF or TXT
resumes. It displays matched, missing, and additional configured skills for
each successfully extracted resume. Scanned PDFs are reported as requiring
OCR because OCR is not part of the current processing stage.
