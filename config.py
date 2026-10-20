"""Central configuration for the resume screening pipeline."""

from typing import Final

MODEL_NAME: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_SCORING_WEIGHTS: Final[dict[str, float]] = {
    "semantic_similarity": 0.50,
    "skill_match": 0.25,
    "experience_match": 0.15,
    "education_match": 0.10,
}

RECOMMENDATION_THRESHOLDS: Final[dict[str, float]] = {
    "shortlist": 80.0,
    "review": 60.0,
}

# Each canonical name maps to phrases that may appear in source documents.
# Adding a skill or alias here makes it available to the extractor.
SKILL_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "Python": ("python",),
    "Java": ("java",),
    "C": ("c",),
    "C++": ("c++", "cpp"),
    "JavaScript": ("javascript", "js"),
    "TypeScript": ("typescript", "ts"),
    "React": ("react", "react.js", "reactjs"),
    "Node.js": ("node.js", "nodejs", "node js"),
    "FastAPI": ("fastapi",),
    "Flask": ("flask",),
    "Django": ("django",),
    "SQL": ("sql",),
    "MySQL": ("mysql",),
    "PostgreSQL": ("postgresql", "postgres"),
    "MongoDB": ("mongodb", "mongo db"),
    "AWS": ("aws", "amazon web services"),
    "Azure": ("azure", "microsoft azure"),
    "GCP": ("gcp", "google cloud platform"),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes", "k8s"),
    "Git": ("git",),
    "Machine Learning": ("machine learning", "ml"),
    "Deep Learning": ("deep learning", "dl"),
    "NLP": ("nlp", "natural language processing"),
    "TensorFlow": ("tensorflow",),
    "PyTorch": ("pytorch", "torch"),
    "Scikit-learn": ("scikit-learn", "scikit learn", "sklearn"),
    "Pandas": ("pandas",),
    "NumPy": ("numpy", "num py"),
    "Power BI": ("power bi", "powerbi"),
    "Tableau": ("tableau",),
    "Excel": ("excel", "microsoft excel"),
    "HTML": ("html",),
    "CSS": ("css",),
    ".NET": (".net", "dotnet"),
}

# Retained as a separate public name so callers can inspect or extend the
# canonical vocabulary without needing to derive it from the alias mapping.
SKILL_DICTIONARY: Final[tuple[str, ...]] = tuple(SKILL_ALIASES)
