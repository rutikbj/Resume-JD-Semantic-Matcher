import spacy

nlp = spacy.load('en_core_web_sm')

CUSTOM_STOPWORDS = {
    "offer", "opportunity", "outcome", "passion",
    "drive", "energetic", "dynamic", "vision", "future",
    "exciting", "enthusiastic", "fast-paced", "synergy", "values",
    "conduct", "solution","optimize","access", "accuracy",
    "access", "accuracy", "attention", "background", "base", "brief", "clearly", "communicate", "communication", "company", "complete", 
    "concept", "concisely", "contribute", "relevant", "solution", "solve", "welcome", "work", "understanding", 
    "time", "title", "support", "system", "task", "team", "technical", "send", "service", "shape", "standard", "stay", "strong", 
    "recently", "related", "renowne", "require", "present", "portfolio", "potential", "preferred", "passionate", "performance", 
    "outcome", "offer", "opportunity", "need", "meeting", "member", "mentorship", "ltd", "maintain", "line", "literature", "location", 
    "late", "lead", "learn", "letter", "leverage", "join", "key", "intersection", "interactive", "innovative", "insight", 
    "global", "good", "hand", "highlight", "impactful", "implement", "include", "inclusion", "industry", 
    "experiment", "exposure", "familiarity", "feature", "field", "finding", "flexible", "functional", "future", "gain", 
    "exciting", "enhance", "entertainment", "enthusiastic", "environment", "etc", "evaluate", 
    "discussion", "distributor", "diverse", "diversity", "document", "domain", "duration", "edge", 
    "deploy", "description", "complete", "concept", "concisely", "conduct", "control", "core", "coursework", "cover", "creator", "cross", "culture", "currently", "cut", "date"
    
}


def preprocess(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop
        and not token.is_alpha == False
        and token.lemma_ not in CUSTOM_STOPWORDS
    ]
    return ' '.join(tokens)

def match_keywords(job_processed, resume_processed):
    job_tokens = set(job_processed.split())
    resume_tokens = set(resume_processed.split())
    matched = job_tokens.intersection(resume_tokens)
    unmatched = job_tokens.difference(resume_tokens)
    return matched, unmatched