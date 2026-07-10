"""Editable candidate profile + scoring rules — the one place to tune preferences.

Everything subjective (salary floors, remote/seniority rules, skill priorities, the
defense-contractor list, resume-routing keywords) lives here so it can be edited without
touching scoring logic. No personal data is hardcoded into business logic elsewhere; the
scorer and resume recommender read from this module.
"""

from __future__ import annotations

# --- who I am / what I want -------------------------------------------------

PROFILE = {
    "clearance": "TS/SCI",  # highest clearance held (see scoring._CLEAR_RANK)
    "has_polygraph": False,
    "min_salary": 120000,  # warn below this
    "floor_salary": 100000,  # likely-skip below this (unless strategic)
    "strong_salary": 140000,
    "very_strong_salary": 170000,
    "remote_pref": "remote",  # remote > hybrid > onsite
    "seniority": "senior",
    "skills": [
        "python",
        "bash",
        "scripting",
        "typescript",
        "fastapi",
        "sql",
        "docker",
        "linux",
        "network",
        "networking",
        "cisco",
        "vpn",
        "firewall",
        "vlan",
        "routing",
        "switching",
        "systems engineer",
        "systems engineering",
        "system engineer",
        "infrastructure",
        "devsecops",
        "secops",
        "security",
        "stig",
        "rmf",
        "ia",
        "multi-enclave",
        "enclave",
        "classified",
        "puppet",
        "katello",
        "ansible",
        "vmware",
        "ai",
        "llm",
        "rag",
        "machine learning",
        "automation",
        "agents",
        "backend",
        "api",
        "cloud",
        "aws",
        "ci/cd",
        "kubernetes",
        "security+",
        "iat",
    ],
    "passions": [
        "ai",
        "llm",
        "rag",
        "machine learning",
        "automation",
        "agents",
        "software",
        "backend",
        "building",
        "platform",
        "data pipeline",
        "genai",
        "api integration",
        "devsecops",
        "scripting",
    ],
}

# --- salary tiers (annual USD) ----------------------------------------------
# Each: (floor, score, label). First tier whose floor the comp clears wins.
SALARY_TIERS = [
    (170000, 100, "very strong (≥$170k)"),
    (140000, 90, "strong (≥$140k)"),
    (120000, 78, "acceptable (≥$120k)"),
    (100000, 55, "caution ($100–120k)"),
    (0, 25, "low (<$100k) — likely skip unless strategic"),
]
SALARY_UNKNOWN_SCORE = 55  # unknown comp → neutral, and drives an "ask recruiter" rec

# --- remote / seniority rules -----------------------------------------------
REMOTE_SCORES = {"remote": 100, "hybrid": 72, "onsite": 30, "field": 12, "unknown": 55}

# Title signals.
SENIOR_TERMS = r"\b(senior|sr\.?|staff|principal|lead|architect|manager|head of|chief)\b"
JUNIOR_TERMS = (
    r"\b(junior|jr\.?|entry[\s-]?level|associate|intern|apprentice|trainee|\bi{1,2}\b|tier\s*[12i]+)\b"
)
# "Beneath me" role markers (independent of seniority word).
BENEATH_TERMS = [
    "field technician",
    "field service",
    "field engineer",
    "help desk",
    "helpdesk",
    "service desk",
    "desktop support",
    "deskside",
    "tier 1",
    "tier i",
    "break/fix",
    "break fix",
    "installer",
    "hardware install",
    "cable",
    "bench technician",
]

# --- cleared / defense preset ----------------------------------------------
CLEARED_PRESET = {
    "boost": [
        "ts/sci",
        "top secret",
        "polygraph",
        "remote",
        "hybrid",
        "systems engineer",
        "network engineer",
        "network engineering",
        "vpn",
        "firewall",
        "devsecops",
        "secops",
        "python",
        "bash",
        "scripting",
        "stig",
        "rmf",
        "multi-enclave",
        "classified",
        "ai",
        "software engineer",
        "backend",
        "automation",
        "security+",
        "iat",
    ],
    "penalize": BENEATH_TERMS,
    "travel_ok": False,
    "onsite_ok": False,
}
PRESETS = {"default": {"penalize": [], "travel_ok": True, "onsite_ok": True}, "cleared": CLEARED_PRESET}

# --- resume routing ---------------------------------------------------------
# Big primes/integrators → Cleared Defense AI.
DEFENSE_CONTRACTORS = [
    "booz allen",
    "leidos",
    "saic",
    "caci",
    "peraton",
    "lockheed",
    "northrop",
    "raytheon",
    "general dynamics",
    "gdit",
    "mantech",
    "parsons",
    "jacobs",
    "kbr",
    "ssci",
    "mitre",
    "aerospace corporation",
    "ball aerospace",
    "l3harris",
    "bae systems",
]

RESUME_RULES = {
    "Cleared Defense AI": r"\b(ts/?sci|secret|clearance|defense|dod|mission|rmf|stig|systems? engineer|network|infrastructure|sysadmin|enclave|cyber)\b",
    "Applied AI": r"\b(applied ai|ai engineer|machine learning|\bml\b|\bllm\b|rag|genai|evals?|data scien|model|prompt)\b",
    "AI Automation": r"\b(automation|automate|\bbots?\b|agents?|workflow|webhook|job queue|integration|rpa|scripting pipeline)\b",
    "AI Software": r"\b(software engineer|backend|full[\s-]?stack|\bapi\b|platform|microservices?|services|distributed)\b",
    # ATS Plain is the fallback.
}
RESUME_ORDER = ["Cleared Defense AI", "Applied AI", "AI Automation", "AI Software", "ATS Plain"]
