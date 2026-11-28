"""Reusable voice-intake question and state engine."""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Awaitable
from dataclasses import dataclass


@dataclass
class VoiceIntakeDependencies:
    parse_raw_data: Callable[[Any], dict]
    detect_employment_gap: Callable[[dict], dict | None]
    llm_client: Any = None
    llm_model: str | None = None
    analyze_prompt: str | None = None
    logger: Any = None


@dataclass(frozen=True)
class VoiceExtractionDependencies:
    llm_client: Any
    model: str
    system_prompt: str
    sanitize_profile: Callable[[dict], dict]
    normalize_profile_text: Callable[[Any], str]
    logger: Any


async def extract_voice_info(
    transcript: str,
    candidate_id: str | None = None,
    vapi_call_id: str | None = None,
    endpoint: str = "/api/voice/candidate-intake",
    request_id: str | None = None,
    dependencies: VoiceExtractionDependencies | None = None,
) -> dict:
    if dependencies is None:
        raise ValueError("Voice extraction dependencies are required")
    try:
        response = await dependencies.llm_client.chat.completions.create(
            model=dependencies.model,
            messages=[
                {"role": "system", "content": dependencies.system_prompt},
                {"role": "user", "content": transcript[:8000]},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            _telemetry={
                "workflow": "voice_info_extraction",
                "function_name": "_extract_voice_info",
                "endpoint": endpoint,
                "vapi_call_id": vapi_call_id,
                "request_id": request_id,
            },
        )
        raw = response.choices[0].message.content or "{}"
        return dependencies.sanitize_profile(json.loads(raw))
    except Exception as exc:
        dependencies.logger.warning("Voice extraction LLM failed: %s", exc)
        return {}


def validate_voice_current_employment(
    voice_data: dict, transcript: str, dependencies: VoiceExtractionDependencies
) -> dict:
    cleaned = dict(voice_data or {})
    if not (cleaned.get("current_role") or cleaned.get("current_company")):
        return cleaned
    evidence = dependencies.normalize_profile_text(transcript).lower()
    current_evidence = bool(re.search(r"\b(?:current(?:ly)?|present(?:ly)?|right now|working at|work at|employed at|current (?:role|job|position|company|employer))\b", evidence))
    historical_evidence = bool(re.search(r"\b(?:previously|formerly|used to|worked at|had worked at|past role|previous role)\b", evidence))
    candidate_segments = " ".join(re.findall(r"(?:candidate|user|answer)\s*:\s*([^\n]+)", evidence, flags=re.I)) or evidence
    candidate_current_evidence = bool(re.search(r"\b(?:current(?:ly)?|present(?:ly)?|right now|working at|work at|employed at)\b", candidate_segments))
    if (historical_evidence and not candidate_current_evidence) or not current_evidence:
        cleaned.pop("current_role", None)
        cleaned.pop("current_company", None)
    return cleaned


def _default_parse_raw_data(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def _default_detect_employment_gap(profile: dict) -> dict | None:
    return None

_dependencies = VoiceIntakeDependencies(_default_parse_raw_data, _default_detect_employment_gap)


def configure_voice_intake_dependencies(dependencies: VoiceIntakeDependencies) -> None:
    global _dependencies
    _dependencies = dependencies


def _parse_raw_data(value: Any) -> dict:
    return _dependencies.parse_raw_data(value)


def _detect_employment_gap(profile: dict) -> dict | None:
    return _dependencies.detect_employment_gap(profile)
def _clean_str(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _normalize_profile_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean_str(value).lower()).strip()


# High-level intake topics used as hints, not as hardcoded question wording.
VOICE_INTAKE_TOPICS = [
    "background_experience",
    "skills_technologies",
    "target_role",
    "career_preferences",
    "responsibilities_projects",
    "education_certifications",
    "availability_location",
]

# Backend-owned topic sequence.  VAPI/LLM supplies natural wording, while the
# resume state owns which unanswered topic comes next.
VOICE_INTAKE_TOPIC_QUESTIONS = {
    "background_experience": "Tell me about your background.",
    "skills_technologies": "What are your key skills?",
    "target_role": "What kind of role would you ideally like to move into next?",
}

VOICE_INTAKE_FALLBACK_NEXT_QUESTIONS = {
    "background_experience": VOICE_INTAKE_TOPIC_QUESTIONS["skills_technologies"],
    "skills_technologies": (
        "What kind of role would you ideally like to move into next?"
    ),
    "target_role": (
        "What kind of responsibilities and projects would you like to work on next?"
    ),
}

VOICE_INTAKE_TOTAL_QUESTIONS = len(VOICE_INTAKE_TOPICS)

# Setup/greeting questions that must never count as intake questions even though
# they end with "?".
_SETUP_QUESTION_PATTERNS = (
    r"^how are you",
    r"^how(?:'re| are) you doing",
    r"^how is your day going",
    r"^how'?s your day going",
    r"^what are you doing today",
    r"^what are you up to today",
    r"^are you ready",
    r"^are you there",
    r"^(?:hi|hello|hey)[,!]?",
    r"^(?:good (?:morning|afternoon|evening))[,!]?",
    r"^(?:nice to meet you|great to meet you)",
    r"^(?:shall we (?:get started|begin|start))",
    r"^(?:ready to (?:get started|begin|start))",
    r"^(?:can you hear me)",
)

# Candidate phrases that are clarification requests, not real answers.
_CLARIFICATION_PATTERNS = (
    r"^can you (?:please )?repeat(?: the question)?",
    r"^(?:could you )?(?:please )?repeat(?: that)?",
    r"^can you (?:please )?(?:re-?frame|rephrase|restate)(?: the question)?",
    (
        r"^(?:i )?(?:don'?t|cannot|can'?t) (?:understand|get|hear)"
        r"(?: (?:you|that|it|the question))?"
    ),
    (
        r"^(?:i )?(?:didn'?t|could not|couldn'?t) (?:understand|get|hear)"
        r"(?: (?:you|that|it|the question))?"
    ),
    (
        r"^(?:sorry[,.]? )?(?:i )?(?:don'?t|didn'?t) (?:understand|get)"
        r"(?: (?:you|that|it|the question))?"
    ),
    r"^(?:are you (?:there|still there))[?.]?",
    r"^(?:hello)[?!]?$",
    r"^(?:can you hear me)[?.]?",
    r"^(?:is (?:anyone|somebody) there)[?.]?",
    r"^(?:what did you say)[?.]?",
    r"^(?:i can'?t (?:hear|get) (?:you|that|it))[.?]?",
)


# Short acknowledgement phrases Eve uses that are NOT real intake questions.
_ACK_PREFIXES = (
    "thanks",
    "thank you",
    "great",
    "got it",
    "perfect",
    "noted",
    "understood",
    "awesome",
    "sounds good",
    "i see",
    "okay",
    "ok,",
    "alright",
    "sure",
    "absolutely",
    "wonderful",
    "nice",
    "good to know",
    "i'll note",
    "i've noted",
    "interesting",
    "that's useful",
)

# Exact-match acknowledgements that must NEVER become a pending_question.
_ACK_EXACT = frozenset(
    {
        "thanks",
        "thanks.",
        "thanks for that",
        "thanks for that.",
        "thanks for sharing",
        "thanks for sharing that",
        "thanks for sharing that.",
        "got it",
        "got it.",
        "great",
        "great.",
        "interesting",
        "interesting.",
        "understood",
        "understood.",
        "that's useful",
        "that's useful.",
        "thanks, that's useful.",
        "thanks, that's useful",
    }
)

_USER_ACK_PREFIXES = (
    "yes",
    "yeah",
    "yep",
    "yup",
    "ready",
    "i'm ready",
    "im ready",
    "sure",
    "absolutely",
    "okay",
    "ok",
    "of course",
    "sounds good",
)

_USER_ACK_EXACT = frozenset(
    {
        "doing good",
        "doing good what about you",
        "i'm good",
        "im good",
        "i am good",
        "i'm fine",
        "im fine",
        "fine",
        "good",
        "hello",
        "hi",
        "hey",
        "yes",
        "yes i'm ready",
        "yes im ready",
        "yes ready",
    }
)


def _is_acknowledgement(text: str) -> bool:
    """Return whether an assistant turn is a short acknowledgement."""
    t = text.lower().strip()
    if t in _ACK_EXACT:
        return True
    # Must be short and not end with a question mark to be an acknowledgement
    if "?" in t:
        return False
    if len(t) > 120:
        return False
    return any(t.startswith(p) for p in _ACK_PREFIXES)


def _is_brief_user_acknowledgement(text: str) -> bool:
    """Return whether a short setup answer should not count as progress."""
    t = _clean_str(text).lower().strip().rstrip(" .!?")
    if not t:
        return False
    if len(t) > 40:
        return False
    if t in _USER_ACK_EXACT:
        return True
    return any(t == p or t.startswith(f"{p} ") for p in _USER_ACK_PREFIXES)


def _is_setup_question(text: str) -> bool:
    """Return True if the text is a setup/greeting question (not an intake question)."""
    t = text.lower().strip()
    return any(re.search(p, t) for p in _SETUP_QUESTION_PATTERNS)


def _is_clarification_request(text: str) -> bool:
    """Return whether a candidate turn is a clarification or connection check."""
    t = _clean_str(text).lower().strip().rstrip(" .!?")
    if not t:
        return False
    # Must be short — real answers are longer
    if len(t) > 80:
        return False
    return any(re.search(p, t) for p in _CLARIFICATION_PATTERNS)


# Statement-form intake prompts that don't end with '?' but request information.
_INTAKE_STATEMENT_PATTERNS = (
    r"\btell me about (?:your|the) background\b",
    r"\btell me about yourself\b",
    r"\bdescribe your (?:background|experience|skills)\b",
    r"\bwalk me through your (?:background|experience|career)\b",
    r"\bshare (?:your|a bit about your) background\b",
)

_INTAKE_QUESTION_PATTERNS = (
    r"\bopportunit(?:y|ies)\b",
    r"\brole(?:s)?\b",
    r"\bjob(?:s)?\b",
    r"\bcareer(?:s)?\b",
    r"\bbackground\b",
    r"\bexperience\b",
    r"\bskill(?:s)?\b",
    r"\btechnolog(?:y|ies)\b",
    r"\bframework(?:s)?\b",
    r"\bproject(?:s)?\b",
    r"\bteam\b",
    r"\benvironment(?:s)?\b",
    r"\bresponsibilit(?:y|ies)\b",
    r"\bindustr(?:y|ies)\b",
    r"\blocation\b",
    r"\bremote\b",
    r"\bonsite\b",
    r"\beducation\b",
    r"\bdegree\b",
    r"\bcertification(?:s)?\b",
    r"\bavailability\b",
    r"\bnotice period\b",
    r"\bsalary\b",
    r"\bcompensation\b",
    r"\bcurrent (?:role|job|company)\b",
    r"\bnext (?:role|job|opportunity|move)\b",
)

_QUESTION_START_PATTERN = re.compile(
    r"\b(?:"
    r"what(?:'s| is| kind of| type of| would| do| are)?|"
    r"which|how|who|why|where|when|"
    r"tell me|can you|could you|would you|do you|are you|is your"
    r")\b",
    re.IGNORECASE,
)


_ASSISTANT_FRAGMENT_FILLERS = frozenset({"um", "uh", "er", "ah"})
_ASSISTANT_FRAGMENT_CONNECTORS = frozenset(
    {
        "using",
        "with",
        "for",
        "to",
        "in",
        "on",
        "at",
        "from",
        "about",
        "of",
        "and",
        "or",
        "but",
    }
)


def _normalize_assistant_question_clause(clause: str) -> str:
    """Clean a single assistant clause before question extraction."""
    text = _clean_str(clause)
    if not text:
        return ""

    lowered = text.lower().strip(" ,.-")
    if lowered in _ASSISTANT_FRAGMENT_FILLERS:
        return ""

    text = re.sub(r"\b(?:um|uh|er|ah)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.-")
    if not text:
        return ""

    lowered = text.lower()
    if lowered in _ASSISTANT_FRAGMENT_CONNECTORS:
        return lowered
    return text


def _normalize_extracted_question_text(question_text: str) -> str:
    """Normalize extracted assistant question text into one canonical question."""
    text = _clean_str(question_text)
    if not text:
        return ""

    clauses = [c.strip() for c in re.split(r"(?<=[?.!])\s+", text) if c.strip()]
    normalized_clauses = [
        _normalize_assistant_question_clause(clause) for clause in clauses
    ]
    text = " ".join(clause for clause in normalized_clauses if clause)
    text = re.sub(
        r"\s+[,.!?]+\s+(?=\b(?:using|with|for|to|in|on|at|from|about|of|and|or|but)\b)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:using|with|for|to|in|on|at|from|about|of|and|or|but)\.\s+(?=\w)",
        lambda m: f"{m.group(0).split('.')[0]} ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:um|uh|er|ah)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.-")
    return text


def _extract_question_from_assistant_turn(text: str) -> str | None:
    """
    Extract the actual intake question from an assistant turn.
    Handles both '?' questions and statement-form intake prompts.
    """
    cleaned = _clean_str(text)
    if not cleaned:
        return None

    # Connection/setup prompts are conversation scaffolding, not intake turns.
    # Check the complete assistant utterance before looking for question words;
    # otherwise "Are you ready?" is mistaken for the first intake question and
    # the candidate's short "Yes" is associated with the next real question.
    if _is_setup_question(cleaned) and not (
        any(re.search(p, cleaned.lower()) for p in _INTAKE_STATEMENT_PATTERNS)
        or any(re.search(p, cleaned.lower()) for p in _INTAKE_QUESTION_PATTERNS)
    ):
        return None

    clauses = [c.strip() for c in re.split(r"(?<=[?.!])\s+", cleaned) if c.strip()]
    intake_start_idx: int | None = None
    for idx, clause in enumerate(clauses):
        lowered = clause.lower().rstrip(" .!?")
        if any(
            re.search(p, lowered) for p in _INTAKE_STATEMENT_PATTERNS
        ) or _QUESTION_START_PATTERN.search(clause):
            intake_start_idx = idx
            break

    # Check statement-form intake prompts (e.g. "Tell me about your background.")
    for clause in reversed(clauses):
        lowered = clause.lower().rstrip(" .!?")
        if any(re.search(p, lowered) for p in _INTAKE_STATEMENT_PATTERNS):
            return clause

    question_clause = next(
        (clause for clause in reversed(clauses) if "?" in clause), ""
    )
    if not question_clause:
        if intake_start_idx is None:
            return None
        question_clause = " ".join(clauses[intake_start_idx:])

    question_text = question_clause[: question_clause.rfind("?") + 1]
    if question_text == question_clause and not question_text.endswith("?"):
        question_text = question_text.rstrip(" .!,-") + "?"
    matches = list(_QUESTION_START_PATTERN.finditer(question_text))
    if matches:
        first_match = matches[0]
        prefix = question_text[: first_match.start()].strip(" ,.-")
        prefix_word_count = len(re.findall(r"[a-zA-Z]+", prefix))
        if first_match.start() > 0 and prefix_word_count <= 5:
            question_text = question_text[first_match.start() :]

    if not _is_intake_question(question_text) and intake_start_idx is not None:
        question_text = _clean_str(" ".join(clauses[intake_start_idx:]))

    question_text = _normalize_extracted_question_text(question_text)
    if not question_text:
        return None

    for filler in (
        "Using",
        "With",
        "For",
        "To",
        "In",
        "On",
        "At",
        "From",
        "About",
        "Of",
        "And",
        "Or",
        "But",
    ):
        question_text = re.sub(rf"\b{filler}\b", filler.lower(), question_text)

    question_text = question_text[0].upper() + question_text[1:]

    if not question_text.endswith("?"):
        question_text = question_text.rstrip(".") + "?"
    if _is_setup_question(question_text):
        return None
    if _is_acknowledgement(question_text):
        return None
    if not _is_intake_question(question_text):
        return None
    return question_text


def _is_intake_question(text: str) -> bool:
    """Return whether an assistant question asks for career/profile information."""
    cleaned = _clean_str(text)
    if not cleaned:
        return False

    if _is_employment_gap_question(cleaned):
        return True
    lowered = cleaned.lower()
    if any(re.search(pattern, lowered) for pattern in _INTAKE_STATEMENT_PATTERNS):
        return True
    return any(re.search(pattern, lowered) for pattern in _INTAKE_QUESTION_PATTERNS)


def _is_employment_gap_question(text: str) -> bool:
    lowered = _clean_str(text).lower()
    if "gap" not in lowered:
        return False
    return (
        "what should i note" in lowered
        or "what happened" in lowered
        or "can you explain" in lowered
        or "tell me about" in lowered
        or "share" in lowered
    )


def _questions_are_rephrasing(q1: str, q2: str) -> bool:
    """
    Return True if q2 is likely a rephrasing of q1 (same topic, different wording).
    Heuristic: significant overlap in the shorter question's content words.
    """
    if not q1 or not q2:
        return False
    stop = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "you",
        "your",
        "me",
        "my",
        "i",
        "it",
        "in",
        "on",
        "at",
        "to",
        "of",
        "and",
        "or",
        "but",
        "for",
        "with",
        "that",
        "this",
        "what",
        "how",
        "would",
        "could",
        "can",
        "will",
        "like",
        "about",
        "any",
        "some",
    }
    w1 = {w for w in re.findall(r"[a-z]+", q1.lower()) if w not in stop and len(w) > 2}
    w2 = {w for w in re.findall(r"[a-z]+", q2.lower()) if w not in stop and len(w) > 2}
    if not w1 or not w2:
        return False
    overlap = len(w1 & w2)
    shorter = min(len(w1), len(w2))
    return overlap / shorter >= 0.4


def _question_in_completed_turns(question: str, completed_turns: list[dict]) -> bool:
    """Return True when question matches one of the completed turns."""
    cleaned_question = _clean_str(question)
    if not cleaned_question:
        return False

    for turn in completed_turns:
        turn_question = _clean_str(turn.get("question"))
        if not turn_question:
            continue
        if cleaned_question == turn_question:
            return True
        if _questions_are_rephrasing(cleaned_question, turn_question):
            return True
        if _questions_are_rephrasing(turn_question, cleaned_question):
            return True
    return False


def _looks_fragmentary_question(question: str) -> bool:
    """Identify short assistant fragments that are not active questions."""
    cleaned = _clean_str(question)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if any(re.search(p, lowered) for p in _INTAKE_STATEMENT_PATTERNS):
        return False
    if _QUESTION_START_PATTERN.search(cleaned):
        return False

    words = re.findall(r"[a-zA-Z]+", cleaned)
    if len(words) <= 6 and cleaned.endswith("?"):
        return True
    return len(words) <= 4


def _choose_active_current_question(current_question: str, next_question: str) -> str:
    """Prefer next_question when current_question is only a fragment."""
    cleaned_current = _clean_str(current_question)
    cleaned_next = _clean_str(next_question)
    if (
        cleaned_current
        and cleaned_next
        and _looks_fragmentary_question(cleaned_current)
    ):
        return cleaned_next
    return cleaned_current


def _voice_notes_from_transcript(transcript: str) -> list[dict]:
    """Parse speaker-labelled transcripts when explicit voice_notes are absent."""
    notes: list[dict] = []
    for raw_line in (transcript or "").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        speaker, content = line.split(":", 1)
        role = speaker.strip().lower()
        text = content.strip()
        if not text:
            continue
        if role in {"assistant", "eve"}:
            notes.append({"role": "assistant", "text": text, "final": True})
        elif role in {"candidate", "user"}:
            notes.append({"role": "user", "text": text, "final": True})
    return notes


_VOICE_INTAKE_OFFTOPIC_PATTERNS = (
    r"\bappointment\b",
    r"\bcar\b",
    r"\bdoctor\b",
    r"\bdentist\b",
    r"\bclinic\b",
    r"\bhospital\b",
    r"\bmedication\b",
    r"\binsurance\b",
    r"\bpickup\b",
    r"\bdrop[- ]?off\b",
    r"\bcommute\b",
    r"\bschool\b",
    r"\bfamily\b",
    r"\bkid(?:s)?\b",
    r"\bhome\b",
    r"\blunch\b",
    r"\bdinner\b",
    r"\bweekend\b",
)

_VOICE_INTAKE_WORK_PATTERNS = (
    r"\bjob\b",
    r"\brole\b",
    r"\bwork\b",
    r"\bcareer\b",
    r"\bexperience\b",
    r"\bdeveloper\b",
    r"\bengineer\b",
    r"\bbackend\b",
    r"\bfrontend\b",
    r"\bjava\b",
    r"\bpython\b",
    r"\bspring\b",
    r"\bfastapi\b",
    r"\bproject\b",
    r"\btechnology\b",
    r"\bskills?\b",
    r"\bindustry\b",
    r"\bteam\b",
    r"\bapi\b",
)


def _candidate_fragment_is_off_topic(question: str, fragment: str) -> bool:
    """
    Best-effort filter for clearly unrelated candidate fragments.

    We only drop fragments when they look like logistics / personal questions and
    do not contain obvious work or career signals. Short but relevant fragments
    are preserved.
    """
    cleaned = _clean_str(fragment)
    if not cleaned:
        return False

    lowered = cleaned.lower()
    if any(re.search(pattern, lowered) for pattern in _VOICE_INTAKE_WORK_PATTERNS):
        return False

    off_topic_hits = sum(
        1 for pattern in _VOICE_INTAKE_OFFTOPIC_PATTERNS if re.search(pattern, lowered)
    )
    if off_topic_hits == 0:
        return False

    if lowered.endswith("?"):
        return True

    if re.search(
        r"^(do|does|did|is|are|was|were|can|could|would|will|should|have|has|had)\b",
        lowered,
    ):
        return True

    question_words = set(re.findall(r"[a-z]+", _clean_str(question).lower()))
    fragment_words = set(re.findall(r"[a-z]+", lowered))
    if question_words and fragment_words and len(question_words & fragment_words) == 0:
        return True

    return off_topic_hits >= 2


def _normalize_voice_notes(voice_notes: Any, transcript: str = "") -> list[dict]:
    if voice_notes:
        normalized: list[dict] = []
        for note in voice_notes:
            if isinstance(note, dict):
                normalized.append(
                    {
                        "role": note.get("role"),
                        "text": note.get("text", ""),
                        "final": note.get("final", True),
                    }
                )
            else:
                normalized.append(
                    {
                        "role": getattr(note, "role", None),
                        "text": getattr(note, "text", ""),
                        "final": getattr(note, "final", True),
                    }
                )
        return normalized
    return _voice_notes_from_transcript(transcript)


def _voice_intake_turn_pairs(
    voice_notes: Any, transcript: str = ""
) -> tuple[list[dict], str | None]:
    """
    Parse voice_notes into completed Q&A pairs.

    Rules:
    - Only final turns are considered.
    - Consecutive assistant fragments are combined before question detection.
    - An assistant turn that is a short acknowledgement does NOT start a new question;
      the previous real question remains active.
    - Any assistant turn containing a real question (ends with '?', not setup/greeting)
      is treated as an intake question — regardless of exact wording.
    - If a new question is a rephrasing of the current pending question, keep the
      original pending question text (do not create a new turn).
    - Consecutive user fragments belonging to the same answer are combined.
    - Candidate clarification requests ("Can you repeat?", "Are you there?", etc.)
      do NOT count as answers and do NOT clear the pending question.
    - User turns that occur before the first real question are discarded.
    - Returns (completed_pairs, pending_question) where pending_question is the last
      real assistant question that has not yet received a genuine user answer.
    """
    completed: list[dict] = []
    pending_question: str | None = None
    answer_parts: list[str] = []
    # Buffer consecutive assistant fragments so multi-fragment questions are combined
    assistant_buffer: list[str] = []

    def _flush_assistant_buffer() -> None:
        """Process buffered assistant fragments as one combined turn."""
        nonlocal pending_question, answer_parts
        if not assistant_buffer:
            return
        combined = " ".join(assistant_buffer)
        assistant_buffer.clear()

        if _is_acknowledgement(combined):
            return

        detected_q = _extract_question_from_assistant_turn(combined)
        if detected_q:
            # Check if this is a rephrasing of the current pending question
            if pending_question and _questions_are_rephrasing(
                pending_question, detected_q
            ):
                # Same question rephrased — keep original pending question, don't flush
                return
            # New real question: flush any accumulated answer first
            if answer_parts and pending_question:
                accumulated_answer = " ".join(answer_parts)
                # VAPI can emit a broad role prompt immediately before the
                # actual background prompt.  If the fragments clearly answer
                # background/experience (rather than role preference), carry
                # them forward so they stay attached to the first real intake
                # question.  This also keeps consecutive fragments together.
                answer_lower = accumulated_answer.lower()
                background_answer = (
                    bool(re.search(r"\b\d+\s+years?\b", answer_lower))
                    or "worked at" in answer_lower
                    or "working at" in answer_lower
                    or "built a product" in answer_lower
                )
                pending_is_role_prompt = bool(
                    re.search(r"\brole|opportunit|target", pending_question.lower())
                )
                next_is_background_prompt = bool(
                    re.search(r"\bbackground|experience|career", detected_q.lower())
                )
                if not (
                    background_answer
                    and pending_is_role_prompt
                    and next_is_background_prompt
                ):
                    completed.append(
                        {
                            "question": pending_question,
                            "answer": accumulated_answer,
                        }
                    )
                answer_parts = []
                if (
                    background_answer
                    and pending_is_role_prompt
                    and next_is_background_prompt
                ):
                    answer_parts.append(accumulated_answer)
            pending_question = detected_q
        else:
            # Non-question assistant chatter should not disturb the active intake
            # question. This includes setup/greeting context and profile statements.
            pass

    for note in _normalize_voice_notes(voice_notes, transcript):
        role = note.get("role")
        text = note.get("text", "")
        is_final = note.get("final", True)

        if not is_final:
            continue

        cleaned = _clean_str(text)
        if not cleaned:
            continue

        if role == "assistant":
            assistant_buffer.append(cleaned)
        elif role == "user":
            # Flush buffered assistant text before processing user turn
            _flush_assistant_buffer()

            if _is_brief_user_acknowledgement(cleaned):
                continue
            if _is_clarification_request(cleaned):
                # Keep pending_question active; do not count as an answer
                continue
            if pending_question and _candidate_fragment_is_off_topic(
                pending_question, cleaned
            ):
                continue
            if pending_question:
                answer_parts.append(cleaned)
            # else: before first real question — discard

    # Flush any remaining assistant buffer
    _flush_assistant_buffer()

    # Flush any trailing answer that hasn't been closed yet
    if answer_parts and pending_question:
        completed.append(
            {
                "question": pending_question,
                "answer": " ".join(answer_parts),
            }
        )
        pending_question = None

    return completed, pending_question


def _next_voice_intake_question(
    completed_turns: list[dict], pending_question: str | None = None
) -> str | None:
    """Return the pending question, or None for LLM selection."""
    return pending_question or None


# ---------- LLM-driven intake analysis ----------

VOICE_INTAKE_ANALYZE_SYSTEM = (
    "You are an expert recruitment assistant analyzing a voice inta"
    "ke conversation.\n\nYou will receive:\n- candidate_profile: exist"
    "ing resume/profile data already known\n- conversation: the voic"
    "e intake conversation so far (Q&A pairs already completed + an"
    "y partial answer)\n- intake_topics: high-level topics that shou"
    "ld be covered\n\nYour job:\n1. Determine what information is alre"
    "ady known (from profile OR conversation)\n2. Identify what was "
    "newly provided in the conversation\n3. Identify which intake to"
    "pics are still genuinely missing\n4. Determine if the current p"
    "ending question has been answered\n5. Suggest the single most u"
    "seful next question to ask (or null if all topics covered)\n\nAP"
    "PLICATION-CONTROLLED VOICE INTAKE SEQUENCE:\nThe application co"
    "ntrols which voice-intake topic/question is currently\nunanswer"
    "ed. If an application-provided next question is present in the"
    "\nconversation context, treat it as authoritative: ask that que"
    "stion, do not\nskip it or substitute a different work-preferenc"
    "e, role, salary, or location\nquestion. You may phrase it natur"
    "ally while preserving its meaning, and ask\nonly one question. "
    "The application sequence normally proceeds through why the\ncan"
    "didate is looking, background/current work, skills/technologie"
    "s, desired\nrole/career direction, preferences, goals, addition"
    "al professional detail,\ncareer-gap clarification, and completi"
    "on; skip topics already known or\nanswered. After background/cu"
    "rrent work, skills/technologies normally comes\nnext unless the"
    " application explicitly supplies another unanswered question.\n"
    "\nRules:\n- Process the newest candidate answer first. Treat it "
    "as the source of truth\n  for the question the candidate just a"
    "nswered, even if an older persisted\n  current_question points "
    "to a different topic. Save/merge that answer before\n  choosing"
    " the next question.\n- Never discard a meaningful answer merely"
    " because it does not match a stale\n  current_question. Classif"
    "y it against all intake topics, acknowledge it, and\n  only the"
    "n continue with the oldest genuinely missing topic.\n- Do NOT a"
    "sk about information already present in the candidate profile\n"
    "- Do NOT create duplicate questions for the same topic\n- A top"
    "ic is covered if the candidate provided meaningful information"
    " about it anywhere in the conversation or profile\n- Before com"
    "pleting career_preferences/availability_location, explicitly c"
    "ollect any still-unknown work mode (Remote/Hybrid/On-site/Flex"
    "ible), employment type, preferred locations, expected salary, "
    "and relocation preference. A candidate may decline a field; re"
    "cord that rather than guessing.\n- Employment history is not co"
    "mplete until every newly introduced or updated job has a start"
    ' date. For a current job, ask "When did you start working as <'
    'role>?" when missing and record end_date exactly as "Present".'
    " For a previous job, collect both start and end dates. Do not "
    "complete the employment-history topic while these dates are mi"
    "ssing unless the candidate explicitly cannot provide them.\n- B"
    "efore accepting any new work or education record, compare its "
    "start/end period with every saved record in that section. If p"
    "eriods overlap, or if more than one record would be current (e"
    "nd_date Present), do not save it yet. Tell the candidate which"
    " records overlap and ask whether they were concurrent or which"
    " dates should be corrected. This confirmation is required for "
    "both work experience and education.\n- next_question must be a "
    "natural, conversational question — not a hardcoded template\n- "
    "If all important topics are covered, set next_question to null"
    ' and completed to true\n\nReturn ONLY valid JSON:\n{\n  "known_top'
    'ics": ["topic1", "topic2"],\n  "missing_topics": ["topic3"],\n  '
    '"newly_provided": {"topic": "summary of what candidate said"},'
    '\n  "current_question_answered": true,\n  "next_question": "What'
    ' kind of backend role are you targeting next?",\n  "completed":'
    " false\n}"
)


async def _llm_analyze_intake(
    candidate_profile: dict,
    completed_turns: list[dict],
    pending_question: str | None,
    partial_answer: str = "",
    authoritative_next_question: str | None = None,
    vapi_call_id: str | None = None,
    *,
    endpoint: str,
    session_id: str | None = None,
    request_id: str | None = None,
) -> dict:
    """Ask the LLM to determine what's known, what's missing, and what to ask next."""
    raw_data = _parse_raw_data(candidate_profile.get("raw_data"))
    profile_summary = {
        "name": candidate_profile.get("name") or candidate_profile.get("name"),
        "current_role": candidate_profile.get("current_role")
        or candidate_profile.get("headline"),
        "current_company": candidate_profile.get("current_company"),
        "experience_years": candidate_profile.get("experience_years"),
        "skills": candidate_profile.get("skills") or [],
        # Include the complete timeline evidence.  Sending only title/company
        # made the analyzer believe resume dates were missing and caused it to
        # repeat the employment-date question.
        "work_experience": [
            {
                "title": w.get("title"),
                "company": w.get("company"),
                "start_date": w.get("start_date") or w.get("start") or "",
                "end_date": w.get("end_date") or w.get("end") or "",
                "dates": w.get("dates") or "",
                "is_current": w.get("is_current") is True,
            }
            for w in (candidate_profile.get("work_experience") or [])[:10]
            if isinstance(w, dict)
        ],
        "education": [
            {"degree": e.get("degree"), "institution": e.get("institution")}
            for e in (candidate_profile.get("education") or [])[:2]
        ],
        "summary": candidate_profile.get("summary") or "",
        "preferred_roles": candidate_profile.get("preferred_roles") or [],
        "preferred_locations": (
            candidate_profile.get("preferred_locations")
            or raw_data.get("preferred_locations")
            or raw_data.get("location_preferences")
            or []
        ),
        "availability": candidate_profile.get("availability") or "",
        "salary_expectation": candidate_profile.get("salary_expectation")
        or raw_data.get("salary_expectation")
        or "",
    }

    context = {
        "candidate_profile": profile_summary,
        # The final turn is the candidate's newest answer. Keep it last and
        # make that ordering explicit so a stale persisted question cannot
        # take precedence over the candidate's latest supplied detail.
        "conversation": completed_turns,
        "pending_question": pending_question,
        "partial_answer": partial_answer,
        "voice_intake_next_question": authoritative_next_question or "",
        "intake_topics": VOICE_INTAKE_TOPICS,
    }

    try:
        client = _dependencies.llm_client
        if callable(client):
            client = client()
        model = _dependencies.llm_model
        if callable(model):
            model = model()
        prompt = _dependencies.analyze_prompt
        if callable(prompt):
            prompt = prompt()
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(context)[:6000]},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            _telemetry={
                "workflow": "voice_intake_analysis",
                "function_name": "_llm_analyze_intake",
                "endpoint": endpoint,
                "vapi_call_id": vapi_call_id,
                "session_id": session_id,
                "request_id": request_id,
            },
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        if _dependencies.logger:
            _dependencies.logger.warning("[voice-intake] LLM analysis failed: %s", e)
        return {}


def _merge_voice_intake_topic_list(existing_values: Any, new_values: Any) -> list[str]:
    """
    Preserve persisted topic state when the incoming LLM analysis is missing or
    empty, while still using any new non-empty analysis.
    """
    if isinstance(new_values, list) and new_values:
        source = new_values
    elif isinstance(existing_values, list):
        source = existing_values
    else:
        source = []

    merged: list[str] = []
    seen: set[str] = set()
    for value in source:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)
    return merged


def _voice_intake_turns_to_transcript(completed_turns: list[dict]) -> str:
    """
    Reconstruct a cleaned transcript from completed turns.
    This intentionally excludes raw interrupted/off-topic fragments so profile
    extraction only sees the normalized conversational content.
    """
    lines: list[str] = []
    for turn in completed_turns:
        question = _clean_str(turn.get("question"))
        answer = _clean_str(turn.get("answer"))
        if question:
            lines.append(f"Assistant: {question}")
        if answer:
            lines.append(f"Candidate: {answer}")
    return "\n".join(lines)


def _promote_active_voice_question(
    current_question: str,
    next_question: str,
    completed_turns: list[dict],
) -> tuple[str, str]:
    """
    Keep the unanswered active question in current_question and ensure answered
    questions never remain there.
    """
    current = _clean_str(current_question)
    next_q = _clean_str(next_question)

    current_answered = _question_in_completed_turns(current, completed_turns)
    next_answered = _question_in_completed_turns(next_q, completed_turns)

    if current and current_answered:
        if next_q and not next_answered:
            current = next_q
        else:
            current = ""
    elif not current and next_q and not next_answered:
        current = next_q
        next_q = ""

    if current and next_q and _questions_are_rephrasing(current, next_q):
        # `next_question` is application-controlled canonical state.  A
        # spoken/LLM rephrasing may become the active current wording, but it
        # must not erase the canonical upcoming question from the resume.
        pass

    return current, next_q


def _merge_completed_turns(
    existing_turns: list[dict], new_turns: list[dict]
) -> list[dict]:
    """
    Return the union of existing and new completed turns.

    We dedupe by question text, but also treat rephrased questions as the same
    turn so repeated cumulative voice_notes do not create duplicates. Existing
    turns remain the source of truth unless the new answer is a longer cumulative
    version of the same turn.
    """
    merged = [dict(t) for t in existing_turns if _clean_str(t.get("question"))]

    def _find_match(question: str) -> int | None:
        for idx, existing in enumerate(merged):
            existing_q = _clean_str(existing.get("question"))
            if not existing_q:
                continue
            if existing_q == question:
                return idx
            if _questions_are_rephrasing(
                existing_q, question
            ) or _questions_are_rephrasing(question, existing_q):
                return idx
        return None

    for turn in new_turns:
        question = _clean_str(turn.get("question"))
        answer = _clean_str(turn.get("answer"))
        if not question:
            continue

        match_idx = _find_match(question)
        if match_idx is None:
            merged.append({"question": question, "answer": answer})
            continue

        existing = merged[match_idx]
        existing_answer = _clean_str(existing.get("answer"))
        if not existing_answer:
            existing["answer"] = answer
            continue
        if not answer or answer == existing_answer:
            continue
        if len(answer) > len(existing_answer) and existing_answer in answer:
            existing["answer"] = answer
        elif len(existing_answer) > len(answer) and answer in existing_answer:
            continue
        elif answer not in existing_answer:
            existing["answer"] = f"{existing_answer} {answer}".strip()
    return merged


def _build_voice_intake_resume_from_notes(
    voice_notes: Any,
    transcript: str = "",
    existing_resume: dict | None = None,
    candidate_profile: dict | None = None,
    llm_analysis: dict | None = None,
    preserve_authoritative_next: bool = True,
) -> dict:
    """
    Build the voice intake resume state from conversation notes.
    Uses LLM analysis when available; falls back to structural parsing.

    completed_turns is ALWAYS the union of previously persisted turns and
    newly parsed turns — never a replacement.
    """
    new_turns, pending_question = _voice_intake_turn_pairs(voice_notes, transcript)

    existing_turns = (existing_resume or {}).get("completed_turns") or []
    normalized_incoming_notes = _normalize_voice_notes(voice_notes, transcript)
    source_turns = (
        _merge_completed_turns(existing_turns, new_turns)
        if existing_turns
        else list(new_turns)
    )
    completed_turns = [
        {
            "question": _clean_str(turn.get("question")),
            "answer": _clean_str(turn.get("answer")),
        }
        for turn in source_turns
        if _clean_str(turn.get("question"))
    ]
    progress = len(completed_turns)

    # Prefer LLM-derived state; fall back to persisted state; fall back to parsed state
    existing_next_q = (existing_resume or {}).get("next_question") or ""
    existing_current_q = (existing_resume or {}).get("current_question") or ""
    existing_missing = (existing_resume or {}).get("missing_topics") or []
    existing_known = (existing_resume or {}).get("known_topics") or []
    existing_status = str(
        (existing_resume or {}).get("status") or "in_progress"
    ).lower()

    answered_question = _question_in_completed_turns(
        existing_current_q, completed_turns
    )

    if llm_analysis:
        llm_next_q = _clean_str(llm_analysis.get("next_question"))
        # A persisted application question is authoritative.  Re-running the
        # LLM while processing the same cumulative transcript must not replace
        # it with a newly inferred topic.
        next_question = existing_next_q or (
            llm_next_q if "next_question" in llm_analysis else ""
        )
        # An LLM may report that there is nothing else to ask for setup-only
        # chatter.  That is not an intake completion: completion requires at
        # least one genuine, answered intake turn.
        is_completed = bool(llm_analysis.get("completed")) and bool(completed_turns)
        missing_topics = _merge_voice_intake_topic_list(
            existing_missing, llm_analysis.get("missing_topics")
        )
        known_topics = _merge_voice_intake_topic_list(
            existing_known, llm_analysis.get("known_topics")
        )
        incoming_matches_existing = bool(existing_turns) and completed_turns == [
            {
                "question": _clean_str(turn.get("question")),
                "answer": _clean_str(turn.get("answer")),
            }
            for turn in existing_turns
            if _clean_str(turn.get("question"))
        ]
        if (
            llm_analysis.get("completed")
            and not pending_question
            and not incoming_matches_existing
        ):
            missing_topics = []
            is_completed = True
    else:
        next_question = existing_next_q
        missing_topics = existing_missing
        known_topics = existing_known
        is_completed = False

    # The question sequence advances by intake topic, not by whichever prompt
    # the model happened to select from the candidate's extracted facts.  In
    # particular, background is followed by skills; later topics remain fully
    # dynamic.  This keeps the assistant's final background prompt separate
    # from the next unanswered question.
    if (
        completed_turns
        and not existing_next_q
        and _normalize_profile_key(completed_turns[-1].get("question"))
        == _normalize_profile_key(VOICE_INTAKE_TOPIC_QUESTIONS["background_experience"])
    ):
        next_question = VOICE_INTAKE_TOPIC_QUESTIONS["skills_technologies"]

    # The active question and the canonical next question are separate state.
    # If analysis supplied an open question but omitted its canonical successor,
    # preserve a backend-owned successor instead of serializing an incomplete
    # in-progress resume.  This does not alter transcript reconstruction.
    if (
        not next_question
        and completed_turns
        and not (llm_analysis and llm_analysis.get("completed"))
    ):
        latest_key = _normalize_profile_key(completed_turns[-1].get("question"))
        for topic, question in VOICE_INTAKE_TOPIC_QUESTIONS.items():
            if latest_key == _normalize_profile_key(question):
                next_question = VOICE_INTAKE_FALLBACK_NEXT_QUESTIONS.get(topic)
                break
        # The first role question has several canonical phrasings in persisted
        # VAPI conversations.  They all advance to the same next topic.
        if not next_question and "role" in latest_key and "target" in latest_key:
            next_question = VOICE_INTAKE_FALLBACK_NEXT_QUESTIONS["target_role"]

    completion_ready = (
        bool(llm_analysis and llm_analysis.get("completed"))
        and bool(completed_turns)
        and not next_question
        and not pending_question
    )
    persisted_turn_questions = {
        _normalize_profile_key(turn.get("question"))
        for turn in existing_turns
        if _clean_str(turn.get("question"))
    }
    reconstructed_turn_questions = {
        _normalize_profile_key(turn.get("question"))
        for turn in completed_turns
        if _clean_str(turn.get("question"))
    }
    completed_state_unchanged = (
        existing_status == "completed"
        and bool(completed_turns)
        and reconstructed_turn_questions == persisted_turn_questions
    )

    # Preserve the existing state machine's completion decision, but do not let
    # stale persisted missing_topics block the final transition once the last
    # required question has been answered.
    if completion_ready:
        is_completed = True
        missing_topics = []
    elif (missing_topics or pending_question) and (
        existing_status != "completed" or pending_question
    ):
        is_completed = False

    current_question = pending_question
    if not current_question and existing_current_q and not answered_question:
        current_question = existing_current_q
    if (
        not current_question
        and answered_question
        and next_question
        and not _question_in_completed_turns(next_question, completed_turns)
    ):
        current_question = next_question
    if (
        not current_question
        and next_question
        and not _question_in_completed_turns(next_question, completed_turns)
    ):
        # Keep the active application question visible after cumulative-turn
        # merging.  It remains next_question as well so callers retain the
        # distinction between active and upcoming state.
        current_question = next_question
    promoted_current = _choose_active_current_question(
        current_question or "", next_question or ""
    )
    if promoted_current and promoted_current != current_question:
        current_question = promoted_current
    # Never infer a country from a city or state. Once a location answer has
    # been collected, require an explicit country confirmation unless one is
    # already present in the canonical profile/raw data.
    profile_raw = _parse_raw_data((candidate_profile or {}).get("raw_data"))
    known_country = (
        (candidate_profile or {}).get("country_code")
        or (candidate_profile or {}).get("country")
        or profile_raw.get("country_code")
        or profile_raw.get("country")
    )
    location_turn = next(
        (
            turn
            for turn in reversed(completed_turns)
            if "location" in _normalize_profile_key(turn.get("question"))
            or (
                "city" in _normalize_profile_key(turn.get("question"))
                and "country" in _normalize_profile_key(turn.get("question"))
            )
        ),
        None,
    )
    if location_turn and _clean_str(location_turn.get("answer")) and not known_country:
        current_question = (
            "And which country is that in?"
            if "city" in _normalize_profile_key(location_turn.get("question"))
            else "Which country are you currently based in?"
        )
        next_question = ""
        is_completed = False
    if (
        current_question
        and next_question
        and _questions_are_rephrasing(current_question, next_question)
        # An empty LLM analysis is a retry/failure signal, not a new state
        # decision.  Keep the persisted canonical next question intact even
        # when persisted current_question contains the same prompt.
        and not (existing_next_q and llm_analysis == {})
    ):
        # Once the unanswered question is promoted to the active slot, it is
        # no longer also an upcoming question.  Keeping both fields populated
        # makes disconnect/resume state appear to contain a duplicate prompt.
        next_question = ""
    genuinely_new_completed_transcript = (
        bool(llm_analysis and llm_analysis.get("completed"))
        and bool(completed_turns)
        and bool(existing_turns)
        and not pending_question
        and completed_turns
        != [
            {
                "question": _clean_str(turn.get("question")),
                "answer": _clean_str(turn.get("answer")),
            }
            for turn in existing_turns
            if _clean_str(turn.get("question"))
        ]
    )
    if genuinely_new_completed_transcript:
        # A genuinely new, complete transcript is allowed to finish the
        # intake; do not inherit the prior interrupted resume's active state.
        is_completed = True
        current_question = None
        next_question = None
    elif completed_state_unchanged and not pending_question:
        # A fresh LLM pass can reconstruct an intake question from the old
        # cumulative notes.  With no new canonical turn, it is stale
        # reconstruction rather than a newly unanswered required question.
        is_completed = True
        current_question = None
        next_question = None
    # An active question always keeps the intake resumable.  `current_question`
    # is application state, not evidence that the overall intake is complete.
    if (
        current_question
        and not genuinely_new_completed_transcript
        and not (existing_status == "completed" and not pending_question)
    ):
        is_completed = False
    elif existing_status == "completed" and not pending_question and completed_turns:
        # A later, non-idempotent transcript may contain no new canonical turn
        # while continuing an intake that was already completed.  Reuse that
        # persisted terminal state instead of reopening it from a fresh LLM
        # interpretation, including when that interpretation reconstructed a
        # stale active/next question.
        is_completed = True
        current_question = None
        next_question = None
    if (
        not is_completed
        and completed_turns
        and not current_question
        and not next_question
        and not pending_question
    ):
        # If the state machine has no active question left, normalize any stale
        # persisted in_progress snapshot into the completed form.
        is_completed = True
        missing_topics = []

    # A completed intake may be submitted again with additional commentary.
    # Preserve its terminal state only after reconstruction confirms that the
    # new transcript did not leave a pending question or canonical next topic.
    # This is deliberately narrower than carrying forward the old status: a
    # genuinely new pending question must still reopen the intake.
    if (
        existing_status == "completed"
        and completed_turns
        and not pending_question
        and not next_question
    ):
        is_completed = True
        missing_topics = []

    completed_turn_answers = {
        _normalize_profile_key(turn.get("question")): _clean_str(turn.get("answer"))
        for turn in completed_turns
        if _clean_str(turn.get("question")) and _clean_str(turn.get("answer"))
    }
    employment_gap = (
        _detect_employment_gap(candidate_profile or {}) if candidate_profile else None
    )
    employment_gaps = [
        dict(g)
        for g in ((existing_resume or {}).get("employment_gaps") or [])
        if isinstance(g, dict)
    ]
    for gap in employment_gaps:
        question_key = _normalize_profile_key(gap.get("question"))
        answered_text = completed_turn_answers.get(question_key)
        if answered_text and not _clean_str(gap.get("answer")):
            gap["answer"] = answered_text

    def _answer_after_gap_question(question_text: str) -> str:
        question_key = _normalize_profile_key(question_text)
        for index, note in enumerate(normalized_incoming_notes):
            if _normalize_profile_key(note.get("text")) != question_key:
                continue
            if note.get("role") != "assistant":
                continue
            for follower in normalized_incoming_notes[index + 1 :]:
                if follower.get("role") != "user":
                    continue
                answer = _clean_str(follower.get("text"))
                if answer:
                    return answer
        return ""

    if employment_gap:
        existing_gap = next(
            (
                gap
                for gap in employment_gaps
                if _normalize_profile_key(gap.get("gap_key"))
                == _normalize_profile_key(employment_gap["gap_key"])
            ),
            None,
        )
        if existing_gap and _clean_str(existing_gap.get("answer")):
            employment_gap = None
        else:
            matched_gap_question = (
                existing_gap.get("question")
                if existing_gap
                else employment_gap["question"]
            )
            gap_answer = _clean_str(existing_gap.get("answer")) if existing_gap else ""
            if not gap_answer and matched_gap_question:
                gap_answer = _answer_after_gap_question(matched_gap_question)
                if gap_answer and existing_gap:
                    existing_gap["answer"] = gap_answer

            if gap_answer:
                employment_gap = None
                if _normalize_profile_key(
                    current_question or ""
                ) == _normalize_profile_key(matched_gap_question):
                    current_question = None
                if _normalize_profile_key(
                    next_question or ""
                ) == _normalize_profile_key(matched_gap_question):
                    next_question = ""
                # Mark gap as explained in employment_gaps list
                if existing_gap:
                    existing_gap["status"] = "explained"
            else:
                # Gap is unanswered — mandatory: override current/next question
                is_completed = False
                gap_q = employment_gap["question"]
                missing_topics = _merge_voice_intake_topic_list(
                    missing_topics, ["employment_gap"]
                )
                if existing_gap:
                    existing_gap["question"] = gap_q
                    existing_gap["gap_days"] = employment_gap["gap_days"]
                    existing_gap["previous_label"] = employment_gap["previous_label"]
                    existing_gap["current_label"] = employment_gap["current_label"]
                    existing_gap["previous_end_label"] = employment_gap[
                        "previous_end_label"
                    ]
                    existing_gap["current_start_label"] = employment_gap[
                        "current_start_label"
                    ]
                    existing_gap.setdefault("status", "unanswered")
                    answered_text = completed_turn_answers.get(
                        _normalize_profile_key(gap_q)
                    )
                    if answered_text:
                        existing_gap["answer"] = answered_text
                        existing_gap["status"] = "explained"
                        employment_gap = None
                else:
                    employment_gaps.append(
                        {
                            "gap_key": employment_gap["gap_key"],
                            "question": gap_q,
                            "gap_days": employment_gap["gap_days"],
                            "previous_label": employment_gap["previous_label"],
                            "current_label": employment_gap["current_label"],
                            "previous_end_label": employment_gap["previous_end_label"],
                            "current_start_label": employment_gap[
                                "current_start_label"
                            ],
                            "status": "unanswered",
                        }
                    )
                # Mandatory: gap question takes priority over any other question
                if employment_gap is not None:
                    current_question = gap_q
                    next_question = ""

    status = "completed" if is_completed else "in_progress"
    if is_completed:
        current_question = None
        next_question = None

    resume: dict = {
        "status": status,
        "progress": progress,
        "voice_notes": _normalize_voice_notes(voice_notes, transcript)
        or (existing_resume or {}).get("voice_notes")
        or [],
        "completed_turns": completed_turns,
        "has_open_question": bool(current_question or next_question),
        "known_topics": known_topics,
        "missing_topics": missing_topics,
    }
    if employment_gaps:
        resume["employment_gaps"] = employment_gaps
    if completed_turns:
        resume["latest_completed_question"] = completed_turns[-1]["question"]
        resume["latest_completed_answer"] = completed_turns[-1]["answer"]
    # Keep the canonical state shape stable for persistence and response
    # serialization, including empty terminal values.
    resume["next_question"] = next_question or None
    resume["current_question"] = current_question or None

    return resume



def _build_voice_intake_resume(profile: dict) -> dict | None:
    raw_data = _parse_raw_data(profile.get("raw_data"))
    voice_intake = _parse_raw_data(raw_data.get("voice_intake"))
    if not voice_intake:
        return None
    status = str(voice_intake.get("status") or "").lower()
    if status == "completed":
        return None
    status = "in_progress"

    saved_completed_turns = voice_intake.get("completed_turns") or []
    saved_progress = voice_intake.get("progress")
    saved_next_question = voice_intake.get("next_question")
    saved_current_question = voice_intake.get("current_question")
    saved_missing_topics = voice_intake.get("missing_topics") or []
    saved_known_topics = voice_intake.get("known_topics") or []

    if saved_completed_turns:
        completed_turns = saved_completed_turns
        progress = (
            int(saved_progress) if saved_progress is not None else len(completed_turns)
        )
        current_question = (
            saved_current_question
            if not _question_in_completed_turns(saved_current_question, completed_turns)
            else None
        )
        next_question = saved_next_question
    else:
        voice_notes = voice_intake.get("voice_notes") or []
        completed_turns, pending_question = _voice_intake_turn_pairs(voice_notes)
        progress = (
            int(saved_progress) if saved_progress is not None else len(completed_turns)
        )
        current_question = pending_question
        if (
            not current_question
            and saved_current_question
            and not _question_in_completed_turns(
                saved_current_question, completed_turns
            )
        ):
            current_question = saved_current_question
        next_question = saved_next_question

    current_question, next_question = _promote_active_voice_question(
        current_question or "", next_question or "", completed_turns
    )

    resume = {
        "status": status,
        "progress": progress,
        "completed_turns": completed_turns,
        "has_open_question": bool(current_question or next_question),
        "missing_topics": saved_missing_topics,
        "known_topics": saved_known_topics,
    }
    if completed_turns:
        resume["latest_completed_question"] = completed_turns[-1]["question"]
        resume["latest_completed_answer"] = completed_turns[-1]["answer"]
    resume["next_question"] = next_question or None
    resume["current_question"] = current_question or None
    return resume
