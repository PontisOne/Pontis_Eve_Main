from server import _chat_profile_preflight, _timeline_collision


def test_work_timeline_collision_is_detected():
    existing = [{
        "title": "Automation Engineer",
        "company": "Pontis One",
        "start_date": "2025-08",
        "end_date": "Present",
    }]
    incoming = [{
        "title": "Python Developer",
        "company": "Qantis",
        "start_date": "2025-09",
        "end_date": "Present",
    }]

    assert _timeline_collision(existing, incoming, "work_experience")


def test_education_timeline_collision_is_detected():
    existing = [{
        "degree": "B.Tech",
        "institution": "University A",
        "start_date": "2020-01",
        "end_date": "2024-06",
    }]
    incoming = [{
        "degree": "M.Tech",
        "institution": "University B",
        "start_date": "2023-09",
        "end_date": "Present",
    }]

    assert _timeline_collision(existing, incoming, "education")


def test_chat_preflight_asks_before_saving_overlapping_work_period():
    candidate = {
        "work_experience": [{
            "title": "Automation Engineer",
            "company": "Pontis One",
            "start_date": "2025-08",
            "end_date": "Present",
        }],
        "education": [],
    }
    result = _chat_profile_preflight(
        "I work as Python Developer at Qantis from September 2025 to Present.",
        candidate,
        [{"role": "user", "content": "I work as Python Developer at Qantis from September 2025 to Present."}],
    )

    assert result["updates"] is None
    assert "overlap" in result["reply"]
