"""Behavioral tests for preferred-location parsing and matching."""

from location_matching import location_preference_matches


def job(**values):
    return values


def test_city_only_preference_requires_city_match():
    assert location_preference_matches(
        "Hyderabad", job(city="Hyderabad", country="India")
    )
    assert not location_preference_matches(
        "Hyderabad", job(city="Pune", country="India")
    )


def test_city_and_state_require_both_components():
    preference = "Hyderabad, Telangana"
    assert location_preference_matches(
        preference, job(city="Hyderabad", state="Telangana", country="India")
    )
    assert not location_preference_matches(
        preference, job(city="Hyderabad", state="Andhra Pradesh", country="India")
    )
    assert not location_preference_matches(
        preference, job(city="Pune", state="Telangana", country="India")
    )


def test_raw_two_part_city_and_state_location_matches_consistently():
    assert location_preference_matches(
        "Hyderabad, Telangana", job(location="Hyderabad, Telangana", country="India")
    )


def test_raw_two_part_state_and_country_location_matches_consistently():
    assert location_preference_matches(
        "California, USA", job(location="California, USA")
    )


def test_raw_two_part_state_country_does_not_match_city_only_preference():
    assert not location_preference_matches(
        "California", job(location="California, USA")
    )


def test_four_part_raw_location_does_not_infer_components():
    assert not location_preference_matches(
        "Hyderabad, Telangana, India",
        job(location="Building, Hyderabad, Telangana, India"),
    )


def test_structured_location_fields_survive_malformed_raw_location():
    assert location_preference_matches(
        "Hyderabad, Telangana, India",
        job(
            location="Building, Floor 2, Hyderabad, Telangana, India",
            city="Hyderabad",
            state="Telangana",
            country="India",
        ),
    )


def test_empty_raw_location_components_do_not_infer_a_match():
    assert not location_preference_matches(
        "Hyderabad, Telangana, India", job(location="Hyderabad,,India")
    )
    assert not location_preference_matches(
        "Hyderabad, Telangana, India", job(location="Hyderabad, Telangana,")
    )


def test_structured_location_fields_survive_empty_raw_components():
    assert location_preference_matches(
        "Hyderabad, Telangana, India",
        job(
            location="Hyderabad,,India",
            city="Hyderabad",
            state="Telangana",
            country="India",
        ),
    )


def test_state_and_country_require_both_components():
    preference = "California, USA"
    assert location_preference_matches(
        preference, job(state="California", country="United States")
    )
    assert not location_preference_matches(
        preference, job(state="California", country="Canada")
    )
    assert not location_preference_matches(
        preference, job(state="Texas", country="United States")
    )


def test_three_part_location_requires_all_components():
    preference = "Hyderabad, Telangana, India"
    assert location_preference_matches(
        preference, job(city="Hyderabad", state="Telangana", country="IN")
    )
    assert location_preference_matches(
        preference, job(location="Hyderabad, Telangana, India")
    )
    assert not location_preference_matches(
        preference, job(location="Hyderabad, Andhra Pradesh, India")
    )
    assert not location_preference_matches(
        preference, job(city="Hyderabad", state="Telangana", country="USA")
    )
    assert not location_preference_matches(
        preference, job(city="Pune", state="Telangana", country="India")
    )
    assert not location_preference_matches(
        preference, job(city="Hyderabad", state="Andhra Pradesh", country="India")
    )


def test_country_only_preference_and_aliases():
    assert location_preference_matches("India", job(country="IN"))
    assert location_preference_matches("IN", job(country="India"))
    assert location_preference_matches("USA", job(country="US"))
    assert not location_preference_matches("India", job(country="USA"))


def test_unrecognized_second_component_is_state_not_country():
    preference = "Hyderabad, Atlantis"
    assert location_preference_matches(
        preference, job(city="Hyderabad", state="Atlantis", country="India")
    )
    assert not location_preference_matches(
        preference, job(city="Pune", state="Atlantis", country="India")
    )
    assert location_preference_matches(
        preference, job(city="Hyderabad", state="Atlantis", country="USA")
    )


def test_missing_components_and_invalid_preferences_fail_closed():
    assert not location_preference_matches(
        "Hyderabad, Telangana", job(state="Telangana", country="India")
    )
    assert not location_preference_matches(
        "Hyderabad, Telangana", job(city="Hyderabad", country="India")
    )
    assert not location_preference_matches(
        "Hyderabad, Telangana, India", job(city="Hyderabad", state="Telangana")
    )
    assert not location_preference_matches(
        "Hyderabad,,India", job(city="Hyderabad", state="Telangana", country="India")
    )
    assert not location_preference_matches("", job(city="Hyderabad", country="India"))
    assert not location_preference_matches("Hyderabad", job())


def test_remote_and_worldwide_jobs_preserve_country_rules():
    assert location_preference_matches("India", job(country="India", remote=True))
    assert not location_preference_matches("India", job(country="USA", remote=True))
    assert not location_preference_matches("India", job(remote_policy="worldwide"))


def test_multiple_preferred_locations_match_any_location():
    assert location_preference_matches(
        ["Pune, Maharashtra, India", "Hyderabad, Telangana, India"],
        job(locations=["Chennai, Tamil Nadu, India", "Hyderabad, Telangana, India"]),
    )


def test_each_string_in_a_preference_list_is_evaluated_independently():
    preferences = ["Hyderabad, Telangana, India", "Bengaluru, Karnataka, India"]
    assert location_preference_matches(
        preferences, job(city="Hyderabad", state="Telangana", country="India")
    )
    assert location_preference_matches(
        preferences, job(city="Bengaluru", state="Karnataka", country="India")
    )


def test_nonmatching_preference_list_does_not_match():
    preferences = ("Pune, Maharashtra, India", "Chennai, Tamil Nadu, India")
    assert not location_preference_matches(
        preferences, job(city="Hyderabad", state="Telangana", country="India")
    )


def test_empty_and_invalid_preference_values_do_not_match():
    assert not location_preference_matches(
        [None, "", 42, {}], job(city="Hyderabad", country="India")
    )
    assert not location_preference_matches(None, job(city="Hyderabad", country="India"))


def test_unknown_country_is_not_a_confirmed_mismatch():
    from location_matching import country_eligible

    assert country_eligible(
        {"location": "Hyderabad"}, job(city="Hyderabad", country="India")
    )
    assert country_eligible(
        {"country": "India"}, job(remote=True, remote_policy="remote")
    )


def test_country_codes_reject_arbitrary_two_letter_values():
    from location_matching import country_code

    assert country_code("ZZ") == ""


def test_structured_provider_location_is_used_when_legacy_fields_are_empty():
    assert location_preference_matches(
        "Pune, Maharashtra, India",
        job(
            structured_data={
                "location_details": {
                    "city": "Pune",
                    "state": "Maharashtra",
                    "country": "India",
                }
            }
        ),
    )


def test_multiple_locations_do_not_allow_partial_text_matches():
    assert not location_preference_matches(
        "Pune, Maharashtra, India",
        job(locations=["Puneet, Maharashtra, India"]),
    )


def test_onsite_job_without_location_does_not_bypass_explicit_preference():
    from candidate_job_matching_service import _preference_eligibility

    signals = {
        "_candidate": {"country": "India"},
        "preferred_locations": ["Hyderabad, Telangana, India"],
        "willing_to_relocate": False,
    }
    eligible, reasons = _preference_eligibility(
        signals,
        {"country": "India", "remote_policy": "onsite"},
        3,
    )
    assert not eligible
    assert "job_outside_explicit_locations" in reasons
