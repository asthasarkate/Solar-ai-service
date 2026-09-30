import pytest
from app.severity_engine import (
    get_recommended_professional,
    analyze_prediction,
    get_diy_guidance,
    is_diy_safe,
    DIY_GUIDANCE
)

def test_get_recommended_professional_valid():
    assert get_recommended_professional("Dust", "Medium") == "Professional Panel Cleaning Service"
    assert get_recommended_professional("Cracks", "High") == "Licensed Electrician + Solar Technician"

def test_get_recommended_professional_invalid():
    with pytest.raises(ValueError):
        get_recommended_professional("InvalidFault", "Low")
    with pytest.raises(ValueError):
        get_recommended_professional("Dust", "Unknown")

def test_diy_guidance_all_combinations():
    for key, text in DIY_GUIDANCE.items():
        assert text.strip() != "", f"Empty guidance for {key}"
        assert get_diy_guidance(*key) == text

def test_is_diy_safe():
    assert is_diy_safe("Dust", "Low") is True
    assert is_diy_safe("Dust", "High") is True
    assert is_diy_safe("Shading", "Medium") is True
    assert is_diy_safe("Shading", "High") is False
    assert is_diy_safe("Cracks", "Low") is False
    assert is_diy_safe("Physical Damage", "Medium") is False

def test_analyze_prediction():
    result = analyze_prediction("cracks", 0.8)
    assert result["faultType"] == "Cracks"
    assert result["severity"] == "High"
    assert result["confidence"] == 80.0
    assert result["recommendation"] == "Replace the damaged panel."
    assert result["recommendedProfessional"] == "Licensed Electrician + Solar Technician"
    assert result["diySafe"] is False
    assert "Do not touch or attempt any repair" in result["diyGuidance"]
