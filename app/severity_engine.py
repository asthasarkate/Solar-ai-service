SEVERITY_MAP = {
    "dust": "Medium",
    "cracks": "High",
    "physical_damage": "High",
    "shading": "Low",
}

RECOMMENDATION_MAP = {
    "dust": "Clean the panels.",
    "cracks": "Replace the damaged panel.",
    "physical_damage": "Replace the damaged panel.",
    "shading": "Remove the obstruction causing shading.",
}

LABEL_MAP = {
    "dust": "Dust",
    "cracks": "Cracks",
    "physical_damage": "Physical Damage",
    "shading": "Shading",
}

PROFESSIONAL_MAP = {
    ("Dust", "Low"): "In-house Maintenance Staff",
    ("Dust", "Medium"): "Professional Panel Cleaning Service",
    ("Dust", "High"): "Professional Panel Cleaning Service",
    ("Shading", "Low"): "Site Maintenance Team",
    ("Shading", "Medium"): "Site Maintenance Team / Arborist",
    ("Shading", "High"): "Solar System Installer",
    ("Cracks", "Low"): "Certified Solar Panel Technician",
    ("Cracks", "Medium"): "Certified Solar Panel Technician",
    ("Cracks", "High"): "Licensed Electrician + Solar Technician",
    ("Physical Damage", "Low"): "Solar Installation Technician",
    ("Physical Damage", "Medium"): "Solar Installation Technician",
    ("Physical Damage", "High"): "Licensed Electrician + Structural Engineer",
}

DIY_GUIDANCE = {
    ("Dust", "Low"): "Safe to clean yourself. Turn off the system if possible, then gently rinse the panel with plain water in the early morning or evening (never on a hot panel, as sudden cooling can crack the glass). Use a soft brush or sponge -- avoid abrasive materials, pressure washers, or harsh chemicals.",
    ("Dust", "Medium"): "Safe to clean yourself. Turn off the system if possible, then gently rinse the panel with plain water in the early morning or evening (never on a hot panel, as sudden cooling can crack the glass). Use a soft brush or sponge -- avoid abrasive materials, pressure washers, or harsh chemicals. If dust is heavily caked on, a mild soap-water solution can help, but rinse thoroughly afterward. Consider cleaning both sides if accessible.",
    ("Dust", "High"): "Safe to clean yourself. Turn off the system if possible, then gently rinse the panel with plain water in the early morning or evening (never on a hot panel, as sudden cooling can crack the glass). Use a soft brush or sponge -- avoid abrasive materials, pressure washers, or harsh chemicals. If accumulation is severe or recurring frequently, consider a professional cleaning service in addition to your own routine cleaning.",
    ("Shading", "Low"): "If shading is from overgrown vegetation, trimming the branch or plant causing it is safe to do yourself. Always prioritize ladder safety and never touch the panel surface while working near it.",
    ("Shading", "Medium"): "If shading is from overgrown vegetation, trimming the branch or plant causing it is safe to do yourself. Always prioritize ladder safety and never touch the panel surface while working near it. If the obstruction isn't vegetation (e.g. a fixed structure), DIY removal may not be possible -- assess whether trimming alone resolves it.",
    ("Shading", "High"): "This likely requires panel repositioning, which involves electrical disconnection. This is not a DIY task -- contact a solar installer.",
    ("Cracks", "Low"): "Not safe to attempt yourself. A cracked panel can expose live electrical components beneath the glass, even at low severity. Avoid touching the panel and contact a certified technician.",
    ("Cracks", "Medium"): "Not safe to attempt yourself. A cracked panel exposes live electrical components beneath the glass. Do not attempt any DIY intervention. Contact a certified technician immediately.",
    ("Cracks", "High"): "Do not touch or attempt any repair. Risk of electric shock is significant. Isolate the panel from the system if safely possible and contact a licensed electrician immediately.",
    ("Physical Damage", "Low"): "Not a DIY repair. Physical damage compromises the frame and potential structural integrity, and can expose live wiring. Do not touch; contact a solar technician.",
    ("Physical Damage", "Medium"): "Not a DIY repair. Physical damage compromises the frame and potential structural integrity, and can expose live wiring. Do not touch; contact a solar technician.",
    ("Physical Damage", "High"): "Not a DIY repair. Severe physical damage compromises structural integrity and presents significant electrical hazard. Do not attempt repairs. Contact a licensed electrician and structural engineer.",
}

def get_severity(class_name: str, confidence: float) -> str:
    if confidence < 0.5:
        return "Low"
    return SEVERITY_MAP.get(class_name, "Low")

def get_recommendation(class_name: str) -> str:
    if class_name not in RECOMMENDATION_MAP:
        raise ValueError(f"Invalid class_name: {class_name}")
    return RECOMMENDATION_MAP[class_name]

def get_recommended_professional(fault_type: str, severity: str) -> str:
    key = (fault_type, severity)
    if key not in PROFESSIONAL_MAP:
        raise ValueError(f"Invalid combination: {fault_type}, {severity}")
    return PROFESSIONAL_MAP[key]

def get_diy_guidance(fault_type: str, severity: str) -> str:
    key = (fault_type, severity)
    if key not in DIY_GUIDANCE:
        raise ValueError(f"Invalid combination for DIY guidance: {fault_type}, {severity}")
    return DIY_GUIDANCE[key]

def is_diy_safe(fault_type: str, severity: str) -> bool:
    if fault_type == "Dust":
        return True
    if fault_type == "Shading" and severity in ("Low", "Medium"):
        return True
    return False

def analyze_prediction(raw_class_name: str, raw_confidence: float) -> dict:
    if raw_class_name not in LABEL_MAP:
        raise ValueError(f"Invalid class_name: {raw_class_name}")

    fault_type = LABEL_MAP[raw_class_name]
    severity = get_severity(raw_class_name, raw_confidence)
    confidence_pct = round(raw_confidence * 100, 1)
    recommendation = get_recommendation(raw_class_name)
    recommended_professional = get_recommended_professional(fault_type, severity)
    diy_guidance = get_diy_guidance(fault_type, severity)
    diy_safe = is_diy_safe(fault_type, severity)

    return {
        "faultType": fault_type,
        "severity": severity,
        "confidence": confidence_pct,
        "recommendation": recommendation,
        "recommendedProfessional": recommended_professional,
        "diyGuidance": diy_guidance,
        "diySafe": diy_safe,
    }
