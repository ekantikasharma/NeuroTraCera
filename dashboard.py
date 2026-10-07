from collections import Counter

def build_dashboard(analyses):
    predictions = [a.get("predicted_class", a.get("prediction")) for a in analyses]
    return {
        "total_analyses": len(analyses),
        "class_distribution": dict(Counter(predictions)),
        "average_confidence": (
            sum(float(a.get("probability", 0)) for a in analyses) / len(analyses)
            if analyses else 0
        ),
        "high_uncertainty_cases": sum(
            1 for a in analyses if a.get("uncertainty_level") == "High"
        ),
    }
