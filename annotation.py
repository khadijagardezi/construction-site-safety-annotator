from __future__ import annotations

import json
from typing import Dict, List, Optional

OBJECT_CATEGORIES = [
    "person",
    "heavy_vehicle",
    "vehicle",
    "truck",
    "excavator",
    "crane",
    "other",
]

HEAVY_VEHICLE_TYPES = {
    "heavy_vehicle", "truck", "excavator", "crane", "bulldozer", "forklift", "loader",
}

PPE_FIELDS = ["helmet", "safety_vest"]


def make_object(
    obj_id: int,
    obj_type: str,
    confidence: float,
    ppe: Optional[Dict[str, bool]] = None,
) -> Dict:
    obj: Dict = {
        "id": obj_id,
        "type": obj_type,
        "confidence": round(float(confidence), 2),
    }
    if obj_type == "person":
        ppe = ppe or {}
        obj["ppe"] = {field: bool(ppe.get(field, False)) for field in PPE_FIELDS}
    return obj


def build_annotation(file_name: str, objects: List[Dict]) -> Dict:
    return {
        "file_name": file_name,
        "objects": objects,
    }


def to_json_string(annotation: Dict) -> str:
    return json.dumps(annotation, indent=2, ensure_ascii=False)


def summarize(annotation: Dict) -> Dict[str, int]:
    objs = annotation["objects"]
    persons = [o for o in objs if o["type"] == "person"]
    return {
        "people_detected": len(persons),
        "heavy_vehicles_detected": sum(1 for o in objs if o["type"] in HEAVY_VEHICLE_TYPES),
        "missing_helmets": sum(1 for o in persons if not o["ppe"]["helmet"]),
        "missing_safety_vests": sum(1 for o in persons if not o["ppe"]["safety_vest"]),
    }
