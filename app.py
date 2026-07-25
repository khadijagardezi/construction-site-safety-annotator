import hashlib
import io

import numpy as np
import streamlit as st
from PIL import Image

import annotation as ann
import detector

st.set_page_config(page_title="Construction-Site Safety Annotator", layout="wide")


@st.cache_resource(show_spinner="Loading YOLO model…")
def get_model():
    return detector.load_model()


@st.cache_data(show_spinner="Detecting objects…")
def run_detection(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.array(img)
    dets = detector.detect(get_model(), img)
    annotated = detector.draw_detections(rgb, dets)
    return dets, annotated


st.title("Construction-Site Safety Annotator")

uploaded = st.file_uploader(
    "Upload a construction-site image", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Upload a JPG / JPEG / PNG image to begin.")
    st.stop()

image_bytes = uploaded.getvalue()
file_id = hashlib.md5(image_bytes).hexdigest()

dets, annotated = run_detection(image_bytes)

if st.session_state.get("file_id") != file_id:
    st.session_state.file_id = file_id
    st.session_state.objects = [
        {
            "id": d["id"],
            "type": d["type"],
            "confidence": d["confidence"],
            "ppe": dict(d.get("ppe") or {f: False for f in ann.PPE_FIELDS}),
        }
        for d in dets
    ]

left, right = st.columns([3, 2])

with left:
    st.subheader("Detected objects")
    st.image(annotated, channels="RGB", use_container_width=True)
    if not dets:
        st.warning("No people or vehicles detected.")

with right:
    st.subheader("Review & correct objects")
    for obj in st.session_state.objects:
        oid = obj["id"]
        with st.expander(f"Object {oid} — {obj['type']} ({obj['confidence']:.0%})", expanded=False):
            obj["type"] = st.selectbox(
                "Category",
                ann.OBJECT_CATEGORIES,
                index=(
                    ann.OBJECT_CATEGORIES.index(obj["type"])
                    if obj["type"] in ann.OBJECT_CATEGORIES
                    else ann.OBJECT_CATEGORIES.index("other")
                ),
                key=f"cat_{oid}",
            )
            if obj["type"] == "person":
                st.caption("PPE auto-detected — correct if wrong:")
                c1, c2 = st.columns(2)
                obj["ppe"]["helmet"] = c1.checkbox(
                    "Helmet", value=obj["ppe"].get("helmet", False), key=f"helmet_{oid}"
                )
                obj["ppe"]["safety_vest"] = c2.checkbox(
                    "Safety vest", value=obj["ppe"].get("safety_vest", False), key=f"vest_{oid}"
                )

export_objects = [
    ann.make_object(
        obj["id"],
        obj["type"],
        obj["confidence"],
        ppe=obj["ppe"] if obj["type"] == "person" else None,
    )
    for obj in st.session_state.objects
]

annotation = ann.build_annotation(file_name=uploaded.name, objects=export_objects)
summary = ann.summarize(annotation)

st.divider()
st.subheader("Summary")
m = st.columns(4)
m[0].metric("People", summary["people_detected"])
m[1].metric("Heavy vehicles", summary["heavy_vehicles_detected"])
m[2].metric("Missing helmets", summary["missing_helmets"])
m[3].metric("Missing vests", summary["missing_safety_vests"])

st.subheader("Export")
json_str = ann.to_json_string(annotation)
st.download_button(
    "⬇️ Download JSON", json_str,
    file_name=f"{uploaded.name}_annotation.json",
    mime="application/json",
)
with st.expander("Preview JSON"):
    st.code(json_str, language="json")
