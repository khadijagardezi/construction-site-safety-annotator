# Construction-Site Safety Annotator

A small prototype I built to demonstrate a construction-site image labeling
workflow which auto-detects people, vehicles and PPE, lets you correct them by hand, tag safety risks, and export structured JSON.

Detection uses a YOLOv8 model trained on construction-site-safety data, so helmets and vests are detected automatically; risk events are tagged manually. The point is to show the annotation workflow, not to be a finished safety system.

## What it does

- Upload a site image
- Auto-detect people, vehicles/machinery and PPE (helmet, vest) in one pass
- Review each object and fix its category or PPE
- Tag risk events + severity, add notes
- Export JSON / CSV and see a quick summary

## Run it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```