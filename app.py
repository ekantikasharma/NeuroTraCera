import os
import sys

# Ensure repository root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

STREAMLIT_DIR = os.path.join(ROOT_DIR, "streamlit")
if STREAMLIT_DIR not in sys.path:
    sys.path.insert(0, STREAMLIT_DIR)

# Execute the primary Streamlit application located in streamlit/app.py
target_script = os.path.join(STREAMLIT_DIR, "app.py")
with open(target_script, "r", encoding="utf-8") as f:
    app_code = compile(f.read(), target_script, "exec")
    exec(app_code, {
        "__file__": target_script,
        "__name__": "__main__",
        "__doc__": None,
        "__package__": None
    })
