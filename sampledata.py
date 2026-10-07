import base64, io
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

SAMPLES = [
    {"sample_id":"sample-glioma-01","label":"Glioma"},
    {"sample_id":"sample-meningioma-02","label":"Meningioma"},
    {"sample_id":"sample-pituitary-03","label":"Pituitary"},
    {"sample_id":"sample-normal-04","label":"No Tumor"},
    {"sample_id":"sample-glioma-05","label":"Glioma"},
    {"sample_id":"sample-normal-06","label":"No Tumor"},
]

def _make_image(label, seed):
    rng = np.random.default_rng(seed)
    size = 224
    a = rng.normal(35, 10, (size,size)).clip(0,255).astype(np.uint8)
    yy, xx = np.ogrid[:size,:size]
    brain = ((xx-112)**2/88**2 + (yy-112)**2/98**2) <= 1
    a[brain] += 45
    if label != "No Tumor":
        cx = int(rng.integers(70,155))
        cy = int(rng.integers(65,155))
        r = int(rng.integers(12,28))
        tumor = (xx-cx)**2 + (yy-cy)**2 <= r*r
        a[tumor] = np.clip(a[tumor] + 100, 0, 255)
    a = np.uint8(np.clip(a,0,255))
    rgb = np.stack([a,a,a], axis=-1)
    return Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(1))

def _data_url(im):
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def get_sample_mri_by_id(sample_id):
    for i, s in enumerate(SAMPLES):
        if s["sample_id"] == sample_id:
            im = _make_image(s["label"], i+10)
            return {**s, "image_url": _data_url(im), "image_size": [224,224],
                    "synthetic": True}
    return None

def get_all_samples_manifest():
    return [get_sample_mri_by_id(s["sample_id"]) for s in SAMPLES]
