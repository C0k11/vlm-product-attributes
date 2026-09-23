"""Streamlit front end for the attribute extraction API.

    streamlit run demo/app.py -- --api http://localhost:8000
"""
import argparse
import json

import requests
import streamlit as st

parser = argparse.ArgumentParser()
parser.add_argument("--api", default="http://localhost:8000")
args, _ = parser.parse_known_args()

st.set_page_config(page_title="Product attribute extraction", layout="centered")
st.title("Product attribute extraction")
st.caption("Qwen3.5-4B + LoRA, trained on Amazon Berkeley Objects. Returns product type, color and material as JSON.")

uploaded = st.file_uploader("Product image", type=["jpg", "jpeg", "png", "webp"])
title = st.text_input("Product title (optional)")

if uploaded is not None:
    st.image(uploaded, width=320)
    if st.button("Extract attributes", type="primary"):
        files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")}
        data = {"title": title} if title.strip() else {}
        try:
            resp = requests.post(f"{args.api}/extract", files=files, data=data, timeout=120)
        except requests.RequestException as exc:
            st.error(f"API not reachable at {args.api}: {exc}")
        else:
            if resp.ok:
                body = resp.json()
                cols = st.columns(3)
                for col, key in zip(cols, ["product_type", "color", "material"]):
                    col.metric(key, body["attributes"][key])
                st.code(json.dumps(body["attributes"], indent=2), language="json")
                note = "image + title adapter" if body["used_title"] else "image-only adapter"
                st.caption(f"{note}, {body['latency_ms']} ms")
            else:
                st.error(f"{resp.status_code}: {resp.text}")
