#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 22:10:08 2026

@author: jamal
"""

import streamlit as st
from ultralytics import YOLO
from collections import Counter
from PIL import Image

st.title("Gewindefehler-Erkennung")
st.write("Lade eine technische Zeichnung hoch, um sie automatisch auf Gewindefehler zu prüfen.")

modell_pfad = "/Users/jamal/Desktop/thread_ai/runs/detect/train-21/weights/best.pt"

klassen_zu_fehlercode = {
    "thread": "E08",
    "dimensions": "E05",
    "top_view": "E09",
    "standard": "kein Fehler"
}

@st.cache_resource
def lade_modell():
    return YOLO(modell_pfad)

model = lade_modell()

hochgeladene_datei = st.file_uploader("Zeichnung auswählen", type=["png", "jpg", "jpeg"])

if hochgeladene_datei is not None:
    bild = Image.open(hochgeladene_datei)
    st.image(bild, caption="Hochgeladene Zeichnung", use_container_width=True)

    if st.button("Zeichnung prüfen"):
        with st.spinner("Analysiere..."):
            results = model.predict(bild, imgsz=1280, conf=0.25, verbose=False)
            result = results[0]

            klassen_ids = result.boxes.cls.tolist()
            klassen_namen = [model.names[int(cls_id)] for cls_id in klassen_ids]
            haeufigkeiten = Counter(klassen_namen)

        st.subheader("Ergebnis")
        if not haeufigkeiten:
            st.success("Keine Fehler gefunden.")
        else:
            for klasse, anzahl in haeufigkeiten.items():
                fehlercode = klassen_zu_fehlercode[klasse]
                st.write(f"**{fehlercode}**: {anzahl}x gefunden")

        st.subheader("Erkennungen im Bild")
        annotiertes_bild = result.plot()
        st.image(annotiertes_bild, caption="Erkannte Fehler markiert", use_container_width=True)