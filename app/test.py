# app/test.py
from app.services.prompts import fetch_cii, fetch_cir, fetch_other
print(fetch_other("vision_describe_images.txt")[:120])
