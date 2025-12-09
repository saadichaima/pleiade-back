# Core/keywords.py
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
GPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def extract_keywords(text, max_keywords=8, with_synonyms=True):
    prompt = f"""
Tu es un assistant de bibliographie.
Donne {max_keywords} mots-clés (2 à 5 mots) en anglais/français, 1 par ligne.
Ajoute après un ';' 1-2 synonymes si utiles.
Texte:
\"\"\"{text[:3000]}\"\"\""""
    resp = client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[{"role":"system","content":"Assistant biblio concis."},
                  {"role":"user","content":prompt}],
        temperature=0.15, max_tokens=400
    )
    raw = (resp.choices[0].message.content or "").strip()
    kws = []
    seen = set()
    for line in raw.splitlines():
        line=line.strip()
        if not line: continue
        main = line.split(";")[0].strip()
        if len(main.split())>=2 and main.lower() not in seen:
            seen.add(main.lower()); kws.append(main)
    return kws[:max_keywords]
