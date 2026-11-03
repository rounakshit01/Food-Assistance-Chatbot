import csv
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import download_hugging_face_embeddings

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME = "restaurant-menu"
GUIDELINES_NAMESPACE = "guidelines"


SEASONING_CODES = {"H013", "E064"} 

def is_seasoning(food):
    return food["code"].startswith("G") or food["code"] in SEASONING_CODES


def short_name(food):
    return food["name"].split("(")[0].strip()


def highlights(foods):
    rules = [("fibre_g", 5, "high in fibre"), ("iron_mg", 4, "rich in iron"), ("calcium_mg", 150, "rich in calcium"),
             ("protein_g", 15, "high in protein"), ("potassium_mg", 600, "rich in potassium")]
    parts = []
    for col, limit, label in rules:
        names = [short_name(f) for f in foods if f[col] not in ("", None) and float(f[col]) >= limit]
        if names:
            parts.append(f"{label}: {', '.join(names)}")
    return "; ".join(parts)


def nutrition_text(food):
    parts = []
    for col, label, unit in [("kcal", "energy", " kcal"), ("protein_g", "protein", " g"), ("fat_g", "fat", " g"),
                             ("fibre_g", "fibre", " g"), ("carb_g", "carbohydrate", " g"),
                             ("calcium_mg", "calcium", " mg"), ("iron_mg", "iron", " mg"),
                             ("sodium_mg", "sodium", " mg"), ("potassium_mg", "potassium", " mg")]:
        if food[col] not in ("", None):
            parts.append(f"{label} {float(food[col]):g}{unit}")
    return f"{short_name(food)}: " + ", ".join(parts)


def load_menu():
    
    with open("data/ifct_foods.csv", newline="", encoding="utf-8") as f:
        ifct = {r["code"]: r for r in csv.DictReader(f)}
    with open("data/dish_ingredients.csv", newline="", encoding="utf-8") as f:
        codes = {r["name"]: r["ifct_codes"].split(";") for r in csv.DictReader(f)}

    docs, ids = [], []
    with open("data/menu.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            foods = [ifct[c] for c in codes.get(row["name"], []) if c in ifct]
            foods = [f for f in foods if not is_seasoning(f)] or foods   
            text = (f"Dish: {row['name']}. Region: {row['state']}. {row['description']}. "
                    f"Ingredients: {row['ingredients']}. Type: {row['type']}. "
                    f"Protein level: {row['protein']}. Price: Rs {row['price']}. "
                    f"Ready in {row['prep_minutes']} minutes.")
            if facts:
                text += " Nutrition of main ingredients, raw, per 100 g (IFCT 2017): " + " | ".join(facts) + "."
                if highlights(foods):
                    text += " Ingredient highlights: " + highlights(foods) + "."
            docs.append(Document(page_content=text,
                                 metadata={"name": row["name"], "price": int(row["price"]), "state": row["state"]}))
            ids.append(row["name"].lower().replace(" ", "-"))
    return docs, ids


def load_guidelines():
    
    docs, ids = [], []
    with open("data/dgi_chunks.csv", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            docs.append(Document(
                page_content=f"[ICMR-NIN Dietary Guidelines for Indians 2024, PDF page {row['page']}] {row['text']}",
                metadata={"page": int(row["page"]), "source": "DGI 2024"}))
            ids.append(f"dgi-{i}")
    return docs, ids


embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,    
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

menu_docs, menu_ids = load_menu()
PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings).add_documents(menu_docs, ids=menu_ids)
print(f"Stored {len(menu_docs)} dishes")

guide_docs, guide_ids = load_guidelines()
PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings,
                    namespace=GUIDELINES_NAMESPACE).add_documents(guide_docs, ids=guide_ids)
print(f"Stored {len(guide_docs)} guideline chunks in namespace '{GUIDELINES_NAMESPACE}'")
