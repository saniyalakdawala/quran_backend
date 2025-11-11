from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

# -----------------------------
# 1️⃣ Load Quran + Tafsir
# -----------------------------
def load_json(file_path):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"❌ File '{file_path}' not found!")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

quran_data = load_json("quran_with_tafsir.json")

# -----------------------------
# 2️⃣ Load precomputed embeddings
# -----------------------------
EMBEDDINGS_FILE = "quran_embeddings.npy"

if not Path(EMBEDDINGS_FILE).exists():
    raise FileNotFoundError(f"❌ Embeddings file '{EMBEDDINGS_FILE}' not found! Please precompute locally.")

print("🔹 Loading embeddings...")
embeddings = np.load(EMBEDDINGS_FILE)
print(f"✅ Loaded {len(embeddings)} embeddings.")

# FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"✅ Indexed {len(embeddings)} Quran verses.")

# Load embedding model (can be smaller, e.g., for query only)
embedder = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

# -----------------------------
# 3️⃣ Search function
# -----------------------------
def search_verses(query, top_k=5):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(np.array(q_emb, dtype="float32"), top_k)
    results = []
    for i in indices[0]:
        v = quran_data[i]
        if v.get("tafsir") and v["tafsir"].strip() != "" and "❌" not in v["tafsir"]:
            results.append(v)
    return results

# -----------------------------
# 4️⃣ Format output
# -----------------------------
def format_verse_output(verses):
    return [
        {
            "surah": v["surah"],
            "ayah": v["ayah"],
            "arabic": v["arabic"],
            "english": v["english"],
            "tafsir": v["tafsir"]
        }
        for v in verses
    ]

# -----------------------------
# 5️⃣ Flask app
# -----------------------------
app = Flask(__name__)
CORS(app)  # allow cross-origin requests from React

# store session context
session_context = {"last_verses": [], "last_index": 0}

@app.route("/query", methods=["POST"])
def query_quran():
    global session_context
    data = request.json
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # handle "next", "previous", "more"
    last_verses = session_context["last_verses"]
    last_index = session_context["last_index"]

    if last_verses:
        if "next" in query.lower():
            last_index += 1
            if last_index < len(last_verses):
                session_context["last_index"] = last_index
                return jsonify(format_verse_output([last_verses[last_index]]))
            else:
                return jsonify({"message": "No more verses"}), 200
        elif "previous" in query.lower():
            last_index -= 1
            if last_index >= 0:
                session_context["last_index"] = last_index
                return jsonify(format_verse_output([last_verses[last_index]]))
            else:
                session_context["last_index"] = 0
                return jsonify({"message": "Already at first verse"}), 200
        elif "more" in query.lower():
            next_verses = last_verses[last_index+1:last_index+6]
            if next_verses:
                session_context["last_index"] += len(next_verses)
                return jsonify(format_verse_output(next_verses))
            else:
                return jsonify({"message": "No more verses"}), 200

    # normal search
    results = search_verses(query)
    if results:
        session_context["last_verses"] = results
        session_context["last_index"] = 0
        return jsonify(format_verse_output(results[:5]))
    else:
        return jsonify({"message": "No verses found"}), 200

if __name__ == "__main__":
    # Render sets PORT env var automatically
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
