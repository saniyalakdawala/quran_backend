from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# -----------------------------
# 1️⃣ Load Quran + Tafsir
# -----------------------------
def load_json(file_path):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"❌ File '{file_path}' not found!")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

quran_data = load_json("quran_with_tafsir.json")

# Combine English + Tafsir for embeddings
texts = [f"{v['english']} {v['tafsir']}" for v in quran_data]

# -----------------------------
# 2️⃣ Embeddings + FAISS index
# -----------------------------
print("🔹 Creating embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.array(embedder.encode(texts, show_progress_bar=True), dtype="float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"✅ Indexed {len(embeddings)} Quran verses.")

# -----------------------------
# 3️⃣ Search function
# -----------------------------
def search_verses(query, top_k=5):
    q_emb = embedder.encode([query])
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
    output_list = []
    for v in verses:
        output_list.append({
            "surah": v["surah"],
            "ayah": v["ayah"],
            "arabic": v["arabic"],
            "english": v["english"],
            "tafsir": v["tafsir"]
        })
    return output_list

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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

