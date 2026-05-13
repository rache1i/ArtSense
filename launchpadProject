import os, torch, skimage, requests, pinecone, glob, IPython.display, io
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
# //get images from file

def load_my_images(root_dir):
    data = []
    for artist in os.listdir(root_dir):
        artist_path = os.path.join(root_dir, artist)
        if not os.path.isdir(artist_path):
            continue
        for img_file in glob.glob(os.path.join(artist_path, "*")):
            try:
                image = Image.open(img_file).convert("RGB")
                data.append({"artist": artist, "image_path": img_file, "image": image})
            except Exception as e:
                print(f"Skipping {img_file}: {e}")

    return pd.DataFrame(data)
        

root_dir = "/Users/rachelli/launchpad/venv/artists"
print("Loading dataset...")
image_data_df = load_my_images(root_dir)
print(f"Loaded {len(image_data_df)} images from {image_data_df['artist'].nunique()} artists.")


# load clip model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID).to(device)
processor = CLIPProcessor.from_pretrained(model_ID)


# Attribute list for similarity explanation

attributes = [
    "bright colors",
    "muted colors",
    "soft brushstrokes",
    "strong texture",
    "geometric shapes",
    "abstract layout",
    "realistic style",
    "warm color palette",
    "cool color palette",
    "high contrast",
    "low contrast",
    "symmetry",
]

# use clip to get embeddings for the images

def get_single_image_embedding(my_image):
    inputs = processor(images=my_image, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        embedding = model.get_image_features(inputs)
    return embedding.cpu().numpy()

def get_all_images_embedding(df, img_column):
    print("Computing embeddings for all images...")
    df["img_embeddings"] = df[img_column].apply(get_single_image_embedding)
    return df

image_data_df = get_all_images_embedding(image_data_df, "image")

# use cos similarity to find how similar images are

def get_top_N_images(query_image, data, top_K=15):
    query_vect = get_single_image_embedding(query_image)
    data["cos_sim"] = data["img_embeddings"].apply(
        lambda x: cosine_similarity(query_vect, x)[0][0]
    )
    most_similar = data.sort_values(by="cos_sim", ascending=False).head(top_K)
    return most_similar[["artist", "image", "image_path", "cos_sim"]].reset_index(drop=True)

def extract_shared_attributes(query_image, match_image):
    """Returns top 3 visual attributes shared between two images."""
    texts = processor(
        text=attributes, return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        text_emb = model.get_text_features(**texts)

        img1 = processor(images=query_image, return_tensors="pt")["pixel_values"].to(device)
        img2 = processor(images=match_image, return_tensors="pt")["pixel_values"].to(device)

        img_emb_1 = model.get_image_features(img1)
        img_emb_2 = model.get_image_features(img2)

    text_emb  = text_emb / text_emb.norm(dim=-1, keepdim=True)
    img_emb_1 = img_emb_1 / img_emb_1.norm(dim=-1, keepdim=True)
    img_emb_2 = img_emb_2 / img_emb_2.norm(dim=-1, keepdim=True)

    sim1 = (img_emb_1 @ text_emb.T).cpu().numpy()[0]
    sim2 = (img_emb_2 @ text_emb.T).cpu().numpy()[0]

    joint = sim1 + sim2

    idxs = joint.argsort()[::-1][:3]
    return [attributes[i] for i in idxs]

# display images
def plot_images_by_side(top_images):
    n = len(top_images)
    n_row = n_col = int(np.ceil(np.sqrt(n)))
    _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
    axs = axs.flatten()
    for idx, row in top_images.iterrows():
        axs[idx].imshow(row.image)
        axs[idx].axis("off")
        axs[idx].set_title(f"Sim: {100*row.cos_sim:.2f}%\n{row.artist[:50]}...")
    plt.tight_layout()
    plt.show()

# show similarity search results to query image
print("Running similarity search...")
# setup FastAPI app

app = FastAPI()

app.mount(
    "/images", 
    StaticFiles(directory="/Users/rachelli/launchpad/venv/artists"),
    name="images"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    print("This is the file running")
    
    
    contents = await file.read()
    query_image = Image.open(io.BytesIO(contents)).convert("RGB")
    top_matches = get_top_N_images(query_image, image_data_df, top_K=15)
    top_row = top_matches.iloc[0]
    shared = extract_shared_attributes(query_image, top_row.image)
    results = []
    for _, row in top_matches.iterrows():
        full_path = row["image_path"]

        relative_path = os.path.relpath(full_path, root_dir)
        relative_path = relative_path.replace("\\", "/")

        while relative_path.startswith(("Users/", "../", "./", "/")):
            relative_path = relative_path.split("/", 1)[1]


        public_url = f"/images/{relative_path}"

        results.append({
            "artist": row.artist,
            "similarity": float(row.cos_sim),
            "image_path": public_url
        })

    return {
        "results": results,
        "shared_attributes": shared
    }

    return {"results": results}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
