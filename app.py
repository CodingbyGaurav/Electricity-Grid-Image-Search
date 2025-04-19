import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
# import faiss # No longer needed
import numpy as np
import os
import io
from sklearn.metrics.pairwise import cosine_similarity 

# --- Configuration ---
MODEL_ID = "openai/clip-vit-base-patch32"
EMBEDDINGS_PATH = "embeddings.npy" # Path to NumPy embeddings
FILEPATHS_LIST_PATH = "filepaths.list"

# --- Caching Resources ---
@st.cache_resource
def load_model_and_processor():
    """Loads the CLIP model and processor."""
    print("Loading CLIP model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    try:
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        model = CLIPModel.from_pretrained(MODEL_ID).to(device)
        print("Model and processor loaded successfully.")
        return model, processor, device
    except Exception as e:
        st.error(f"Error loading model/processor: {e}")
        print(f"Error loading model/processor: {e}")
        return None, None, None

# Cache the embeddings and filepaths loading
@st.cache_resource
def load_embeddings_and_filepaths():
    """Loads the precomputed embeddings NumPy array and corresponding filepaths."""
    print("Loading embeddings and filepaths...")
    if not os.path.exists(EMBEDDINGS_PATH):
        st.error(f"Embeddings file not found at {EMBEDDINGS_PATH}. Please generate it using the notebook.")
        print(f"Embeddings file not found at {EMBEDDINGS_PATH}")
        return None, None
    if not os.path.exists(FILEPATHS_LIST_PATH):
        st.error(f"Filepaths list file not found at {FILEPATHS_LIST_PATH}")
        print(f"Filepaths list file not found at {FILEPATHS_LIST_PATH}")
        return None, None

    try:
        embeddings = np.load(EMBEDDINGS_PATH)
        with open(FILEPATHS_LIST_PATH, 'r') as f:
            filepaths = [line.strip() for line in f.readlines()]

        # Validation check
        if embeddings.shape[0] != len(filepaths):
            st.error(f"Mismatch between number of embeddings ({embeddings.shape[0]}) and filepaths ({len(filepaths)}).")
            return None, None

        print(f"Embeddings loaded (shape: {embeddings.shape}). {len(filepaths)} filepaths loaded.")
        return embeddings, filepaths
    except Exception as e:
        st.error(f"Error loading embeddings or filepaths: {e}")
        print(f"Error loading embeddings or filepaths: {e}")
        return None, None

# --- Core Search Function (using Cosine Similarity) ---
def find_similar_images_cosine(query_image, model, processor, device, dataset_embeddings, filepaths, top_k):
    """Computes embedding for the query image and finds similar images using cosine similarity."""
    if model is None or processor is None or dataset_embeddings is None or filepaths is None:
        st.error("Search components not loaded correctly.")
        return []

    try:
        # Process the query image to get its embedding
        inputs = processor(text=None, images=query_image, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            query_embedding = model.get_image_features(**inputs)
        query_embedding = query_embedding.cpu().numpy() # Keep as numpy array

        # Ensure query embedding is 2D for cosine_similarity
        if query_embedding.ndim == 1:
             query_embedding = np.expand_dims(query_embedding, axis=0)

        # Calculate cosine similarities
        # cosine_similarity returns shape (n_query_samples, n_dataset_samples)
        similarities = cosine_similarity(query_embedding, dataset_embeddings)[0] # We have 1 query, so take the first row

        # Get the indices of the top_k most similar images (excluding the query itself if it's in the dataset)
        # argsort returns indices from lowest to highest similarity, so we negate or reverse.
        # Using partition is slightly more efficient than full sort for finding top k
        # We add 1 to top_k because the image itself might be the most similar
        k_to_find = top_k + 1
        if k_to_find > len(similarities):
            k_to_find = len(similarities) # Handle cases with very small datasets

        # Get indices of top k similarities (highest values)
        # Partition gets the k-th largest element in place, elements >= k-th are to its right
        # indices = np.argpartition(similarities, -k_to_find)[-k_to_find:]
        # sorted_indices = indices[np.argsort(similarities[indices])[::-1]] # Sort just the top k

        # Simpler approach: Sort all similarities and take top K+1
        sorted_indices = np.argsort(similarities)[::-1] # Indices from highest to lowest similarity

        # Retrieve filepaths and distances (similarities)
        results = []
        print(f"Search Results (Top {top_k} by Cosine Similarity):")
        count = 0
        for idx in sorted_indices:
            # Optional: Try to skip the exact same image if paths match
            # This check might be fragile if paths differ slightly
            # if filepaths[idx] == query_image.filename: # Requires query_image to have filename attribute
            #      continue

            similarity = similarities[idx]
            filepath = filepaths[idx]
            # Convert similarity to distance? Optional. Cosine distance = 1 - cosine similarity
            distance = 1.0 - similarity
            results.append({"filepath": filepath, "distance": distance, "similarity": similarity})
            print(f"  - Rank {count+1}: Index={idx}, Sim={similarity:.4f}, Path={filepath}")
            count += 1
            if count >= top_k:
                 break # Stop after finding top_k results (potentially excluding self)

        return results

    except Exception as e:
        st.error(f"An error occurred during search: {e}")
        print(f"An error occurred during search: {e}")
        return []


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Grid Part Similarity Search")

# --- Sidebar --- 
st.sidebar.title("Settings")
st.sidebar.divider()

# Add slider for TOP_K
top_k_slider = st.sidebar.slider("Number of similar images to show:", min_value=1, max_value=15, value=5, step=1)

st.sidebar.divider()
st.sidebar.header("Info")
# Load resources early to display info
model, processor, device = load_model_and_processor()
embeddings, filepaths = load_embeddings_and_filepaths()
st.sidebar.markdown(f"**Model:** `{MODEL_ID}`")
st.sidebar.markdown(f"**Device:** `{device if device else 'N/A'}`")
if embeddings is not None:
    st.sidebar.markdown(f"**Indexed Images:** `{embeddings.shape[0]}`")
    st.sidebar.markdown(f"**Embedding Dim:** `{embeddings.shape[1]}`")
else:
    st.sidebar.markdown("**Indexed Images:** N/A")

# --- Main Page --- 
st.title("⚡ Electricity Grid Part - Image Similarity Search")
st.write("Upload an image of an electricity grid part to find similar images from the dataset using CLIP and Cosine Similarity.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.divider()

if uploaded_file is not None:
    query_image_bytes = uploaded_file.getvalue()
    try:
        query_image = Image.open(io.BytesIO(query_image_bytes)).convert("RGB")

        # --- Display Query Image First ---
        st.subheader("Your Query Image")
        # Use columns to center the query image slightly (optional, adjust width as needed)
        q_col1, q_col2, q_col3 = st.columns([1, 2, 1]) # Give center column more weight
        with q_col2:
             st.image(query_image, use_container_width=True)

        st.divider() # Add divider between query and results

        # --- Display Similar Images Below ---
        st.subheader(f"Top {top_k_slider} Similar Images")
        if embeddings is not None and filepaths is not None:
            with st.spinner(f"Searching for top {top_k_slider} similar images..."):
                similar_images = find_similar_images_cosine(query_image, model, processor, device, embeddings, filepaths, top_k=top_k_slider)

            if similar_images:
                # Display results in columns below the query image
                num_cols = min(top_k_slider, 5) # Max 5 columns for display
                cols = st.columns(num_cols)
                for i, result in enumerate(similar_images):
                    col_index = i % num_cols
                    with cols[col_index]:
                        img_path = result['filepath']
                        similarity = result['similarity']
                        if os.path.exists(img_path):
                            try:
                                result_image = Image.open(img_path)
                                # Format similarity as percentage
                                similarity_percent = similarity * 100
                                st.image(result_image,
                                         caption=f"**Similarity: {similarity_percent:.1f}%**\n{os.path.basename(img_path)}",
                                         use_container_width=True)
                            except Exception as img_e:
                                st.warning(f"Load failed: {os.path.basename(img_path)}", icon="⚠️")
                                print(f"Warning: Could not load result image: {img_path}. Error: {img_e}")
                        else:
                            st.warning(f"Not Found: {os.path.basename(img_path)}", icon="⚠️")
                            print(f"Warning: Result image path not found: {img_path}")
            else:
                st.info("No similar images found in the index.")
        else:
            st.error("Embeddings or filepaths could not be loaded. Cannot perform search.")

    except Exception as e:
        st.error(f"Error processing uploaded image: {e}")
        print(f"Error processing uploaded image: {e}")

else:
    st.info("Upload an image using the button above to start the similarity search.")
