from config import CATALOG_PATH, EMBEDDING_MODEL
import streamlit as st
from google import genai
import json
import pandas as pd
import numpy as np
from google.genai import types

from helper import extract_duration, safe_join

@st.cache_resource
def get_gemini_client():
    """Initializes and caches the Gemini client."""
    try:
        # Fetch key from Streamlit secrets
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)
        return client
    except KeyError:
        st.error("🚨 Gemini API Key not found. Please set `GEMINI_API_KEY` in `.streamlit/secrets.toml`.")
        st.stop()
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        st.stop()

@st.cache_resource
def load_data_and_embeddings(_client):
    """Loads data, generates source text, and computes embeddings (expensive operation)."""
    client = _client
    # 1. Load Data
    try:
        with open(CATALOG_PATH, 'r',  encoding='utf-8') as f:
            catalog_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {CATALOG_PATH}. Please ensure it is uploaded.")
        return pd.DataFrame(), None
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {CATALOG_PATH}.")
        return pd.DataFrame(), None

    assessment_df = pd.DataFrame.from_dict(catalog_data, orient='index').reset_index()

    # 2. Prepare Source Text
    assessment_df['source_text'] = (
        "Assessment Name: " + assessment_df['name'] + ". " +
        "Description: " + assessment_df['description'] + ". " +
        "Languages: " + assessment_df['language'].apply(lambda x: safe_join(x)) + ". " +
        "Job Levels: " + assessment_df['joblevel'].apply(lambda x: safe_join(x)) + ". " +
        "Test Types: " + assessment_df['test_type'].apply(lambda x: safe_join(x, is_test_type=True))
    )
    assessment_df["test_duration"] = assessment_df["assessment_length"].apply(extract_duration)
    
    st.write(f"✅ Catalog loaded with {len(assessment_df)} assessments.")

    # 3. Generate Embeddings
    texts = assessment_df['source_text'].tolist()
    embeddings = []
    
    # Use a progress bar for the expensive embedding step
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, text in enumerate(texts):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            embeddings.append(response.embeddings[0].values)
            progress_bar.progress((i + 1) / len(texts))
            status_text.text(f"Generating embeddings: {i+1}/{len(texts)}")
        except Exception as e:
            st.warning(f"Embedding error for index {i}: {e}. Skipping this entry.")
            # Append NaN or empty vector if embedding fails
            embeddings.append(np.full(768, np.nan)) # 768 is a typical embedding dimension

    progress_bar.empty()
    status_text.empty()
    
    final_embeddings = np.array(embeddings)
    # Remove rows where embedding generation failed
    valid_indices = ~np.isnan(final_embeddings).all(axis=1)
    
    return assessment_df[valid_indices], final_embeddings[valid_indices]