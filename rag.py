import streamlit as st
import json
from google.genai.errors import APIError
from sklearn.metrics.pairwise import cosine_similarity
from google.genai import types
import numpy as np
import pandas as pd

from helper import translate_test_types, join_description
from config import EMBEDDING_MODEL

def extract_constraints_with_gemini(client, query: str):
    """Uses Gemini to extract structured constraints."""
    
    prompt = f"""
    You are an AI that extracts hiring-related information from text.

    Given the following query or job description, extract and return structured information as a JSON object.
    Make sure your response is a valid JSON — no explanations or extra text. STRICTLY FOLLOE ALL THE RULES.

    JSON format:
    {{
      "description": {{
        "job_level": [],
        "technical_skills": [],
        "test_types": [],
        "job_family": [],
        "experience": "",
        "industry": [],
        "language": ""
      }},
      "test_duration": 0
    }}

    Extraction Rules:
    - "job_level": choose atleast one (the most relevant) [Director, Entry-Level, Graduate, Manager, Mid-Professional, Supervisor, Executive, Frontline Manager, General Population, Professional Individual Contributor].
    - "technical_skills": choose atleast or can add skill not listed here(the most relevant) (Python, SQL, Excel, JavaScript, Web Development, Engineering Fields).
    - "soft_skills": choose at least 2–3 from ['Communication', 'Teamwork', 'Problem Solving', 'Adaptability', 'Critical Thinking', 'Leadership', 'Time Management', 'Creativity', 'Conflict Resolution'] or add other most relevant ones if not listed here.
    - "test_types": choose atleast one (the most relevant):
      ['Ability & Aptitude', 'Biodata & Situational Judgement', 'Competencies',
       'Development & 360', 'Assessment Exercise', 'Knowledge & Skills',
       'Personality & Behaviour', 'Simulation'].
    - "job_family": choose atleast one (the most relevant):
      ['Business', 'Clerical', 'Contact Center', 'Customer Service',
       'Information Technology', 'Safety', 'Sales'].
    - "experience": extract numeric or qualitative experience terms ('2-5 years', 'Entry-level', 'Senior').
    - "industry": choose atleast one (the most relevant) from:
      ['Banking/Finance', 'Healthcare', 'Hospitality', 'Insurance',
       'Manufacturing', 'Oil & Gas', 'Retail', 'Telecommunications'].
    - "language": Specify the language preference mentioned in the query. If no language is explicitly stated, infer it from the country mentioned (use the country's national language). If neither is provided, default to English.
    - "test_duration": extract total time in minutes.
        • If duration is in range (e.g., 25–35), pick the higher value.
        • If no number is mentioned, return -1.

    QUERY: {query}
    """
    
    # Note: Streamlit's UI will handle the spinner, but a temporary message is useful
    with st.spinner("Extracting constraints from the job description..."):
        try:
            # Use the cached client
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)

        except APIError as e:
            st.error(f"⚠️ Gemini API Error during constraint extraction: {e}")
        except json.JSONDecodeError as e:
            st.error(f"⚠️ JSON Decode Error from Gemini: {e}")

        # Return a fallback in case of errors
        return {"description": {"job_level": [], "technical_skills": [], "test_types": [], "job_family": [], "experience": "", "industry": [], "language": ""}, "test_duration": 0}

def recommend_assessments(client, query: str, df: pd.DataFrame, embeddings: np.ndarray, top_k_retrieval: int = 20, final_k: int = 10):
    """Performs hybrid recommendation: Vector Search + LLM Constraint Filtering."""
    
    if embeddings is None or len(embeddings) == 0:
        return []

    # 1. LLM Constraint Extraction
    constraints = extract_constraints_with_gemini(client, query)
    llm_desc = join_description(constraints)
    print(constraints)

    # 2. Vector Search (Retrieval)
    with st.spinner("Performing semantic search..."):
        query_embedding_result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[llm_desc],
            # contents=[query],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_embedding = np.array(query_embedding_result.embeddings[0].values)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
        
        # Top-k retrieval
        top_indices = np.argsort(similarities)[::-1][:top_k_retrieval]
        candidates_df = df.iloc[top_indices].copy()
        candidates_df['similarity_score'] = similarities[top_indices]

    # 3. Constraint Filtering (Re-ranking)
    filtered_df = candidates_df.copy()

    # Filter 1: Max Duration
    if constraints.get('test_duration', 0) is not None and constraints['test_duration'] > 0:
        filtered_df = filtered_df[filtered_df['test_duration'] <= constraints['test_duration']]

    # Sort by similarity and select final
    filtered_df = filtered_df.sort_values(by='similarity_score', ascending=False)
    final_recommendations = filtered_df.head(final_k)

    # Format output for Streamlit DataFrame display
    results = []
    for _, row in final_recommendations.iterrows():
        # Get the full list of test types from the original dataframe and format it
        if row['test_duration'] == -1:
            row['test_duration'] = "Untimed"
        test_types_list = translate_test_types(row.get('test_type', []))
        test_types_str = ", ".join(test_types_list) if test_types_list else ""

        results.append({
            'Assessment Name': row.get('name', ''),
            'Description Snippet': row.get('description', '')[:100] + '...',
            'Duration (min)': row.get('test_duration', ''),
            'Test Type': test_types_str,
            'URL': row.get('url', ''),
            'Similarity Score': f"{row['similarity_score']:.4f}"
        })

    return results

