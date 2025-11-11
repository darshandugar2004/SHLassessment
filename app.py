import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

from rag import recommend_assessments
from embeddings_cache import get_gemini_client, load_data_and_embeddings

# --- Streamlit Caching Functions ---

llm_desc = ""

# --- Streamlit Application Layout ---

def main():
    st.set_page_config(
        page_title="SHL Assessment Recommendation System", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("💡 SHL Assessment Recommendation System")
    st.markdown("Enter a Job Description or natural language query below to receive recommended assessments.")
    
    # 0. Initialize Resources
    client = get_gemini_client()
    with st.spinner("Loading Assessment Data and Computing Embeddings... (This runs once)"):
        assessment_df, assessment_embeddings = load_data_and_embeddings(_client = client)

    if assessment_df.empty or assessment_embeddings is None:
        st.error("Cannot run recommendations because data or embeddings failed to load.")
        return

    # 1. User Input
    user_query = st.text_area(
        "Job Description or Query:",
        placeholder=f"Write your Query here", # Use the example query from your script
        height=300
    )

    # 2. Trigger Button
# Single button
    if st.button("Get Assessment Recommendations", type="primary", key="recommend_button"):

        if user_query.strip():
            # Run the RAG pipeline
            recommended_results = recommend_assessments(
                client=client,
                query=user_query,
                df=assessment_df,
                embeddings=assessment_embeddings,
                final_k=10
            )
    
            if recommended_results:
                st.subheader(f"Top {len(recommended_results)} Recommended Assessments")
                results_df = pd.DataFrame(recommended_results)
    
                # Format as a table with links
                st.dataframe(
                    results_df,
                    column_config={
                        "URL": st.column_config.LinkColumn("URL", display_text="Open Assessment Link")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                st.success("Recommendations complete!")
            else:
                st.warning("No relevant assessments found based on your query and filters.")
        else:
            st.warning("Please enter a job description or query to get recommendations.")

if __name__ == "__main__":
    main()