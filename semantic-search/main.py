import google.generativeai as genai
import numpy as np
import pandas as pd

# Configure the Gemini API key
## use your own api key - genai.configure(api_key='')

def embed_fn2(text):
    """Generate embeddings for a given text."""
    return genai.embed_content(model="models/text-embedding-004",
                               content=text,
                               task_type="retrieval_document",
                               title="Construction Activities")["embedding"]

def save_embeddings(dataframe, filepath):
    """Generate embeddings and save them as an additional column in the Excel file."""
    # Generate embeddings for each row in the DataFrame
    dataframe['Embeddings'] = dataframe['Concat'].apply(embed_fn2)
    # Save the DataFrame with embeddings to an Excel file
    dataframe.to_excel(filepath, index=False)
    print("Embeddings saved to", filepath)

def find_best_passage2(query, dataframe, top_n=3):
    """
    Compute the distances between the query and each document in the dataframe
    using the cosine similarity and return the top N descriptions with their similarity percentages.
    """
    # Embed the query using the specified model
    query_embedding = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query"
    )["embedding"]

    # Convert embeddings to numpy arrays
    query_embedding = np.array(query_embedding)
    embeddings = np.stack(dataframe['Embeddings'].apply(eval).apply(np.array).values)

    # Compute cosine similarities
    similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))

    # Get the top N matches
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_descriptions = dataframe.iloc[top_indices]['Description']
    top_similarities = similarities[top_indices]

    # Convert similarities to percentages
    top_percentages = top_similarities * 100

    # Prepare the results
    results = list(zip(top_descriptions, top_percentages))
    return results

if __name__ == "__main__":
    session_on = True
    # Load the dataset
    filepath = 'datasets-usefull/embeddings.xlsx'
    df = pd.read_excel(filepath)

    # Check if 'Embeddings' column exists, if not generate and save embeddings
    if 'Embeddings' not in df.columns:
        print("Creating embeddings ...")
        save_embeddings(df, filepath)
        print("Embeddings created and saved!")

    # Main loop for querying
    while session_on:
        prompt = input("Enter the query: ")
        if prompt == "QUIT":
            session_on = False
        else:
            passage = find_best_passage2(prompt, df, top_n=3)
            for description, similarity in passage:
                print(f"Description: {description}\nSimilarity: {similarity:.2f}%\n")
