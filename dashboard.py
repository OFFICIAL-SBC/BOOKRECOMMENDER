from typing import List, Optional
from sqlalchemy import create_engine, text
import os
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma


# ------------------------------------------------------------------ #
# 1⃣  DATA & VECTOR DB SET‑UP
# ------------------------------------------------------------------ #

load_dotenv()

# ----------  BOOKS CSV  ----------
df_books = pd.read_csv("books_with_emotions.csv")

df_books["large_thumbnail"] = (
    df_books["thumbnail"].fillna("assets/cover-not-found.jpg") + "&fife=w800"
)


# ----------  TAGGED TEXT ➜ VECTOR DB  ----------

DATA_DIR = Path(__file__).parent  # folder where dashboard.py lives

raw_documents = TextLoader(
    DATA_DIR / "tagged_description.txt",
    encoding="utf-8-sig"
).load()

# text_splitter = CharacterTextSplitter(
#     separator="\n", chunk_size=0, chunk_overlap=0)

# documents = text_splitter.split_documents(raw_documents)

# vector_db_books = Chroma.from_documents(
#     documents=documents,
#     embedding=OpenAIEmbeddings(),
#     persist_directory="./chroma_db_books"
# )


# ! Read created vector database
vector_db_books = Chroma(
    persist_directory="chroma_db_books",
    embedding_function=OpenAIEmbeddings()
)


# ! Now we are gonna create a function that is going retrieve those semantic recommendations
# ! from out book dataset and it is also going to apply filtering by category and sorting based on emotional tone.

def retrieve_semantic_recommendations(
    query: str,
    category: str = "All",
    tone: str | None = None,
    initial_top_k: int = 50,
    final_top_k: int = 16
) -> pd.DataFrame:

    recs = vector_db_books.similarity_search(
        query=query,
        k=initial_top_k
    )

    recommended_books_isbn = [
        str(rec.page_content.strip('"').split(" ")[0]) for rec in recs]

    # recommended_books = df_books[df_books["isbn13"].isin(
    #     recommended_books_isbn)].head(initial_top_k)

    recommended_books = get_books_by_isbn13(recommended_books_isbn)

    if category != "All":
        recommended_books = recommended_books[recommended_books["simple_categories"]
                                              == category][:final_top_k]
    else:
        recommended_books = recommended_books[:final_top_k]

    if tone == "Happy":
        recommended_books.sort_values(by="joy", ascending=False, inplace=True)
    if tone == "Surprising":
        recommended_books.sort_values(
            by="surprise", ascending=False, inplace=True)
    if tone == "Angry":
        recommended_books.sort_values(
            by="anger", ascending=False, inplace=True)
    if tone == "Suspenseful":
        recommended_books.sort_values(by="fear", ascending=False, inplace=True)
    if tone == "Sad":
        recommended_books.sort_values(
            by="sadness", ascending=False, inplace=True)

    return recommended_books


def get_books_by_isbn13(isbn13_list: List[str]) -> pd.DataFrame:

    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    schema = os.getenv("DB_SCHEMA")

    required_vars = [user, password, host, port, schema]
    if not all(required_vars):
        missing_vars = []
        if not user:
            missing_vars.append("DB_USER")
        if not password:
            missing_vars.append("DB_PASSWORD")
        if not host:
            missing_vars.append("DB_HOST")
        if not port:
            missing_vars.append("DB_PORT")
        if not schema:
            missing_vars.append("DB_SCHEMA")

        raise ValueError(
            f"The following environment variables are missing: {', '.join(missing_vars)}")

    table = "books_with_emotions"

    if not isbn13_list or not isinstance(isbn13_list, list):
        return pd.DataFrame()

    isbn13_list = [isbn for isbn in isbn13_list if isbn and str(isbn).strip()]

    if not isbn13_list:
        return pd.DataFrame()

    try:
        engine = create_engine(
            f"mysql+pymysql://{user}:{password}@{host}:{port}/{schema}",
            pool_pre_ping=True,
            pool_recycle=3600
        )
    except Exception as e:
        print(f"Error creating database engine: {e}")
        return pd.DataFrame()

    placeholders = ", ".join([":isbn" + str(i)
                             for i in range(len(isbn13_list))])

    params = {f"isbn{i}": isbn for i, isbn in enumerate(isbn13_list)}

    query = f"SELECT * FROM {table} WHERE isbn13 IN ({placeholders})"

    conn = None
    try:
        conn = engine.connect()
        result = pd.read_sql_query(text(query), conn, params=params)
        return result

    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()

    finally:
        if conn:
            conn.close()
        engine.dispose()  # Clean up the connection pool


def recommend_books(
    query: str,
    category: str,
    tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    result = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split(" ")
        truncated_desc = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{", ".join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row["title"]} by {authors_str}: {truncated_desc}"
        result.append((row["large_thumbnail"], caption))

    return result


# ------------------------------------------------------------------
# 3⃣  STREAMLIT INTERFACE
# ------------------------------------------------------------------
ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="📚 Semantic Book Recommender", layout="wide")

# ----------  ROW 1 :  logo (col‑1)  +  title spanning col‑2 & col‑3 -------
col_logo, col_title1 = st.columns([2, 4], gap="small")

with col_logo:
    st.image(ASSETS / "mom.png", width=400)            # <-- your image

with col_title1:

    st.markdown(
        "<h1 style='margin-top: 200px; line-height:1.2; margin-bottom:0'>👋🏻 Hi!! 😊 My name is Gloria, your personal semantic book recommender. 📚</h1>",
        unsafe_allow_html=True,
    )
# col_title2 intentionally left blank so the title visually spans both columns

st.markdown(" ")  # tiny vertical spacer between rows

# ----------  ROW 2 :  prompt + filters + button --------------------------
prompt_col, cat_col, tone_col, btn_col = st.columns([6, 2, 2, 1], gap="medium")

with prompt_col:
    user_query = st.text_input(
        "Describe what you feel like reading",
        placeholder="e.g. witty coming‑of‑age fantasy, Scandinavian noir…",
        key="query_input",
    )

with cat_col:
    category = st.selectbox(
        "Category",
        ["All"] + sorted(df_books["simple_categories"].unique()),
        index=0,
    )

with tone_col:
    tone = st.selectbox(
        "Tone", ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"], index=0,
    )

with btn_col:
    st.markdown(" ")
    st.markdown(" ")
    go = st.button("Find books", use_container_width=True, type="primary")

# ----------  RESULTS GRID -----------------------------------------------
if go:
    if not user_query.strip():
        st.warning("Please enter a description before searching.")
        st.stop()

    with st.spinner("Fetching recommendations…"):
        cards = recommend_books(user_query, category, tone)

    if not cards:
        st.error("No recommendations found. Try a different query or filters.")
    else:
        st.success(f"Showing {len(cards)} recommendation(s)")
        n_cols = 4  # 4‑column responsive grid
        for row_start in range(0, len(cards), n_cols):
            cols = st.columns(n_cols)
            for col, (img, caption) in zip(cols, cards[row_start: row_start + n_cols]):
                with col:
                    st.image(img, use_container_width=True, caption=caption)
