from dotenv import load_dotenv
import os
import re

from flask import Flask, render_template, request
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

GEMINI_MODEL = "gemini-3.5-flash-lite"
INDEX_NAME = "restaurant-menu"
GUIDELINES_NAMESPACE = "guidelines"

embeddings = download_hugging_face_embeddings()


menu_store = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
guide_store = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings,
                                                      namespace=GUIDELINES_NAMESPACE)


STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]
NICKNAMES = { 
    "bengal": "West Bengal", "bengali": "West Bengal", "gujarati": "Gujarat", "punjabi": "Punjab",
    "maharashtrian": "Maharashtra", "kashmir": "Jammu and Kashmir", "kashmiri": "Jammu and Kashmir",
    "j&k": "Jammu and Kashmir", "goan": "Goa", "malayali": "Kerala", "tamil": "Tamil Nadu", "tamilnadu": "Tamil Nadu",
    "assamese": "Assam", "odia": "Odisha", "orissa": "Odisha", "rajasthani": "Rajasthan", "bihari": "Bihar",
    "naga": "Nagaland", "manipuri": "Manipur", "mizo": "Mizoram", "khasi": "Meghalaya", "sikkimese": "Sikkim",
    "andaman": "Andaman and Nicobar Islands", "nicobar": "Andaman and Nicobar Islands",
    "daman": "Dadra and Nagar Haveli and Daman and Diu", "diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra": "Dadra and Nagar Haveli and Daman and Diu", "hyderabadi": "Telangana", "pondicherry": "Puducherry",
    "himachali": "Himachal Pradesh", "haryanvi": "Haryana", "chhattisgarhi": "Chhattisgarh",
}


def find_states(question):
    text = question.lower()
    found = []
    for state in STATES:
        if re.search(r"\b" + re.escape(state.lower()) + r"\b", text) and state not in found:
            found.append(state)
    for nickname, state in NICKNAMES.items():
        if re.search(r"(?<![a-z])" + re.escape(nickname) + r"(?![a-z])", text) and state not in found:
            found.append(state)
    return found


def get_context_docs(data):
    """The dishes and the guideline text that the answer is built from."""
    question = data["input"]
    guide_docs = guide_store.similarity_search(question, k=4)         # best 4 guideline chunks

    states = find_states(question)
    if states:                                                       # "Gujarat menu": search only that state
        state_filter = {"state": states[0]} if len(states) == 1 else {"state": {"$in": states}}
        dishes = menu_store.similarity_search(question, k=40, filter=state_filter)
    else:                                                            # health problem or any other question
        hint = guide_docs[0].page_content.split("] ", 1)[-1][:300] if guide_docs else ""
        found = menu_store.similarity_search(question, k=15) + menu_store.similarity_search(question + " " + hint, k=15)
        seen, dishes = set(), []
        for doc in found:
            if doc.metadata.get("name") not in seen:                 # no dish twice
                seen.add(doc.metadata.get("name"))
                dishes.append(doc)
    return dishes + guide_docs


retriever = RunnableLambda(get_context_docs)

chatModel = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    try:
        response = rag_chain.invoke({"input": msg})
    except Exception as e:
        print("Error:", e)
        return "Sorry, I could not think of an answer right now. Please try again."
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
