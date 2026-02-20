# 
from flask import Flask, request, jsonify, render_template
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Load transcript once
video_id = "SJKr7BPOXY0"
try:
    transcript_list = YouTubeTranscriptApi().get_transcript(video_id, languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
except TranscriptsDisabled:
    transcript = "No captions available for this video."

# Chunk + Embed
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant. Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    Context: {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    result = main_chain.invoke(question)
    return jsonify({"answer": result})


if __name__ == "__main__":
    app.run(debug=True)
