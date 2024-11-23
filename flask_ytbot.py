from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from urllib.parse import urlparse, parse_qs
from io import BytesIO
from reportlab.pdfgen import canvas
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os

app = Flask(__name__)

# Set up environment for LLM
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual API key
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                       model_kwargs={'device': 'cpu'},
                                       encode_kwargs={'normalize_embeddings': True})
llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"],
               model_name="llama3-70b-8192")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context and also with the information present in LLM.
    <context>
    {context}
    </context>
    Questions: {input}
    """
)


# Utility Functions
def extract_video_id(link):
    """Extract the video ID from a YouTube video link."""
    parsed_url = urlparse(link)
    if parsed_url.netloc == "www.youtube.com":
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]
    elif parsed_url.netloc == "youtu.be":
        return parsed_url.path.lstrip("/")
    return None


def get_subtitles(video_id):
    """Retrieve subtitles for the given video ID."""
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    formatter = TextFormatter()
    return formatter.format_transcript(transcript)


def transcript_to_pdf(lines, output_path="transcript.pdf"):
    """Generate a PDF transcript from subtitles."""
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, 750, "YouTube Transcript")
    y_pos = 700
    for line in lines:
        pdf.drawString(50, y_pos, line)
        y_pos -= 15
    pdf.save()
    with open(output_path, "wb") as output_file:
        output_file.write(pdf_buffer.getvalue())

@app.route('/')
def home():
    return "Welcome to the YouTube Transcript and PDF Generator API!"
# Endpoints
@app.route('/generate_pdf', methods=['GET'])
def generate_pdf():
    """Endpoint to generate a PDF from a YouTube link."""
    video_link = request.args.get("video_link")

    if not video_link:
        return jsonify({"error": "No video link provided."}), 400

    video_id = extract_video_id(video_link)
    if not video_id:
        return jsonify({"error": "Invalid YouTube link."}), 400

    try:
        subtitles = get_subtitles(video_id)
        lines = subtitles.splitlines()
        transcript_to_pdf(lines)
        return jsonify({"message": "PDF generated successfully.", "pdf_path": "transcript.pdf"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/query_pdf', methods=['GET'])
def query_pdf():
    """Endpoint to query the PDF."""
    query = request.args.get("query")

    if not query:
        return jsonify({"error": "No query provided."}), 400

    try:
        # Load PDF and process
        pdf_loader = PyPDFLoader("transcript.pdf")
        pdf_docs = pdf_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        pdf_final_documents = text_splitter.split_documents(pdf_docs)

        # Create FAISS vector store and retrieval chain
        pdf_vectors = FAISS.from_documents(pdf_final_documents, embeddings)
        document_chain = create_stuff_documents_chain(llm, prompt)
        pdf_retriever = pdf_vectors.as_retriever()
        pdf_retrieval_chain = create_retrieval_chain(pdf_retriever, document_chain)
        # Get response from LLM
        response = pdf_retrieval_chain.invoke({"input": query})
        return jsonify({"response": response['context'][0].page_content}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run()
