import chromadb
import os 
import dotenv
import PyPDF2
import google.generativeai as genai
import chromadb.utils.embedding_functions as embedding_functions
dotenv.load_dotenv()

genai.configure(api_key=os.environ.get('GOOGLE_API_KEY')) 
def generate_response(prompt, data=""):
    model = genai.GenerativeModel('gemini-pro')
    response=model.generate_content(f""" 
                                    You are a friendly AI Chatbot named "Bot".
                                    Use the given data to answer the question from the user, 
                                    if the data is not relevent then respond based on your knowledge
                                    Data: {str(data)}
                                    User: {prompt}
 """)
    response.resolve()
    return response.text
    
google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.environ.get("GOOGLE_API_KEY"))    
client = chromadb.Client()
collection = client.get_or_create_collection(name="main")

def db(text, embed, file, ids):
    collection.add(
    documents=[text],
    embeddings=[embed],
    ids=[file+" "+ids]
    )

def embed(file):
    with open(file, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            embeddings = google_ef([text])
            db(text, embeddings[0][0], str(file), str(page_num))

def query_search(text):
    embedding=google_ef([text])
    res=collection.query(
        query_embeddings=[embedding[0][0]],
        n_results=5,
    )
    return res

