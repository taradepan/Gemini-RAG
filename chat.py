import requests
import chromadb
import os 
import dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
dotenv.load_dotenv()

genai.configure(api_key=os.environ.get('GEMINI_API_KEY')) 
def generate_response(prompt, data=""):
    print("*"*100)
    print(data)
    model = genai.GenerativeModel('gemini-pro')
    response=model.generate_content(f""" 
                                    You are a friendly AI Chatbot named "Bot". Respond to the user in natural language.
                                    Use the given data to answer the question from the user, 
                                    if the data is not relevent then respond based on your knowledge
                                    Data: {str(data)}
                                    User: {prompt}
 """)
    response.resolve()
    return response.text
       
client = chromadb.PersistentClient(path='db')
collection = client.create_collection(name="main")

def db(text, embed, file, ids):
    collection.add(
    documents=[text],
    embeddings=[embed],
    ids=[file+" "+ids]
    )

def hf_emb(data):
    API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
    headers = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_KEY')}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
        
    output = query({
        "inputs": data,
    })
    return output

def upload(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    for page_num in range(len(pages)):
        try:
            page = pages[page_num].page_content
            embeddings = hf_emb(page)
            db(page, embeddings, str(file), str(page_num))
            print("Embedded page " + str(page_num) + " of " + file)
        except Exception as e:
            print("Error in " + file + " page " + str(page_num))
            print(e)
            pass
    os.remove(file)

def query_search(text):
    embedding=hf_emb(text)
    print(embedding)
    res=collection.query(
        query_embeddings=[embedding],
        n_results=3,
    )
    return res["documents"]