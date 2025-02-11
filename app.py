import os
from flask import Flask, request, render_template
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_community.chat_models import AzureChatOpenAI  # UPDATED IMPORT
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI  # Corrected import


# Load environment variables
load_dotenv()

# Fetch API credentials from .env
api_key = os.getenv("OPENAI_GPT_API_KEY")
api_endpoint = os.getenv("OPENAI_GPT_ENDPOINT")
api_version = "2024-08-01-preview"

# Azure Search Configuration
SEARCH_SERVICE_ENDPOINT = os.getenv("SEARCH_SERVICE_ENDPOINT")
SEARCH_ADMIN_KEY = os.getenv("SEARCH_ADMIN_KEY")
SEARCH_INDEX_NAME = os.getenv("SEARCH_INDEX_NAME")

# Initialize Azure Search Client
search_client = SearchClient(
    endpoint=SEARCH_SERVICE_ENDPOINT,
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_ADMIN_KEY)
)

def get_search_results(query):
    """Retrieve relevant search results from Azure Search."""
    results = search_client.search(
        search_text=query,  
        select=["text"],  # Fetch only the relevant text field
        top=3,  # Limit results to 3 to avoid repetition
        query_type="simple"
    )
    
    return [result["text"] for result in results]

# Initialize LangChain Azure OpenAI Model
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # Updated parameter
    azure_endpoint=api_endpoint,
    api_key=api_key,
    api_version=api_version,
    temperature=0
)


def chat_with_gpt(query):
    """Generate a GPT response using search results as context with LangChain."""
    chunks = get_search_results(query)

    if chunks:
        context = "\n\n".join(chunks[:3])
        system_message = f"You are an AI assistant providing helpful information. Use the following context to answer the user's query:\n\n{context}"
    else:
        system_message = "You are an AI assistant providing helpful information."

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=query)
    ]

    response = llm.invoke(messages)  # Corrected method

    return response.content.strip()

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    search_results = []
    gpt_response = ""
    
    if request.method == "POST":
        query = request.form.get("query")  # Get user input safely
        
        if query:
            search_results = get_search_results(query)  
            gpt_response = chat_with_gpt(query)

    return render_template("index.html", results=search_results, response=gpt_response)

if __name__ == "__main__":
    app.run(debug=True)
