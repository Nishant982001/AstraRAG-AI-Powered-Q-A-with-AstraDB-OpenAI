from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from datasets import load_dataset

import cassio
from PyPDF2 import PdfReader

ASTRA_DB_APPLICATION_TOKEN = ""
ASTRA_DB_ID = ""

OPENAI_API_KEY = ""
pdfreader = PdfReader("bs2023_24.pdf")

from typing_extensions import Concatenate
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text+=content

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

llm = OpenAI(openai_api_key = OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name='qa_mini_demo',
    session=None,
    keyspace=None
)

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 800,
    chunk_overlap = 200,
    length_function = len,
)

text = text_splitter.split_text(raw_text)

astra_vector_store.add_texts(text[:50])

print("Inserted %i headlines." % len(text[:50]))

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

first_question = True
while True:
    if first_question:
        query_text = input("\nEnter Your question (or type 'quit' to exit): ").strip()
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit ): ").strip()

    if query_text.lower() == "quit":
        break
    if query_text == "":
        continue

    first_question = False
    print("\nQUESTION : \"%s\"" % query_text)
    answer = astra_vector_index.query(query_text, llm = llm).strip()
    print("ANSWER : \"%s\n" % answer)

    print("FIRST DOCUMENTS BY RELEVANCE: ")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        print("    [%0.4f] \"%s ....\"" %(score, doc.page_content[:84]))
