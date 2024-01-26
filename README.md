# Retrieval-Augmented-Generation-for-Production-with-LangChain-LlamaIndex

## Basic Concepts

Here, we'll review the essential features of LangChain. We will examine the architecture comprising various components, such as data loading, processing, and segmentation, to provide optimal information to language models. Additionally, we will highlight the significance of indexing and retrieval.

## 1. Preprocessing the Data
LangChain's approach to structuring documents is particularly favorable for developers and researchers.

It provides tools that help structure documents for convenient use with LLMs. Document loaders simplify the process of loading data into documents, and text splitters break down lengthy pieces of text into smaller chunks for better processing. Finally, the indexing process involves creating a structured database of information that the language model can query to enhance its understanding and responses.

### 1-1. Document Loaders
Document Loaders are responsible for loading documents into structured data. They handle various types of documents, including PDFs, and convert them into a data type that can be processed by the other LangChain functions. It enables loading data from multiple sources into Document objects. LangChain provides over 100 document loaders and integrations with other major providers in the space, like AirByte and Unstructured, and from all sources, such as private S3 buckets and public websites.

Read from Files/Directories

Handling various input formats and transforming them into the Document format is easy. For instance, you can load the CSV data using the CSVLoader. Each row in the CSV file will be transformed into a separate Document.

```
from langchain.document_loaders import CSVLoader

# Load data from a CSV file using CSVLoader
loader = CSVLoader("./data/data.csv")
documents = loader.load()

# Access the content and metadata of each document
for document in documents:    
		content = document.page_content    
		metadata = document.metadata
```

Some of the popular loaders include the TextLoader for text files, the DirectoryLoader for loading all the files in a directory, the UnstructuredMarkdownLoader for markdown files, and the PyPDFLoader for loading PDF files.

Public Source Loaders

Loaders for popular public sources allow the data to be transformed into Document objects. For example, the WikipediaLoader retrieves the content of the specified Wikipedia page and loads it into a Document.

```
from langchain.document_loaders import WikipediaLoader

# Load content from Wikipedia using WikipediaLoader
loader = WikipediaLoader("Machine_learning")
document = loader.load()
```

Another popular loader is UnstructuredURLLoader, which allows reading from public web pages.

Proprietary Data loaders

These loaders are designed to handle proprietary sources that may require additional authentication or setup. For example, a loader could be created to load custom data from an internal database or an API with proprietary access.

Popular loaders of this category are GoogleDriveLoader for loading documents from Google Drive and MongodbLoader for loading documents from a MongoDB database.

### 1-2. Document transformers (chunking methods)
A crucial part of retrieval is fetching only the relevant details of documents. This involves several transformation steps to prepare the documents for retrieval. One of the primary steps here is splitting (or chunking) a large document into smaller segments. LangChain provides several transformation algorithms and optimized logic for specific document types.

LangChain offers several key chunking transformation strategies.

- Fixed-size chunks that define a fixed size that's sufficient for semantically meaningful paragraphs (for example, 300 words) and allows for some overlap (for example, an additional 30 words). Overlapping ensures continuity and context preservation between adjacent chunks of data, improving the coherence and accuracy of the created chunks. For example, you may use the CharacterTextSplitter splitter to split every N character or token if configured with a tokenizer.
- Variable-sized chunks partition the data based on content characteristics, such as end-of-sentence punctuation marks, end-of-line markers, or using features in the NLP libraries. It ensures the preservation of coherent and contextually intact content in all chunks. An example is the RecursiveCharacterTextSplitter splitter.
- Customized chunking when dealing with large documents, you might use variable-sized chunks but also append the document title to chunks from the middle of the document to prevent context loss. This can be done, for example, with the MarkdownHeaderTextSplitter.

Chunking offers a mixed bag of strengths and weaknesses in processing extensive documents with LLMs.

A key advantage is its ability to manage documents that exceed the context window of an LLM. This capability enables the model to handle and analyze significantly larger texts than it could in a single pass, expanding its applicability and utility in processing lengthy documents.

However, this approach comes with a notable drawback. In dividing a document into chunks, there's a risk of losing vital context related to the overall document. While individually coherent, each chunk might only partially capture the nuances and interconnected elements present in the full text. This can lead to a fragmented or incomplete understanding of the document, as important details and subtleties might be overlooked or misinterpreted when the text is not viewed cohesively.

### 1-3. Indexing
Indexing is a process that involves storing and organizing data from various sources into a vector store, which is essential for efficient storing and retrieving. The process typically consists of storing the chunk along with an embedding representation of it, which captures the meaning of the text and makes it easy to retrieve chunks by semantic similarity. Embeddings are usually generated by embedding models, such as the OpenAIEmbeddings models.

## 2. Models

### 2-1. LLMs
LangChain provides an LLM class for interfacing with various language model providers, such as OpenAI, Cohere, and Hugging Face. Here's an example of some integrations. The rest of the list can be found in the conclusion section. Every LLM listed has a direct link to the document page with more detailed information. 

| LLM | DESCRIPTION |
|----------|----------|
| OpenAI| An AI research organization, very popular for the ChatGPT product|
|----------|----------|
| Hugging Face Hub| A platform that hosts thousands of pre-trained models and datasets|
|----------|----------|
| Cohere| A platform that provides natural language understanding APIs powered by large-scale neural networks|
|----------|----------|
| Llama-cpp| A library that makes it easy to load (small) language models locally on your PC|
|----------|----------|
| Azure OpenAI| A cloud service that provides access to OpenAI’s LLMs|





