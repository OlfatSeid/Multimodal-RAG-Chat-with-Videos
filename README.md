# Multimodal RAG: Chat with Videos


## Overview

This project provides a web application interface for interacting with YouTube videos by querying their transcripts. Users can input a YouTube video URL and a question related to the video content. The application fetches the video transcript, processes it to create a vector store for similarity search, and generates responses using a language model.

## Features

- Fetches transcripts of YouTube videos.
- Uses FAISS for vector similarity search on the video transcript.
- Employs a HuggingFace LLM pipeline for generating responses.
- Provides a user-friendly web interface built with Gradio.

## Prerequisites

- Python 3.7 or later.
- Required libraries:
  - `langchain`
  - `gradio`
  - `transformers`
  - `sentence_transformers`
  - `youtube_transcript_api`
  - `faiss`

Install dependencies using the following command:

```bash
pip install langchain gradio transformers sentence_transformers youtube-transcript-api faiss-cpu
```

## Project Structure

- **`main_interface(video_url, query)`**: Main function to process the user query and return the response and embedded video display.
- **Gradio UI**: User interface for entering video URLs, asking questions, and displaying results.
- **Transcript Fetching**: Uses `YouTubeTranscriptApi` to get the transcript of the given video.
- **Vector Store Creation**: Processes the transcript using SentenceTransformer embeddings and stores it in a FAISS index for similarity search.
- **Response Generation**: Utilizes LangChain and HuggingFace pipeline to generate answers based on the context extracted from the transcript.

## Usage

1. **Launch the application**:
   Run the script using:

   ```bash
   python app.py
   ```

2. **Interact with the UI**:

   - Enter a valid YouTube video URL.
   - Ask a question related to the video content.
   - View the response and the embedded video.

## Code Explanation

### Key Components

#### Loading Models

```python
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", device=0)
llm = HuggingFacePipeline(pipeline=llm_pipeline)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

This section initializes the text generation model (`google/flan-t5-small`) and the embedding model (`sentence-transformers/all-MiniLM-L6-v2`).

#### Fetching YouTube Transcripts

```python
def fetch_youtube_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        return text
    except Exception as e:
        return f"Error fetching transcript: {e}"
```

This function retrieves the transcript of a given YouTube video using `YouTubeTranscriptApi`.

#### Processing Queries

```python
def process_query(video_url, query):
    transcript = fetch_youtube_transcript(video_url)
    if "Error" in transcript:
        return transcript, ""

    vector_store = create_vector_store(transcript)

    docs = vector_store.similarity_search(query, k=5)

    if not docs:
        return "No relevant context found for your query.", ""

    context = " ".join([doc.page_content for doc in docs])
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template="You are an assistant answering questions based on the following video transcript:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:",
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)

    try:
        response = chain.run({"context": context, "query": query})
        return response, video_url
    except Exception as e:
        return f"Error generating response: {e}", video_url
```

This function processes the query by:

1. Fetching the transcript.
2. Creating a vector store for similarity search.
3. Extracting relevant context.
4. Generating a response using LangChain.

### Gradio UI

```python
with gr.Blocks(css="""
    body {
        background-color:black !important; /* Ensures the black background is applied */
        color: white !important; /* Ensures text is white */
    }
    button {
        background-color: #1a73e8 !important; /* Blue buttons */
        color: black !important; /* Button text color */
    }
    button:hover {
        background-color: green !important; /* Darker blue on hover */
    }
""") as app:
    gr.Markdown("Chat with YouTube Video")
    with gr.Row():
        video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the video URL here")
    with gr.Row():
        query = gr.Textbox(label="Ask a Question", placeholder="Ask something about the video transcript")
    with gr.Row():
        response = gr.Textbox(label="Response", placeholder="The assistant's response will appear here", interactive=False)
    with gr.Row():
        video_display = gr.HTML(value="")
    with gr.Row():
        submit_btn = gr.Button("Submit")

    submit_btn.click(main_interface, inputs=[video_url, query], outputs=[response, video_display])

if __name__ == "__main__":
    app.launch()
```

This creates a Gradio-based UI for user interaction with the application.







