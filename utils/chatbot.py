"""Core logic for question answering using LangChain, memory, and chains."""
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate

# prompt to turn chat history into standalone question (used internally)
_CONDENSE_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
)

# QA prompt to answer using context with citations
_QA_PROMPT = PromptTemplate(
    template="""You are a helpful research assistant. Use the following pieces of context to answer the question at the end. Be precise and include citations in square brackets. If you don't know the answer from the context, say you don't know.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

import settings
from utils.indexer import get_retriever

# Custom prompt to improve QA accuracy and enforce citations
_QA_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. 
Be precise and cite your sources. If you don't know the answer from the provided context, say so.

Context:
{context}

Question: {question}

Answer (with citations):"""

QA_PROMPT = PromptTemplate(
    template=_QA_PROMPT_TEMPLATE, input_variables=["context", "question"]
)


def get_memory():
    if settings.CONVERSATION_MEMORY == "summary":
        return ConversationSummaryMemory(llm=get_llm(), memory_key="chat_history")
    else:
        return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def get_llm():
    # Placeholder: adapt for LLaMA, Gemini, etc.
    if settings.LLM_TYPE == "openai":
        return ChatOpenAI(temperature=0, model=settings.OPENAI_MODEL)
    elif settings.LLM_TYPE == "llama":
        # prefer a local LLaMA model if the path is specified
        if settings.LLAMA_MODEL_PATH:
            from langchain.llms import Llama
            return Llama(model_path=settings.LLAMA_MODEL_PATH, temperature=0.7)
        # otherwise fall back to a small HuggingFace model so the assistant can
        # run completely offline without any API keys. gpt2 is light and widely
        # available, but you may change this to another cached transformer.
        try:
            from langchain.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            return HuggingFacePipeline(
                pipeline=hf_pipe,
                model_kwargs={"temperature": 0.7},
            )
        except Exception:
            # if even the HF fallback fails, revert to OpenAI (user may provide key)
            return ChatOpenAI(temperature=0, model=settings.OPENAI_MODEL)
    elif settings.LLM_TYPE == "ollama":
        # use the Ollama HTTP interface; base_url and model can be configured
        from langchain_community.llms import Ollama
        return Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.7,
        )
    else:
        # default to ChatOpenAI
        return ChatOpenAI(temperature=0, model=settings.OPENAI_MODEL)


def build_chain():
    """Construct a conversational retrieval chain with memory.

    Uses ConversationalRetrievalChain which handles QA with chat history
    and document retrieval.
    """
    retriever = get_retriever()
    memory = get_memory()

    # build a conversational retrieval chain with custom prompts
    # output_key='answer' tells memory which field to track
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        memory=memory,
        output_key='answer',
        condense_question_prompt=_CONDENSE_PROMPT,
        combine_docs_chain_kwargs={"prompt": _QA_PROMPT},
    )

    return qa_chain
