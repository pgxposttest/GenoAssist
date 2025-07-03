# To start chatbot, run: chainlit run model.py -w

# Git Sync:
# git init
# git add .
# git commit -m "Initial commit"
# git branch -M main
# git push -u origin main

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.callbacks.manager import CallbackManager
from langchain.schema.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from tabulate import tabulate
from openai import OpenAI

import fitz
import chainlit as cl
import os
import string
import pandas as pd
import dataframe_image as dfi
import suggestive_search  
import asyncio
import matplotlib.pyplot as plt
import os

DB_FAISS_PATH = "vectorstore/db_faiss"

template = """ Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question.
Do not mention that you derived your answer from the given context.
Your answer must be no more than 150 words. If there are no relevant documents on a specific topic, simply state "Sorry, I don’t have the relevant information on that topic."
If you don't know the answer, please just say that you do not know the answer, don't try to make up an answer. Allow translations into Chinese, Malay and Tamil if requested by the user. Allow tables to be generated if requested by the user.
------

If asked for counselling points, ignore above word limit to comprehensively cover all of the following counselling points. Write in first person (as a pharmacist), using simple language that a layperson can understand, following the template below. 

Start with these exact words every time: “I am (your name), a pharmacist. I will be explaining your pharmacogenomics (PGx) test results. They are out on HealthHub, you can view the detailed report on the NalaGenetics app.” 
(your name) is for the user to fill in their name, don’t fill in “ChatGPT”.

Then use the following format and break up each section into bullet points:

1. **Background & Purpose**  
  - Say these exact words every time: “To recap, this PGx test was done to find out how your genes affect your body’s response to medications. Our genes are made up of DNA, which is like the instruction manual for our bodies. It determines many processes, including how our bodies process medications. Not everyone responds to a medication in the same way — for some people, a medication may be effective, while for others, the same medication may be harmful or not effective. Hence knowing what kind of genes we carry will allow our doctor to select the best medication for us that is safe and effective.”

2. **Explanation of Test Results**  
   - Say these exact words every time: “I will now go through your test results with you, feel free to ask me if you have any questions.”
   - Explain in layman terms what the gene and the enzyme it codes for does (e.g. “CYP2C19 helps activate clopidogrel”).  
   - Explain the phenotype/metabolizer status for each gene (e.g. Poor/Intermediate/Normal/Rapid/Ultrarapid metabolizer).
   - Extract the patient medication list/medication records from the report and list all the medications. Then list as sub-bullet points the medications that are implicated by the patient’s genotype, and state the corresponding implications on drug response, effectiveness, side effects risk, and potential changes the doctor may make to dosing or choice of therapy. Do not miss any medications and do not mention medications the patient is not taking.
   - If no patient medication list/medication records can be found in the report, prompt user to ask patient for medications they are currently taking.

3. **Actionable Next Steps**  
   - Say these exact words every time: “Don’t worry, you don’t have to remember all your PGx test results as the hospital electronic medical record system will alert the doctor if they prescribe any medication that interacts with your genes.” 
   - Emphasize: “Do **not** change any medications on your own. Always discuss with your doctor.”

4. **Caveats**  
   - Say these exact words every time: 
   - “Take note that the test results are used to predict responses to medications, not to diagnose, treat or cure any disease.” 
   - “Genetic counselling and disease screening are not necessary. Your family members do not need to undergo genetic testing.”
   - “Besides your genes, other factors like environment, organ function and medications that you are currently taking may affect how you respond to medications.”
   - “The test only covers common genetic variants that affect response to medications. It may not detect rare variants.”
   - “Please inform your doctor if you have undergone procedures such as recent blood transfusion, liver or stem cell transplant, as they may affect your test results.”
   - “The interpretation of your test results is accurate as of now, but may evolve as our understanding of PGx advances, we will notify you.”

5. **App & Follow-Up**  
   - Say these exact words every time: “If you would like, I can show you how to use the NalaGenetics app to access your test report. You can share your report with other healthcare providers.”

------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
Question: {question}

Only return the helpful answer below and explain.
"""

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
GENE_DATAFRAME = suggestive_search.load_database()
CPIC_DATAFRAME = suggestive_search.load_cpic()

MEDICATIONS = pd.concat([GENE_DATAFRAME["Medication"], CPIC_DATAFRAME["Medication"]]).str.lower().sort_values().unique()
GENES = pd.concat([GENE_DATAFRAME["Gene"], CPIC_DATAFRAME["Gene"]]).str.lower().sort_values().unique()

print(MEDICATIONS)
print(GENES)


def set_custom_prompt():
    prompt = PromptTemplate(template=template, input_variables=["history", "context", "question","gene","medication"])
    return prompt


def load_llm():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = ChatOpenAI(model_name='gpt-4.1-mini', streaming=True)
    return llm


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 7}),
        return_source_documents=True,
        chain_type_kwargs={
            "verbose": False,
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"),
        }
    )
    return qa_chain


def qa_bot():
    embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    llm = load_llm()
    qa_prompt = set_custom_prompt()
    db = FAISS.load_local(DB_FAISS_PATH, embedding, allow_dangerous_deserialization=True)
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({"query": query})
    return response


def extract_summary_table_from_pdf(filepath):
    doc = fitz.open(filepath)
    summary_text = ""
    
    for page in doc[:10]:  # Limit to first 10 pages
        text = page.get_text()
        if "Genomic Information" in text and "Phenotype" in text:
            summary_text = text
            break

    lines = summary_text.splitlines()
    results = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if any(g in line for g in ["CYP2C19", "CYP2C9", "CYP2D6", "SLCO1B1", "CYP3A4", "VKORC1", "DPYD", "TPMT"]):
            try:
                gene = line
                genotype = lines[i + 1].strip()
                phenotype = lines[i + 2].strip()

                # Skip any rows that are headers or repeated
                if any(x in phenotype for x in ["Current Medications", "Genotype", "Phenotype"]):
                    i += 1
                    continue

                results.append([gene, genotype, phenotype])
                i += 3
            except IndexError:
                break
        else:
            i += 1

    if results:
        df = pd.DataFrame(results, columns=["Gene", "Genotype", "Phenotype"])
        return df
    else:
        return None


@cl.on_chat_start
async def start():    
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()

    await asyncio.sleep(1)
    msg.content = "Hi, welcome to GenoAssist! How can I help you? Click on “Readme” (top right icon) to find out more about me!"
    await msg.update()

    action = await cl.AskActionMessage(
        content="Choose an option below:",
        actions=[
            cl.Action(name="upload", value="upload", label="📎 Upload Patient's PGx Test Report", payload={}),
            cl.Action(name="chat", value="chat", label="💬 Start Chatting", payload={})
        ]
    ).send()

    extracted_text = ""

    if action and action.get("name") == "upload":
        files = await cl.AskFileMessage(
            content="Upload patient's PGx test report",
            accept=[".pdf", ".docx", ".xlsx", ".csv"],
            max_size_mb=35
        ).send()

        if not files:
            await cl.Message(content="No file was uploaded. You can still chat with me.").send()
        else:
            file = files[0]
            file_path = file.path

            try:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file_path)
                    extracted_text = df.to_string(index=False)

                elif file.name.endswith(".xlsx"):
                    df = pd.read_excel(file_path)
                    extracted_text = df.to_string(index=False)

                elif file.name.endswith(".docx"):
                    from docx import Document
                    doc = Document(file_path)
                    extracted_text = "\n".join([p.text for p in doc.paragraphs])

                elif file.name.endswith(".pdf"):
                    doc = fitz.open(file_path)
                    extracted_text = "\n".join([page.get_text() for page in doc])

                    # 🧬 Try PGx summary table extraction
                    summary_df = extract_summary_table_from_pdf(file_path)
                    if summary_df is not None:
                        summary_md = tabulate(summary_df, headers="keys", tablefmt="github")

                        cl.user_session.set("summary_df", summary_df)

                        await cl.Message(
                            content=f"🧬 Here's a summary of the patient's PGx test results:\n\n{summary_md}"
                        ).send()
                    else:
                        await cl.Message(content="⚠️ PGx summary table not found in this PDF.").send()

                cl.user_session.set("reference_text", extracted_text)
                await cl.Message(content="✅ File uploaded and stored for future questions.").send()

            except Exception as e:
                await cl.Message(content=f"❌ Failed to process file: {e}").send()

    else:
        await cl.Message(content="Session timeout, you can still chat with me or refresh the webpage to upload a patient’s PGx test report.").send()

    cl.user_session.set("chain", chain)
    cl.user_session.set("reference_text", extracted_text)

    
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    reference_text = cl.user_session.get("reference_text", "")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = ""
    image = ""
    image_name = "result_df.png"

    # Step 1: Determine if user requested a specific language
    user_lang_instruction = ""
    if "in chinese" in message.content.lower():
        user_lang_instruction = "Please respond in Chinese."
    elif "in malay" in message.content.lower():
        user_lang_instruction = "Please respond in Malay."
    elif "in tamil" in message.content.lower():
        user_lang_instruction = "Please respond in Tamil."

    # 👇 Integrate the uploaded file's content into the query
    query = message.content
    if reference_text:
        query = f"""Use the following uploaded file content as context if relevant:\n\n{reference_text}\n\nUser question: {message.content}"""
        
    if user_lang_instruction:
        query = f"{user_lang_instruction}\n\n{query}"
    
    ai_res = await chain.acall({"query": query}, callbacks=[cb])
    print(ai_res["result"])

    for word in message.content.split():
        word = word.translate(str.maketrans("", "", string.punctuation)).lower()
        if word in MEDICATIONS or word in GENES:
            gene_res_cap = suggestive_search.search(word.capitalize(), GENE_DATAFRAME, False)
            gene_res_upper = suggestive_search.search(word.upper(), GENE_DATAFRAME, False)
            gene_res = pd.concat([gene_res_cap, gene_res_upper])

            cpic_res_cap = suggestive_search.search(word.capitalize(), CPIC_DATAFRAME, False)
            cpic_res_upper = suggestive_search.search(word.upper(), CPIC_DATAFRAME, False)
            cpic_res = pd.concat([cpic_res_cap, cpic_res_upper])

            # Creating DataFrame
            resulting_dataframe = pd.merge(gene_res, cpic_res, on=["Medication", "Gene"], how="outer") \
                .sort_values("Medication") \
                .fillna("Not enough information")

            table_md = tabulate(resulting_dataframe.drop(columns=["Guideline"]),
                    headers="keys",
                    tablefmt="github")
            res += f"Here's a table with some information on **{word}**:\n\n{table_md}"

   
    # Append LLM result only if not already streamed by callback
    if not cb.answer_reached:
        res += f"\n\n{ai_res['result']}"

    if image:
        await cl.Message(
            content=res,
            elements=[image]
        ).send()
    else:
        await cl.Message(
            content=res
        ).send()
