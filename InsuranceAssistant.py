import os
import json
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate

from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speechsdk
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.ai.textanalytics import TextAnalyticsClient


class InsuranceAssistant:
    def __init__(self):
        load_dotenv()  # take environment variables from .env.
        self.SPEECH_KEY = os.getenv("SPEECH_KEY")
        self.SPEECH_REGION = os.getenv("SPEECH_REGION")
        self.O_API_KEY = os.getenv("O_API_KEY")
        self.endpointDocIntel = os.getenv("endpointDocIntel")
        self.keyDocIntel = os.getenv("keyDocIntel")
        self.MULTI_KEY = os.getenv("MULTI_KEY")
        self.MULTI_REGION = os.getenv("MULTI_REGION")
        self.MULTI_ENDPOINT = os.getenv("MULTI_ENDPOINT")

        with open("prompts.json", "r") as file:
            self.promptLibrary = json.load(file)

        self.promptLibrary["USER_INTENT_RECOGNITION"] = self.USER_INTENT_PROMPT()
        self.client = OpenAI(api_key=self.O_API_KEY)
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.SPEECH_KEY, region=self.SPEECH_REGION
        )

    def USER_INTENT_PROMPT(self):
        return """
            Refer to the context and the inferred intent in the examples 
            below: 
            
            Context: I was in a car accident yesterday and need to get my vehicle repaired. 
            Intent: File Accident Claim 
            
            Context: Can you help me with that? 
            Intent: Not known
            
            Context:Hello, I'd like to start the process of filing a claim. What information do you need from me? 
            Intent: Not known
            
            Context:I had a medical emergency last night and was hospitalized. 
            Intent: File Medical Claim

            Context:A close relative relative passed away unexpectedly. 
            Intent: File Life Claim 

            If not sure - Say 'Not Known'. 
            
            You must make sure that if the context has accident or medical then the intent must 
            include that. Even for slightest doubt Say 'Not Known'. You must infer the 
            intent from the context
            Think step by step.
        """

    def prompt_to_question(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional chatbot, given a input convert it to a relevant question.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.1,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content

    def prompt_creation(self, query, prmpt, temp=0.1):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prmpt},
                {"role": "user", "content": query},
            ],
            max_tokens=75,
            temperature=temp,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content

    def semantic_search(self, query, chroma_store, top_k=5):
        results = chroma_store.similarity_search(query, top_k=top_k)
        return results

    def generate_response(self, user_intent):
        query = f"What are the mandatory documents needed for {user_intent}."
        list_of_vector_db = {
            "disability": "chromaDB/Disability_Claim",
            "life": "chromaDB/Life_Claims",
            "death": "chromaDB/Death_Claim",
            "cancer": "chromaDB/Cancer_Claim",
            "critical": "chromaDB/Critical_Illness_Claim",
        }
        segment = ""
        user_intent = user_intent.lower()
        for k in list_of_vector_db.keys():
            if k in user_intent:
                segment = list_of_vector_db[k]
                break

        emb_model = OpenAIEmbeddings(api_key=self.O_API_KEY)
        intent_db = Chroma(persist_directory=segment, embedding_function=emb_model)

        search_results = self.semantic_search(query, intent_db)
        responses = [result.page_content for result in search_results]
        sources = {
            result.metadata["source"].split("/")[-1] for result in search_results
        }
        doc_source = " ".join(sources)
        context = " ".join(responses)

        prompt_and_context = [
            ("human", "{query}."),
            (
                "system",
                "You are an expert insurance agent for claim processing. Only refer the context provided in to answer the query in a concise, bulleted points if necessary and professional manner,  #### Context:{context}.",
            ),
        ]
        chat = ChatOpenAI(
            model="gpt-3.5-turbo-0125", api_key=self.O_API_KEY, temperature=0.1
        )
        chat_template = ChatPromptTemplate.from_messages(prompt_and_context)
        message = chat_template.format_messages(context=context, query=query)

        ai_resp = chat.invoke(message)
        return ai_resp.content + "For further details refer: \n\n" + doc_source + "\n\n"

    def japanese_rag(self, query):
        chat = ChatOpenAI(
            model="gpt-3.5-turbo-0125", api_key=self.O_API_KEY, temperature=0.1
        )
        emb_model = OpenAIEmbeddings(api_key=self.O_API_KEY)
        intent_db = Chroma(
            persist_directory="chromaDB/JapanRAG", embedding_function=emb_model
        )
        search_results = intent_db.similarity_search(query, top_k=5)
        responses = [result.page_content for result in search_results]
        context = " ".join(responses)
        prompt_and_context = [
            ("human", "Given the context:{context}, {query}."),
            (
                "system",
                "You are an expert in Japanese and English language with extensive knowledge of Japanese culture and economy. Your response must be in JAPANESE. Think STEP BY STEP",
            ),
        ]
        chat_template = ChatPromptTemplate.from_messages(prompt_and_context)
        message = chat_template.format_messages(query=query, context=context)
        ai_resp = chat.invoke(message)
        return ai_resp.content

    def extractKeyPhrases(self, text):
        # A document is a single unit to be analyzed by the predictive models in the Language service. The input for each operation is passed as a list of documents.
        doc = [text]
        ta_cred = AzureKeyCredential(self.MULTI_KEY)
        ta_client = TextAnalyticsClient(
            endpoint=self.MULTI_ENDPOINT, credential=ta_cred
        )
        try:
            response = ta_client.extract_key_phrases(documents=doc)[0]
            if not response.is_error:
                print("Key Phrases:")
                for phrase in response.key_phrases:
                    print(f"\t{phrase}")
            else:
                print(f"Error: {response.code}, {response.message}")
        except Exception as err:
            print(f"Encountered exception. {err}")

    def extract_PII_Japanese_Text_JP(self, japText):
        prompt_and_context = [
            # ("human", "{query}."),
            (
                "system",
                """You are an expert data entry operator in Japanese. 
             Extract PII data such as name or date or birth or policy number 
             or address from the given sentence. 
             You must only return response in 
             key valye pair - like {{name:extractedName}},
             {{date_of_birth:extracted_date_of_birth}}. 
             If there is no PII information return {{null}} only return
             key that has a value - dont make mistake. #### Context:{context}. 
             
            These are PII information accuracy is very important think step 
            by step""",
            ),
        ]
        chat = ChatOpenAI(model="gpt-4o", api_key=self.O_API_KEY, temperature=0.1)
        chat_template = ChatPromptTemplate.from_messages(prompt_and_context)
        message = chat_template.format_messages(context=japText)
        ai_resp = chat.invoke(message)
        return ai_resp.content

    def extract_PII_Japanese_Text_EN(self, japText):

        try:
            prompt_and_context = [

                (
                    "system",
                    """You are an expert data entry operator in English.
                    Given a sentence extract the context from it - say
                data or key information such as name or date or birth or policy number 
                or address or email or nature of illness. 
                You must extract and return only the value from the sentence.

                Example:
                Sentence: "My name is Albert Pinto Jr"
                Extracted_Value: "Albert Pinto Jr"

                Sentence: "Captain Jack Sparrow"
                Extracted_Value: "Captain Jack Sparrow"

                Sentence: "My date of birth 2nd June 2024"
                Extracted_Value: "2024-06-02"

                Extracted_Value: Sentence: "3rd July 2009"
                Extracted_Value: "2009-07-03"

                Sentence: "My email is ab2@gmail.com"
                Extracted_Value: "ab2@gmail.com"

                Sentence: "My policy number is 989898"
                Extracted_Value: "989898"

                Sentence: "989898"
                "989898"

                Sentence: "I dont remeber let me recheck"
                Extracted_Value: "null"

                Sentence: "Cataract Surgery happended on the left eye"
                Extracted_Value: "Cataract Surgery"

                #### Sentence:
                {context}

                For the date of birth feild the returned value must be in YYYY-MM-DD format.
                Accuracy is very important,returned value must be in english. 
                The response is only one value and not the whole sentence or paragraph.
                Think step by step""",
                ),
            ]
            chat = ChatOpenAI(model="gpt-4o", api_key=self.O_API_KEY, temperature=0.1)
            chat_template = ChatPromptTemplate.from_messages(prompt_and_context)
            message = chat_template.format_messages(context=japText)
            ai_resp = chat.invoke(message)
            return ai_resp.content
        except Exception as e:
            return str(e)

    def extract_PII_Japanese_Text(self, japText, LANG):

        if LANG == "en":
            LANG = "english"
        elif LANG == "ja":
            LANG = "japanese"
        else:
            LANG = "english"

        prompt_and_context = [
            # ("human", "{query}."),
            (
                "system",
                """You are an expert data entry operator in {LANG}. 
             Extract PII data such as name or date or birth or policy number 
             or address from the given sentence. 
             You must only return response in
             key valye pair - like name:extractedName,
             date_of_birth:extracted_date_of_birth. 
             If there is no PII information return null only return
             key that has a value - dont make mistake. #### Context:{context}. 
             
            These are PII information accuracy is very important,returned value
            must be in {LANG} think step by step""",
            ),
        ]
        chat = ChatOpenAI(
            model="gpt-3.5-turbo-0125", api_key=self.O_API_KEY, temperature=0.1
        )
        chat_template = ChatPromptTemplate.from_messages(prompt_and_context)
        message = chat_template.format_messages(context=japText, LANG=LANG)
        ai_resp = chat.invoke(message)
        return ai_resp.content
