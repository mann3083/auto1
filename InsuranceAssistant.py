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



    def extract_PII_Japanese_Text_ENG(self, japText):

        try:
            prompt_and_context = [
                (
                    "system",
                    """You are an expert data extraction assistant. Your task is to accurately extract key information such as name, policy number, date of birth, email, etc., from given sentences. The user will speak the entire sentence, and you must only return the extracted value. 

                The extraction should follow these rules:
                1. **Name Extraction**:
                - Sentence: "My name is Albert Pinto Jr"
                - Extracted_Value: "Albert Pinto Jr"
                - Sentence: "Captain Jack Sparrow"
                - Extracted_Value: "Captain Jack Sparrow"

                2. **Date of Birth Extraction**:
                - Sentence: "My date of birth 2nd June 2024"
                - Extracted_Value: "2024-06-02"
                - Sentence: "3rd July 2009"
                - Extracted_Value: "2009-07-03"

                3. **Email Extraction**:
                - Sentence: "My email is ab2@gmail.com"
                - Extracted_Value: "ab2@gmail.com"

                4. **Policy Number Extraction**:
                - Sentence: "My policy number is 989898"
                - Extracted_Value: "989898"
                - Sentence: "989898"
                - Extracted_Value: "989898"

                5. **Null Extraction**:
                - Sentence: "I don't remember, let me recheck"
                - Extracted_Value: "null"

                6. **Medical Procedure Extraction**:
                - Sentence: "Cataract Surgery happened on the left eye"
                - Extracted_Value: "Cataract Surgery"

                Please follow the above rules and ensure the extracted values are accurate. If the sentence does not contain any of the key information, return "null".

                Examples:

                1. Sentence: "My name is Albert Pinto Jr"
                Extracted_Value: "Albert Pinto Jr"

                2. Sentence: "My email is ab2@gmail.com"
                Extracted_Value: "ab2@gmail.com"

                3. Sentence: "3rd July 2009"
                Extracted_Value: "2009-07-03"

                4. Sentence: "I don't remember, let me recheck"
                Extracted_Value: "null"

                Here is the sentence to process:
                Sentence: "{context}"
                Extracted_Value:""",
                ),
            ]
            chat = ChatOpenAI(model="gpt-4o", api_key=self.O_API_KEY, temperature=0.1)
            chat_template = ChatPromptTemplate.from_messages(prompt_and_context)
            message = chat_template.format_messages(context=japText)
            ai_resp = chat.invoke(message)
            return ai_resp.content
        except Exception as e:
            return str(e)



    def extract_PII_Japanese_Text_JP(self, japText):

        try:
            prompt_and_context = [
                (
                    "system",
                    """あなたは専門的なデータ抽出アシスタントです。あなたのタスクは、名前、保険番号、生年月日、メールアドレスなどの重要な情報を与えられた文章から正確に抽出することです。ユーザーは文章全体を話しますが、あなたは抽出された値のみを返す必要があります。

                    抽出は以下のルールに従って行ってください：
                    1. **名前の抽出**：
                    - 文章: "私の名前はアルバート・ピント・ジュニアです"
                    - 抽出された値: "アルバート・ピント・ジュニア"
                    - 文章: "キャプテン・ジャック・スパロウ"
                    - 抽出された値: "キャプテン・ジャック・スパロウ"

                    2. **生年月日の抽出**：
                    - 文章: "私の生年月日は2024年6月2日です"
                    - 抽出された値: "2024-06-02"
                    - 文章: "2009年7月3日"
                    - 抽出された値: "2009-07-03"

                    3. **メールアドレスの抽出**：
                    - 文章: "私のメールアドレスはab2@gmail.comです"
                    - 抽出された値: "ab2@gmail.com"

                    4. **保険番号の抽出**：
                    - 文章: "私の保険番号は989898です"
                    - 抽出された値: "989898"
                    - 文章: "989898"
                    - 抽出された値: "989898"

                    5. **nullの抽出**：
                    - 文章: "覚えていません。確認させてください"
                    - 抽出された値: "null"

                    6. **医療手続きの抽出**：
                    - 文章: "左目に白内障手術をしました"
                    - 抽出された値: "白内障手術"

                    上記のルールに従って、抽出された値が正確であることを確認してください。文章にキー情報が含まれていない場合は、「null」と返してください。

                    例：

                    1. 文章: "私の名前はアルバート・ピント・ジュニアです"
                    抽出された値: "アルバート・ピント・ジュニア"

                    2. 文章: "私のメールアドレスはab2@gmail.comです"
                    抽出された値: "ab2@gmail.com"

                    3. 文章: "2009年7月3日"
                    抽出された値: "2009-07-03"

                    4. 文章: "覚えていません。確認させてください"
                    抽出された値: "null"

                    こちらが処理する文章です：
                    文章: "{context}"
                    抽出された値:""",
                ),
            ]
            chat = ChatOpenAI(model="gpt-4o", api_key=self.O_API_KEY, temperature=0.1)
            chat_template = ChatPromptTemplate.from_messages(prompt_and_context)
            message = chat_template.format_messages(context=japText)
            ai_resp = chat.invoke(message)
            return ai_resp.content
        except Exception as e:
            return str(e)



    def extract_PII_Japanese_Text_JAP(self, japText):
        try:
            prompt_and_context = [
                (
                    "system",
                    """You are an expert data extraction assistant. Your task is to accurately extract key information such as name, policy number, date of birth, email, etc., from given sentences. The user will speak the entire sentence, and you must only return the extracted value. 

                    The extraction should follow these rules:

                    **Japanese Instructions**:
                    1. **名前の抽出**:
                    - 文: "私の名前はアルバート・ピント・ジュニアです"
                    - 抽出された値: "アルバート・ピント・ジュニア"
                    - 文: "キャプテン・ジャック・スパロウ"
                    - 抽出された値: "キャプテン・ジャック・スパロウ"

                    2. **生年月日の抽出**:
                    - 文: "私の生年月日は2024年6月2日です"
                    - 抽出された値: "2024-06-02"
                    - 文: "2009年7月3日"
                    - 抽出された値: "2009-07-03"

                    3. **メールアドレスの抽出**:
                    - 文: "私のメールアドレスはab2@gmail.comです"
                    - 抽出された値: "ab2@gmail.com"

                    4. **ポリシー番号の抽出**:
                    - 文: "私のポリシー番号は989898です"
                    - 抽出された値: "989898"
                    - 文: "989898"
                    - 抽出された値: "989898"

                    5. **nullの抽出**:
                    - 文: "思い出せません、再確認させてください"
                    - 抽出された値: "null"

                    6. **医療手続きの抽出**:
                    - 文: "左目で白内障手術が行われました"
                    - 抽出された値: "白内障手術"

                    これらのルールに従い、抽出された値が正確であることを確認してください。文に重要な情報が含まれていない場合は、「null」を返してください。

                    Examples:

                    1. 文: "私の名前はアルバート・ピント・ジュニアです"
                    抽出された値: "アルバート・ピント・ジュニア"

                    2. 文: "私のメールアドレスはab2@gmail.comです"
                    抽出された値: "ab2@gmail.com"

                    3. 文: "2009年7月3日"
                    抽出された値: "2009-07-03"

                    4. 文: "思い出せません、再確認させてください"
                    抽出された値: "null"

                    Here is the sentence to process:
                    Sentence: "{context}"
                    Extracted_Value: """,
                ),
            ]
            chat = ChatOpenAI(model="gpt-4", api_key=self.O_API_KEY, temperature=0.1)
            chat_template = ChatPromptTemplate.from_messages(prompt_and_context)
            message = chat_template.format_messages(context=japText)
            ai_resp = chat.invoke(message)
            return ai_resp.content
        except Exception as e:
            return str(e)
