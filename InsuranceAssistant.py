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

        # appdir/prompts.json
        # with open("prompts.json", "r") as file:
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


    def extraction_KYC_ENG(self, japText):

        try:
            prompt_and_context = [
                (
                    "system",
                    """You are an expert extraction algorithm and you must only return the extracted value
                    if not sure return null.

                    The extraction should follow these rules:
                    
                    1. **Name Extraction**:
                    - Sentence: 'hmm!.... Albert Pinto Jr'
                    "Albert Pinto Jr"
                    - Sentence: 'hmm!.... Albus Dumbledore'
                    "Albus Dumbledore"
                    - Sentence: "My name is Albert Pinto Jr"
                    "Albert Pinto Jr"
                    - Sentence: "Captain Jack Sparrow"
                    "Captain Jack Sparrow"
                    - Sentence: "I am called Jane Doe"
                    "Jane Doe"
                    - Sentence: "My full name is Dr. Meredith Grey"
                    "Dr. Meredith Grey"

                    2. **Date Extraction**:
                    - Sentence: "My date of birth is 2nd June 2024"
                    "2024-06-02"
                    - Sentence: "err...let me check....hmm! its 2nd August 1983"
                    "1983-08-02"
                    - Sentence: "3rd July 2009"
                    "2009-07-03"
                    - Sentence: "I was born on 15th August 1990"
                    "1990-08-15"
                    - Sentence: "Her birthdate is December 25, 1985"
                    "1985-12-25"

                    3. **Email Extraction**:
                    - Sentence: "My email is ab2@gmail.com"
                    "ab2@gmail.com"
                    - Sentence: "Contact me at john_doe123@outlook.com"
                    "john_doe123@outlook.com"
                    - Sentence: "My email address is jane.smith@example.com"
                    "jane.smith@example.com"
                    - Sentence: "Send the details to mike@domain.co"
                    "mike@domain.co"

                    4. **Number Extraction**:
                    - Sentence: "My policy number is 989898"
                    "989898"
                    - Sentence: "989898"
                    "989898"
                    - Sentence: "Policy ID: 123456789"
                    "123456789"
                    - Sentence: "Please note my policy number is AB1234567"
                    "AB1234567"

                    Please follow the above rules and ensure the extracted values are accurate.

                    Here is the sentence to process:
                    Sentence: "{context}"
                    """,
                ),
            ]
            chat = ChatOpenAI(
                model="gpt-4o-mini", api_key=self.O_API_KEY, temperature=0
            )
            chat_template = ChatPromptTemplate.from_messages(prompt_and_context)
            message = chat_template.format_messages(context=japText)
            ai_resp = chat.invoke(message)
            return ai_resp.content

        except Exception as e:
            return str(e)

    def extraction_Intent_ENG(self, japText):

        try:
            prompt_and_context = [
                (
                    "system",
                    """You are a medical assistant at a reputable hospital specializing in summarizing patient concerns.
                    Your task is to accurately identify and translate patient descriptions into precise medical terminology.

                        Examples:

                        Caller: "I've been feeling really down and uninterested in things I used to love."
                        "Major Depressive Disorder"

                        Caller: "There's this weird rash that's red and itchy all over my arms."
                        "Contact Dermatitis"

                        Caller: "Sometimes, when I eat bread, my stomach hurts a lot."
                        "Gluten Intolerance (Celiac Disease)"

                        Caller: "Where can I get a water"
                        "null"

                        Now, process the following caller description accordingly.
                        Caller: "{context}"
                    """,
                ),
            ]
            chat = ChatOpenAI(
                model="gpt-4o-mini", api_key=self.O_API_KEY, temperature=0
            )
            chat_template = ChatPromptTemplate.from_messages(prompt_and_context)
            message = chat_template.format_messages(context=japText)
            ai_resp = chat.invoke(message)
            return ai_resp.content

        except Exception as e:
            return str(e)

    def extraction_KYC_JP(self, japText):

        try:
            prompt_and_context = [
                (
                    "system",
                    """あなたはエキスパートの抽出アルゴリズムであり、抽出された値のみを返す必要があります。確信が持てない場合はnullを返してください。.

                    抽出は次のルールに従う必要があります：

                    1. **名前の抽出**:
                    文: 'うーん！.... アルバート・ピント・ジュニア'
                    "アルバート・ピント・ジュニア"
                    文: 'うーん！.... アルバス・ダンブルドア'
                    "アルバス・ダンブルドア"
                    文: "私の名前はアルバート・ピント・ジュニアです"
                    "アルバート・ピント・ジュニア"
                    文: "キャプテン・ジャック・スパロウ"
                    "キャプテン・ジャック・スパロウ"
                    文: "私はジェーン・ドウと呼ばれています"
                    "ジェーン・ドウ"
                    文: "私のフルネームはドクター・メレディス・グレイです"
                    "ドクター・メレディス・グレイ"

                    1. 文: "私の生年月日は2024年6月2日です"
                    - "2024-06-02"
                    2. 文: "えーっと...確認させてください....うーん！1983年8月2日です"
                    - "1983-08-02"
                    3. 文: "2009年7月3日"
                    - "2009-07-03"
                    4. 文: "私は1990年8月15日に生まれました"
                    - "1990-08-15"
                    5. 文: "彼女の誕生日は1985年12月25日です"
                    - "1985-12-25"
                    6. 文: "12月25日"
                    - "XXXX-12-25"
                    7. 文: "誕生日は5月6日です"
                    - "XXXX-05-06"

                    文: "私のポリシー番号は989898です"
                    "989898"
                    文: "989898"
                    "989898"
                    文: "ポリシーID: 123456789"
                    "123456789"
                    文: "私のポリシー番号はAB1234567ですのでご確認ください"
                    "AB1234567"

                    上記のルールに従い、抽出された値が正確であることを確認してください

                    ここに処理する文があります:
                    文: "{context}"
                    """,
                ),
            ]
            chat = ChatOpenAI(
                model="gpt-4o-mini", api_key=self.O_API_KEY, temperature=0
            )
            chat_template = ChatPromptTemplate.from_messages(prompt_and_context)
            message = chat_template.format_messages(context=japText)
            ai_resp = chat.invoke(message)
            return ai_resp.content

        except Exception as e:
            return str(e)
        



    def extraction_Intent_JP(self, japText):

        try:
            prompt_and_context = [
                (
                    "system",
                    """あなたは評判の良い病院で働く医療アシスタントで、患者の懸念を要約することを専門としています。あなたの任務は、患者の説明を正確に特定し、適切な医療用語に翻訳することです。

                        例:

                            発信者: "最近、気分がとても落ち込んでいて、以前好きだったことに興味が持てなくなっています。"
                            "大うつ病性障害"

                            発信者: "腕全体に赤くてかゆい変な発疹があります。"
                            "接触皮膚炎"

                            発信者: "パンを食べると、時々お腹がとても痛くなります。"
                            "グルテン不耐症（セリアック病）"

                            発信者: "どこで水を手に入れられますか？"
                            "null"

                            それでは、次の発信者の説明を適切に処理してください。
                            発信者: "{context}"
                    """,
                ),
            ]
            chat = ChatOpenAI(
                model="gpt-4o-mini", api_key=self.O_API_KEY, temperature=0
            )
            chat_template = ChatPromptTemplate.from_messages(prompt_and_context)
            message = chat_template.format_messages(context=japText)
            ai_resp = chat.invoke(message)
            return ai_resp.content

        except Exception as e:
            return str(e)

