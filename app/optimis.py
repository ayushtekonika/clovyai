from langchain_groq import ChatGroq
import os
import json
from dotenv import load_dotenv
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import JinaEmbeddings
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from typing import Optional

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
jina_key = os.getenv("JINA_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    streaming=True,
    max_retries=2,
    # other params...
)

llm2= ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    streaming=True,
    max_retries=2,
    # other params...
)

with open('questionMapping.json', 'r') as file:
    jsonData = json.load(file)

questionSetStrc1= jsonData['questionSet1']
questionSetStrc2= jsonData['questionSet2']

def get_summary(query: str) -> str:
    
    summaryPrompt = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a medical expert AI agent using the LLAMA model. Your task is to summarize the patient's query in key points, providing a concise, reliable summary that is easy to understand. Output only the key points—no introduction, explanation, or extra commentary. Provide the summary as a paragraph.

        Example 1:
        Query: I’ve been experiencing headaches for the past week, especially after using my computer for long hours. It’s usually a dull ache around my forehead and sometimes spreads to my temples. I tried taking painkillers, but they don’t seem to help much. I also noticed my eyes feel strained, and the room sometimes feels too bright.

        Summary:

        The patient reports having headaches for the past week, worsening with prolonged computer use. The headaches, characterized by a dull ache around the forehead extending to the temples, are unresponsive to painkillers. Symptoms also include eye strain and light sensitivity.
        Example 2:
        Query: Lately, I’ve been feeling fatigued, even after a full night's sleep. I don’t seem to have the energy I used to, and I’ve noticed that I get short of breath when walking up stairs. I also experienced some mild chest discomfort once or twice but didn’t think much of it.

        Summary:

        The patient has been experiencing fatigue even after a full night's sleep, along with low energy levels. They report shortness of breath while climbing stairs and have felt mild chest discomfort on several occasions.
        <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{""" + query + """}\n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    messages = summaryPrompt
    
    response = llm.invoke(messages)
    if hasattr(response, 'content') and response.content:
        return response.content
    else:
        return ""
    
def stringToJson(answers, questionSet):
    response_list = []
    for key, question_info in questionSet.items():
        question = question_info.get("question")
        question_id = question_info.get("id")
        if isinstance(answers, dict):
            answer = answers.get(key)
        else:
            answer = getattr(answers, key)
        
        response_list.append({
            "question": question,
            "id": question_id,
            "answer": answer
        })
    return response_list


class QuestionToAnswer(BaseModel):
    allergies: Optional[str] = Field(default="NA", description="Do you have allergies? (Yes/No/NA)")
    anemia: Optional[str] = Field(default="NA", description="Do you have anemia? (Yes/No/NA)")
    anxiety: Optional[str] = Field(default="NA", description="Do you experience anxiety? (Yes/No/NA)")
    arthritis: Optional[str] = Field(default="NA", description="Do you have arthritis? (Yes/No/NA)")
    asthma: Optional[str] = Field(default="NA", description="Do you have asthma? (Yes/No/NA)")
    autoimmune_disorder: Optional[str] = Field(default="NA", description="Do you have an autoimmune disorder? (Yes/No/NA)")
    diagnosed_with_cancer: Optional[str] = Field(default="NA", description="Have you been diagnosed with cancer? (Yes/No/NA)")
    cardiac_conditions: Optional[str] = Field(default="NA", description="Do you have any cardiac conditions? (Yes/No/NA)")
    cardiac_pacemaker: Optional[str] = Field(default="NA", description="Do you have a cardiac pacemaker? (Yes/No/NA)")
    chemical_dependency: Optional[str] = Field(default="NA", description="Are you dealing with chemical dependency? (Yes/No/NA)")
    circulation_problems: Optional[str] = Field(default="NA", description="Do you have circulation problems? (Yes/No/NA)")
    diagnosed_with_covid19: Optional[str] = Field(default="NA", description="Have you been diagnosed with COVID-19? (Yes/No/NA)")
    currently_pregnant: Optional[str] = Field(default="NA", description="Are you currently pregnant? (Yes/No/NA)")
    suffer_from_depression: Optional[str] = Field(default="NA", description="Do you suffer from depression? (Yes/No/NA)")
    diabetes: Optional[str] = Field(default="NA", description="Do you have diabetes? (Yes/No/NA)")
    dizzy_spells: Optional[str] = Field(default="NA", description="Do you experience dizzy spells? (Yes/No/NA)")
    emphysema_or_bronchitis: Optional[str] = Field(default="NA", description="Do you have emphysema or bronchitis? (Yes/No/NA)")
    fibromyalgia: Optional[str] = Field(default="NA", description="Do you have fibromyalgia? (Yes/No/NA)")
    fractures: Optional[str] = Field(default="NA", description="Have you had any fractures? (Yes/No/NA)")
    gallbladder_problems: Optional[str] = Field(default="NA", description="Do you have gallbladder problems? (Yes/No/NA)")
    frequent_headaches: Optional[str] = Field(default="NA", description="Do you experience frequent headaches? (Yes/No/NA)")
    hearing_impairment: Optional[str] = Field(default="NA", description="Do you have hearing impairment? (Yes/No/NA)")
    diagnosed_with_hepatitis: Optional[str] = Field(default="NA", description="Have you been diagnosed with hepatitis? (Yes/No/NA)")
    high_cholesterol: Optional[str] = Field(default="NA", description="Do you have high cholesterol? (Yes/No/NA)")
    high_or_low_blood_pressure: Optional[str] = Field(default="NA", description="Do you have high or low blood pressure? (Yes/No/NA)")
    diagnosed_with_hiv_aids: Optional[str] = Field(default="NA", description="Have you been diagnosed with HIV/AIDS? (Yes/No/NA)")
    incontinence: Optional[str] = Field(default="NA", description="Do you experience incontinence? (Yes/No/NA)")

class QuestionToAnswer2(BaseModel):
    kidney_problems: Optional[str] = Field(default="NA", description="Do you have kidney problems? (Yes/No/NA)")
    metal_implants: Optional[str] = Field(default="NA", description="Do you have any metal implants? (Yes/No/NA)")
    diagnosed_with_mrsa: Optional[str] = Field(default="NA", description="Have you been diagnosed with MRSA? (Yes/No/NA)")
    multiple_sclerosis: Optional[str] = Field(default="NA", description="Do you have multiple sclerosis? (Yes/No/NA)")
    muscular_disease: Optional[str] = Field(default="NA", description="Do you have a muscular disease? (Yes/No/NA)")
    diagnosed_with_osteoporosis: Optional[str] = Field(default="NA", description="Have you been diagnosed with osteoporosis? (Yes/No/NA)")
    parkinsons_disease: Optional[str] = Field(default="NA", description="Do you have Parkinson's disease? (Yes/No/NA)")
    diagnosed_with_rheumatoid_arthritis: Optional[str] = Field(default="NA", description="Have you been diagnosed with rheumatoid arthritis? (Yes/No/NA)")
    history_of_seizures: Optional[str] = Field(default="NA", description="Do you have a history of seizures? (Yes/No/NA)")
    smoker: Optional[str] = Field(default="NA", description="Are you a smoker? (Yes/No/NA)")
    speech_problems: Optional[str] = Field(default="NA", description="Do you experience speech problems? (Yes/No/NA)")
    had_a_stroke: Optional[str] = Field(default="NA", description="Have you had a stroke? (Yes/No/NA)")
    thyroid_disease: Optional[str] = Field(default="NA", description="Do you have thyroid disease? (Yes/No/NA)")
    diagnosed_with_tuberculosis: Optional[str] = Field(default="NA", description="Have you been diagnosed with tuberculosis? (Yes/No/NA)")
    vision_problems: Optional[str] = Field(default="NA", description="Do you have vision problems? (Yes/No/NA)")
    medical_precautions: Optional[str] = Field(default="NA", description="Any Medical Precautions? (Text)")
    injured_from_fall_last_year: Optional[str] = Field(default="NA", description="Have you been injured as a result of a fall in the past year? (Yes/No/NA)")
    two_or_more_falls_last_year: Optional[str] = Field(default="NA", description="Have you had two or more falls in the last year? (Yes/No/NA)")
    risk_for_falls: Optional[str] = Field(default="NA", description="Are you considered to be at risk for falls? (Yes/No/NA)")
    surgical_history_body_region: Optional[str] = Field(default="NA", description="What is the body region involved in your surgical history? (Text)")
    deny_taking_medications: Optional[str] = Field(default="NA", description="Do you deny taking medications? (Yes/No/NA)")
    medications_scanned_into_file: Optional[str] = Field(default="NA", description="Have your medications been scanned into the file? (Yes/No/NA)")
    reviewed_current_medications: Optional[str] = Field(default="NA", description="Have you reviewed your current medications, including name, dosage, frequency, and route? (Yes/No/NA)")
    current_medication_name: Optional[str] = Field(default="NA", description="What is the name of the medication you are currently taking? (Text)")

sampleResponse1 = {
    "allergies": "NA",
    "anemia": "NA",
    "anxiety": "NA",
    "arthritis": "NA",
    "asthma": "NA",
    "autoimmune_disorder": "NA",
    "diagnosed_with_cancer": "NA",
    "cardiac_conditions": "NA",
    "cardiac_pacemaker": "NA",
    "chemical_dependency": "NA",
    "circulation_problems": "NA",
    "diagnosed_with_covid19": "NA",
    "currently_pregnant": "NA",
    "suffer_from_depression": "NA",
    "diabetes": "NA",
    "dizzy_spells": "NA",
    "emphysema_or_bronchitis": "NA",
    "fibromyalgia": "NA",
    "fractures": "NA",
    "gallbladder_problems": "NA",
    "frequent_headaches": "NA",
    "hearing_impairment": "NA",
    "diagnosed_with_hepatitis": "NA",
    "high_cholesterol": "NA",
    "high_or_low_blood_pressure": "NA",
    "diagnosed_with_hiv_aids": "NA",
    "incontinence": "NA"
}
sampleResponse2 = {
    "kidney_problems": "NA",
    "metal_implants": "NA",
    "diagnosed_with_mrsa": "NA",
    "multiple_sclerosis": "NA",
    "muscular_disease": "NA",
    "diagnosed_with_osteoporosis": "NA",
    "parkinsons_disease": "NA",
    "diagnosed_with_rheumatoid_arthritis": "NA",
    "history_of_seizures": "NA",
    "smoker": "NA",
    "speech_problems": "NA",
    "had_a_stroke": "NA",
    "thyroid_disease": "NA",
    "diagnosed_with_tuberculosis": "NA",
    "vision_problems": "NA",
    "medical_precautions": "NA",
    "injured_from_fall_last_year": "NA",
    "two_or_more_falls_last_year": "NA",
    "risk_for_falls": "NA",
    "surgical_history_body_region": "NA",
    "deny_taking_medications": "NA",
    "medications_scanned_into_file": "NA",
    "reviewed_current_medications": "NA",
    "current_medication_name": "NA"
}

def extract_entity(query: str):

    prompt = PromptTemplate(
        template="""<|start_header_id|>system<|end_header_id|>
            You are responsible for analyzing medical information directly related to user queries. Your specific task is to extract and verify information solely from the provided medical summary to answer a predefined list of questions. For each question, especially those requiring a Yes/No response, ensure your answer strictly aligns with the facts presented in the summary. If a question cannot be answered with the available information, respond with 'NA'. It is crucial that your responses are derived exclusively from the provided text and do not include any assumptions, interpretations, or fictional elaborations.
            
            Here's the user's questionnaire:
            
            1. Do you have allergies? (Yes/No/NA)
            2. Do you have anemia? (Yes/No/NA)
            3. Do you experience anxiety? (Yes/No/NA)
            4. Do you have arthritis? (Yes/No/NA)
            5. Do you have asthma? (Yes/No/NA)
            6. Do you have an autoimmune disorder? (Yes/No/NA)
            7. Have you been diagnosed with cancer? (Yes/No/NA)
            8. Do you have any cardiac conditions? (Yes/No/NA)
            9. Do you have a cardiac pacemaker? (Yes/No/NA)
            10. Are you dealing with chemical dependency? (Yes/No/NA)
            11. Do you have circulation problems? (Yes/No/NA)
            12. Have you been diagnosed with COVID-19? (Yes/No/NA)
            13. Are you currently pregnant? (Yes/No/NA)
            14. Do you suffer from depression? (Yes/No/NA)
            15. Do you have diabetes? (Yes/No/NA)
            16. Do you experience dizzy spells? (Yes/No/NA)
            17. Do you have emphysema or bronchitis? (Yes/No/NA)
            18. Do you have fibromyalgia? (Yes/No/NA)
            19. Have you had any fractures? (Yes/No/NA)
            20. Do you have gallbladder problems? (Yes/No/NA)
            21. Do you experience frequent headaches? (Yes/No/NA)
            22. Do you have hearing impairment? (Yes/No/NA)
            23. Have you been diagnosed with hepatitis? (Yes/No/NA)
            24. Do you have high cholesterol? (Yes/No/NA)
            25. Do you have high or low blood pressure? (Yes/No/NA)
            26. Have you been diagnosed with HIV/AIDS? (Yes/No/NA)
            27. Do you experience incontinence? (Yes/No/NA)
            
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {query}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["query"]
    )
    
    prompt2 = PromptTemplate(
        template="""<|start_header_id|>system<|end_header_id|>
            You are responsible for analyzing medical information directly related to user queries. Your specific task is to extract and verify information solely from the provided medical summary to answer a predefined list of questions. For each question, especially those requiring a Yes/No response, ensure your answer strictly aligns with the facts presented in the summary. If a question cannot be answered with the available information, respond with 'NA'. It is crucial that your responses are derived exclusively from the provided text and do not include any assumptions, interpretations, or fictional elaborations.

            Here's the user's questionnaire:

            1. Do you have kidney problems? (Yes/No/NA)
            2. Do you have any metal implants? (Yes/No/NA)
            3. Have you been diagnosed with MRSA? (Yes/No/NA)
            4. Do you have multiple sclerosis? (Yes/No/NA)
            5. Do you have a muscular disease? (Yes/No/NA)
            6. Have you been diagnosed with osteoporosis? (Yes/No/NA)
            7. Do you have Parkinson's disease? (Yes/No/NA)
            8. Have you been diagnosed with rheumatoid arthritis? (Yes/No/NA)
            9. Do you have a history of seizures? (Yes/No/NA)
            10. Are you a smoker? (Yes/No/NA)
            11. Do you experience speech problems? (Yes/No/NA)
            12. Have you had a stroke? (Yes/No/NA)
            13. Do you have thyroid disease? (Yes/No/NA)
            14. Have you been diagnosed with tuberculosis? (Yes/No/NA)
            15. Do you have vision problems? (Yes/No/NA)
            16. Any Medical Precautions? (Text)
            17. Have you been injured as a result of a fall in the past year? (Yes/No/NA)
            18. Have you had two or more falls in the last year? (Yes/No/NA)
            19. Are you considered to be at risk for falls? (Yes/No/NA)
            20. What is the body region involved in your surgical history? (Text)
            21. Do you deny taking medications? (Yes/No/NA)
            22. Have your medications been scanned into the file? (Yes/No/NA)
            23. Have you reviewed your current medications, including name, dosage, frequency, and route? (Yes/No/NA)
            24. What is the name of the medication you are currently taking? (Text)

            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {query}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["query"]
    )

    structured_llm = llm2.with_structured_output(QuestionToAnswer)

    structured_llm2 = llm2.with_structured_output(QuestionToAnswer2)

    chain = prompt | structured_llm

    response = chain.invoke({"query": query})

    if response is None:
        response = sampleResponse1

    chain2 = prompt2 | structured_llm2

    response2 = chain2.invoke({"query": query})

    if response2 is None:
        response2 = sampleResponse2
    
    return stringToJson(response, questionSetStrc1) + stringToJson(response2, questionSetStrc2)


class VECTOR_CREATION:
    def __init__(self, jina_key):
        self.chunk_size=500 
        self.chunk_overlap=200
        self.jina_key=jina_key
        
    def vectorCreation(self, doc):

        loader = TextLoader(doc)
        loader.load()

        # loader = UnstructuredTextLoader('output.txt')
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=80)
        splits = text_splitter.split_documents(data)

        embeddings = JinaEmbeddings(
            jina_auth_token="jina_ee35ec59fede4457832af3a2fffbacaazLpsBj-lgE8SpfBkChq0DDZkyyuB", model_name='jina-embeddings-v2-base-en'
        )
        
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        return vectorstore

class RETRIEVER_LLM:
    def __init__(self):
        self.model = llm
    def retrieverLLM(self, vectorstore):
        
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        prompt = hub.pull("rlm/rag-prompt") + """Focus on the body part and the problems/description provided in the question. \n 
        1. Always provide the ICD10 code, \n
        2. Provide multiple ICD10 codes.\n
        3. Provide response in array of objects <code: <icd10 code>, description: <icd10 code description>> and don't send anything else and use double quotes for string\n"""

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.model
            | StrOutputParser()
        )

        return rag_chain
    
    def response(self, rag_chain, query):
        response = rag_chain.invoke(query)
        return response
    
vector_creator = VECTOR_CREATION(jina_key)  # Renamed instance
vectorstore = vector_creator.vectorCreation("icd10.txt")
retriever_llm = RETRIEVER_LLM()  # Renamed instance
rag_chain = retriever_llm.retrieverLLM(vectorstore)
 
def icd10(query: str):
    response = retriever_llm.response(rag_chain, query)
    return json.loads(response)

def getTopPattern(query:str):

    # Convert string to list of dictionaries (array of objects)
    data_list = json.loads(query)

    # Sort by percentage in descending order
    sorted_data = sorted(data_list, key=lambda x: x['percentage'], reverse=True)

    # Select the top item
    top_one = sorted_data[:1]

    # Output the result
    return top_one


def patterns(query: str):
    prompt = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        Given the following patterns and associated tests,
        evaluate each pattern type's suitability (on a scale of 0-100%) based on its relevance to diagnosing or assessing the specified condition.
        The input is structured in the format of "pattern": ["tests"].
        For each issue, rate each pattern type based on how well it addresses the issue criteria in the provided user document summary, considering factors like reliability, relevance, and specificity to the condition.
        Give response in Array[{"pattern": <pattern-name>, "percentage" : <percentage>}], don't provide any other information.

        Pattern and Test Types:

        Pattern: Shoulder Adhesive Capsulitis / Mobility Deficits:
        tests:
        Stiffness reported - shoulder region (gradual onset)
        Passive movement tests (shoulder)
        Active movement tests (shoulder)
        Accessory movement tests (shoulder)
        
        Pattern: Shoulder Instability / Coordination Impairments:
        tests:
        Onset mechanism - shoulder dislocation/subluxations
        Special tests - shoulder labral tests
        Special tests - shoulder instability tests
        Ligament integrity tests (shoulder)
        
        Pattern: Thoracic Outlet Syndrome / Shoulder and Arm Radiating Pain:
        tests:
        Paresthesia - upper limb
        Nerve tension tests (upper limb)
        Aggravating factors - limb positions that involve nerve tension
        Palpation - upper quarter nerve entrapment site (provocation reproduces symptoms)
        Neurological status (upper quarter)
        
        Pattern: Rotator Cuff Syndrome / Muscle Power Deficits:
        tests:
        Observation - arc of movement pain with shoulder motions
        Resistive tests (rotator cuff)
        Aggravating factors - repetitive overhead activity
        Special tests - shoulder impingement tests
        Palpation - rotator cuff (provocation reproduces symptoms)
        
        Task: For each issue criteria, rate only the corresponding pattern types in percentage (%), where 100% indicates the pattern is highly suitable for diagnosing/assessing the issue, and 0% indicates it is not suitable.Give response in Array[{"pattern": <pattern-name>, "percentage" : <percentage>}], don't provide any other information.
        <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{""" + query + """}\n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    response = llm.invoke(prompt)
    return getTopPattern(response.content)