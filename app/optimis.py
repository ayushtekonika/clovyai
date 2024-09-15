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
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = os.getenv("LANGCHAIN_ENDPOINT")
# os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

if groq_key is None:
    os.environ["GROQ_API_KEY"] = groq_key

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

def get_summary(query: str) -> str:
    
    summaryPrompt = [
    (
        "system",
        "The human query that is provided is a conversation between a medical expert and a patient. so You are a medical expert AI agent summarizer equipped with comprehensive knowledge. Your primary function is to summarize the patient's query by providing accurate summary. Your responses are concise, reliable, and presented in an easy-to-understand format. Make sure the response is in key points",
    ),
    ("human", query)
    ]

    messages = summaryPrompt
    
    response = llm.invoke(messages)
    if hasattr(response, 'content') and response.content:
        return response.content
    else:
        return ""
    
def stringToJson(entity: str):
    # Given data
    data = entity

    # Remove the "break" statement and split the data into two parts
    data_parts = data.replace("break", "").split("[")
    data_parts = [part.strip() for part in data_parts if part.strip()]

    # Initialize an empty list to store the JSON data
    json_data = []

    # Process each part of the data
    for part in data_parts:
        # Remove the trailing comma and closing bracket
        part = part.rstrip(", ]")
        
        # Add the opening bracket and split into individual items
        items = "[" + part + "]"
        items = json.loads(items)
        
        # Add the items to the JSON data list
        json_data.extend(items)

    # Convert the JSON data list to a JSON string
    # json_string = json.dumps(json_data, indent=4)

    # Print the JSON string
    return json_data
    
def extract_entity(query: str):
    
    extractionPrompt1 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance of a retrieved document to a user's medical questionnaire. Your task is to extract information from the document and answer the user's questions based on the provided response options. For Yes/No questions, select one of the given options based on the document. For questions with no provided options, extract the relevant information from the document. \n
            Provide the answers in just a json array format with answer being a object { question: question that is mentioned, id: id mentioned beside question, answer: answer of the question }. If the document does not contain information relevant to a particular question, return "NA" as the answer. \n
            Here's the user's questionnaire: \n
            1. Do you have allergies? (Yes/No) (66dfc96f4bef40d7e657705b)
            2. Do you have anemia? (Yes/No) (66dfc9804bef40d7e657705d)
            3. Do you experience anxiety? (Yes/No) (66dfc9954bef40d7e657705f)
            4. Do you have arthritis? (Yes/No) (66e11bce848e9ed414b028cd)
            5. Do you have asthma? (Yes/No) (66e11bde848e9ed414b028cf)
            6. Do you have an autoimmune disorder? (Yes/No) (66e11c18c5ebd184854310e4)
            7. Have you been diagnosed with cancer? (Yes/No) (n66dfc9ed4bef40d7e6577061)
            8. Do you have any cardiac conditions? (Yes/No) (66e11c34c5ebd184854310e6)
            9. Do you have a cardiac pacemaker? (Yes/No) (66e11c64c5ebd184854310e8)
            10. Are you dealing with chemical dependency? (Yes/No) (66e11c78c5ebd184854310ea)
            11. Do you have circulation problems? (Yes/No) (66e11c8bc5ebd184854310ec)
            12. Have you been diagnosed with COVID-19? (Yes/No) (66dfc9fa4bef40d7e6577063)
            13. Are you currently pregnant? (Yes/No) (66e11cadc5ebd184854310ee)
            14. Do you suffer from depression? (Yes/No) (66e11d2fc5ebd184854310f0)
            15. Do you have diabetes? (Yes/No) (66e11d40c5ebd184854310f2)
            16. Do you experience dizzy spells? (Yes/No) (66e11d67c5ebd184854310f4)
            17. Do you have emphysema or bronchitis? (Yes/No) (66e11d8ac5ebd184854310f6)
            18. Do you have fibromyalgia? (Yes/No) (66e11d9ec5ebd184854310f8)
            19. Have you had any fractures? (Yes/No) (66e11db5c5ebd184854310fa)
            20. Do you have gallbladder problems? (Yes/No) (66e11dc6c5ebd184854310fc)
            21. Do you experience frequent headaches? (Yes/No) (66e11dd6c5ebd184854310fe)
            22. Do you have hearing impairment? (Yes/No) (66e11e08c5ebd18485431100)
            23. Have you been diagnosed with hepatitis? (Yes/No) (66e11e27c5ebd18485431102)
            24. Do you have high cholesterol? (Yes/No) (66e11e3bc5ebd18485431104)
            25. Do you have high or low blood pressure? (Yes/No) (66e11ebdc5ebd18485431106)
            26. Have you been diagnosed with HIV/AIDS? (Yes/No) (66e11f24c5ebd18485431108)
            27. Do you experience incontinence? (Yes/No) (66e11f6ac5ebd1848543110a)

            <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{""" + query + """}\n\n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    extractionPrompt2 = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance of a retrieved document to a user's medical questionnaire. Your task is to extract information from the document and answer the user's questions based on the provided response options. For Yes/No questions, select one of the given options based on the document. For questions with no provided options, extract the relevant information from the document. \n
            Provide the answers in just a json array format with answer being a object { question: question that is mentioned, id: id mentioned beside question, answer: answer of the question }. If the document does not contain information relevant to a particular question, return "NA" as the answer. \n
            Here's the user's questionnaire: \n
            28. Do you have kidney problems? (Yes/No) (66e11f97c5ebd1848543110c)
            29. Do you have any metal implants? (Yes/No) (66e13affc5ebd1848543110e)
            30. Have you been diagnosed with MRSA? (Yes/No) (66e13b12c5ebd18485431110)
            31. Do you have multiple sclerosis? (Yes/No) (66e13b3cc5ebd18485431112)
            32. Do you have a muscular disease? (Yes/No) (66e13b6ac5ebd18485431114)
            33. Have you been diagnosed with osteoporosis? (Yes/No) (66e13b81c5ebd18485431116)
            34. Do you have Parkinson's disease? (Yes/No) (66e13b9cc5ebd18485431118)
            35. Have you been diagnosed with rheumatoid arthritis? (Yes/No) (66e13bbec5ebd1848543111a)
            36. Do you have a history of seizures? (Yes/No) (66e13bd5c5ebd1848543111c)
            37. Are you a smoker? (Yes/No) (66e13be8c5ebd1848543111e)
            38. Do you experience speech problems? (Yes/No) (66e13c18c5ebd18485431120)
            39. Have you had a stroke? (Yes/No) (66e13c32c5ebd18485431122)
            40. Do you have thyroid disease? (Yes/No) (66e13c52c5ebd18485431124)
            41. Have you been diagnosed with tuberculosis? (Yes/No) (66e13c63c5ebd18485431126)
            42. Do you have vision problems? (Yes/No) (66e13c7bc5ebd18485431128)
            43. Any Medical Precautions? (Text) (66dfcaef4bef40d7e6577067)
            44. Have you been injured as a result of a fall in the past year? (Yes/No) (66dfcb774bef40d7e657706a)
            45. Have you had two or more falls in the last year? (Yes/No) (66dfcb964bef40d7e657706c)
            46. Are you considered to be at risk for falls? (Yes/No) (66dfcbb24bef40d7e657706e)
            47. What is the body region involved in your surgical history? (Text) (66e06f44973e7e3e485f6d1a)
            50. Do you deny taking medications? (Yes/No/NA) (66dfcd174bef40d7e6577070)
            51. Have your medications been scanned into the file? (Yes/No/NA) (66dfcd2b4bef40d7e6577072)
            52. Have you reviewed your current medications, including name, dosage, frequency, and route? (Yes/No) (66dfcd3a4bef40d7e6577074)
            53. What is the name of the medication you are currently taking? (Text) (66e07216973e7e3e485f6d2a)

            <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{""" + query + """}\n\n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    
    messages = extractionPrompt1
    
    response = llm.invoke(messages)
    
    messages2 = extractionPrompt2
        
    response2 = llm.invoke(messages2)
    
    finalResponse = response.content + " \n break \n " + response2.content
    
    # if hasattr(response, 'content') and response.content:
    return stringToJson(finalResponse)
    # else:
    #     return ""


class VECTOR_CREATION:
    def __init__(self):
        self.chunk_size=500 
        self.chunk_overlap=200
        
    def vectorCreation(self, doc):

        loader = TextLoader(doc)
        loader.load()

        # loader = UnstructuredTextLoader('output.txt')
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        splits = text_splitter.split_documents(data)

        # embeddings = JinaEmbeddings(
        #     jina_auth_token='jina_15520dc13bb84171854a95f2108a76abCgsuwjqyX2SrXgjLpOpdCPzfhees', model_name='jina-embeddings-v2-base-en'
        # )
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
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
    
vector_creator = VECTOR_CREATION()  # Renamed instance
vectorstore = vector_creator.vectorCreation("icd10.txt")
retriever_llm = RETRIEVER_LLM()  # Renamed instance
rag_chain = retriever_llm.retrieverLLM(vectorstore)
 
def icd10(query: str):
    response = retriever_llm.response(rag_chain, query)
    return json.loads(response)

def getTopPatter(query:str):

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
    return getTopPatter(response.content)