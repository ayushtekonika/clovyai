from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

if groq_key is None:
    print(groq_key, "groq")
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
    
def extract_entity(query: str) -> str:
    extractionPrompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance of a retrieved document to a user's medical questionnaire. Your task is to extract information from the document and answer the user's questions based on the provided response options. For Yes/No questions, select one of the given options based on the document. For questions with no provided options, extract the relevant information from the document. \n
            Provide the answers in an array format with answer being a object { question: question that is mentioned, id: id mentioned beside question, answer: answer of the question }. If the document does not contain information relevant to a particular question, return "NA" as the answer. \n
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
            28. Do you have kidney problems? (Yes/No) (66e11f97c5ebd1848543110c)
            29. Do you have any metal implants? (Yes/No) (66e13affc5ebd1848543110e)
            30. Have you been diagnosed with MRSA? (Yes/No) (66e13b12c5ebd18485431110)
            31. Do you have multiple sclerosis? (Yes/No) (66e13b3cc5ebd18485431112)
            32. Do you have a muscular disease? (Yes/No) (66e13b6ac5ebd18485431114)
            33. Have you been diagnosed with osteoporosis? (Yes/No) (66e13b81c5ebd18485431116)
            34. Do you have Parkinson’s disease? (Yes/No) (66e13b9cc5ebd18485431118)
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
            48. What type of surgery have you undergone? (Text) (na)
            49. When did you have surgery? (Text) (na)
            50. Do you deny taking medications? (Yes/No/NA) (66dfcd174bef40d7e6577070)
            51. Have your medications been scanned into the file? (Yes/No/NA) (66dfcd2b4bef40d7e6577072)
            52. Have you reviewed your current medications, including name, dosage, frequency, and route? (Yes/No) (66dfcd3a4bef40d7e6577074)
            53. What is the name of the medication you are currently taking? (Text) (66e07216973e7e3e485f6d2a)
            54. What is the dosage and frequency of your medication intake? (Text) (na)
            55. What is the route of administration for your medication? (Text) (na)
            56. What is the reason for you taking your medication? (Text) (na)

            <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the retrieved document: \n\n{""" + query + """}\n\n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    messages = extractionPrompt
    
    response = llm.invoke(messages)
    if hasattr(response, 'content') and response.content:
        return response.content
    else:
        return ""