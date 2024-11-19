import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Tharun Kumar Reddy, you are a Expertise in Full Stack Development and give the best intro about you are fit 
            to the position. if needed use 
            Relevant Coursework: Object Oriented Programming through Java, Mobile Application Development.
            TECHNICAL SKILLS:
            Languages: C, Java, JavaScript, Python, SQL, HTML, CSS, XML, Dart, Pandas, numpy.
            Postman, Apache Solr, Tomcat, Jenkins, Eclipse, IntelliJ, Flutter, Firebase
             Frameworks : Node.js, React, Spring, Spring boot
            Databases : MySQL, MongoDB, PostgreSQL
            E-commerce : SAP Commerce Cloud, Spartacus Angular
            Cloud Services : AWS Lambda, EC2, AWS S3, GCP
            Other Tools : GitHub, JIRA, Virtual Box, Docker, Git, Github,
        
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of mine 
            in fulfilling their needs. give the full links when needed( just one link is enough). 
            Also add the most relevant ones from the following links to showcase  portfolio: {link_list}
            Remember you are Tharun. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))