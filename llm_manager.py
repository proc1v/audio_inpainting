import os
import json
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from custom_prompts import few_shot_prompt_template


class LLM_Manager:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = OpenAI(api_key=self.api_key, temperature=0)
        self.entity_extracting_prompt = PromptTemplate(
            input_variables=["input"], template=few_shot_prompt_template
        )
        self.llm_chain = self._initialize_llm_chain()

    def _initialize_llm_chain(self):
        return LLMChain(prompt=self.entity_extracting_prompt, llm=self.llm)

    def run_chain(self, input_text):
        response = self.llm_chain.run(input=input_text)
        try:
            response = json.loads(response)
            return response
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None


# Example usage
if __name__ == "__main__":
    llm_manager = LLM_Manager()
    input_text = "Inpaint the gray rock and black sport car from the image"
    response = llm_manager.run_chain(input_text)
    if response:
        print(response)
    else:
        print("Chain execution failed.")
