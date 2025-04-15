import json
import random
from llm_api import volcengine_llm_api

prompt_template = """
你是一名专业的金融分析师和IPO尽调专家。

请根据给定的公司，根据其业务背景，对应的行业信息，生成相似的IPO尽调问题

这些问题应：

- 涉及公司业务模式、财务状况、风险因素、募资用途、治理结构、行业竞争等IPO关注的核心领域；
- 帮助投资者或分析师深入理解公司的潜在风险与机会；
- 问题应具体且清晰，不要过于宽泛或笼统。

要求：
- 问题数量不少于40个
- 通过列表的形式给出生成问题，格式如下 'question'\n'question'

示例如下：
{question_examples}

给定的公司名称: {company_name}
"""

class GenQuestion:
    def __init__(self):
        self.question_suffix = []
        with open("gen_data/question_base", "r") as f:
            for line in f:
                self.question_suffix.append(line.strip())
        self.question_examples = []
        with open("gen_data/question_v0", "r") as f:
            for line in f:
                self.question_examples.append(line.strip())
        

    def generate_query(self, company_name):
        suffix = random.sample(self.question_suffix, 1)[0]
        query = f"{company_name} {suffix}".replace("公司公司", "公司")
        return query
    
    def generate_prompt(self, company_name):
        examples = random.sample(self.question_examples, 10)
        prompt = prompt_template.format(
            question_examples="\n".join(examples),
            company_name=company_name
        )
        return prompt
    
    def llm_api_get_question(self, company_name):
        prompt = self.generate_prompt(company_name)
        result = volcengine_llm_api(prompt)
        print(result)
        try:
            question_list = [x.strip("'") for x in result.split("\n")]
            return question_list
        except:
            return []

if __name__ == "__main__":
    gen_question = GenQuestion()
    query = gen_question.generate_query("阿里巴巴")
    #print(query)
    prompt = gen_question.generate_prompt("阿里巴巴")
    #print(prompt)
    result = gen_question.llm_api_get_question("阿里巴巴")
    print(result)