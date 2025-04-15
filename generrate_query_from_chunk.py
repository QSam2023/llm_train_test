import json
import random
from llm_api import volcengine_llm_api


prompt_template = """
你是一名专业的金融分析师和IPO尽调专家。

请根据给定的公司，根据给定的参考内容，生成可能出现的对应问题，

这些问题应符合如下内容
- 可以从给定的参考内容，查到对应的问题答案
- 问题应具体且清晰，不要过于宽泛或笼统

要求：
- 问题数量不少于5个
- 通过列表的形式给出生成问题，格式如下 'question'\n'question'

示例如下：
{question_examples}

给定的公司名称: {company_name}

给定的参考内容:
{reference_content}

"""


class GenQuestion:
    def __init__(self):
        self.question_examples = []
        with open("gen_data/question_v0", "r") as f:
            for line in f:
                self.question_examples.append(line.strip())

    def generate_prompt(self, company_name, reference_content):
        examples = random.sample(self.question_examples, 10)
        prompt = prompt_template.format(
            question_examples="\n".join(examples),
            company_name=company_name,
            reference_content=reference_content,
        )
        return prompt
    
    def llm_api_get_question(self, company_name, reference_content):
        prompt = self.generate_prompt(company_name, reference_content)
        result = volcengine_llm_api(prompt)
        try:
            question_list = [x.strip("'") for x in result.split("\n")]
            return question_list
        except:
            return []

if __name__ == "__main__":
    gen_question = GenQuestion()
    company_name = "技源集团股份有限公司"
    reference_content = """
    1、采购模式
    公司采购的主要原材料包括次氯酸钠、二丙酮醇、粗品盐酸盐、软骨粉等，
    市场供应相对较为充足，可选供应商较多。公司以保障产品质量和安全为首要采
    购原则，制定了《采购管理规范》《供应商管理规范》等一系列科学完善的采购
    管理制度，建立了安全稳定的供应商管理体系，保证采购物料的充足完备，保障
    原辅料储备和正常生产运营活动，持续优化公司物资管理综合水平，实现从供应
    商选择、价格谈判到质检入库全过程的有效管理。
    2、生产模式
    公司主要采取“以销定产”与“合理库存”相结合的生产模式，结合公司销
    售计划和库存的实际情况，合理组织生产活动，提高公司的营运效率。公司生产
    部门根据销售部门上报的销售计划、客户订单和发货计划，编制年度、月度及每
    周的具体生产计划，计算生产用料需求，经分管领导批准后组织实施生产活动。
    3、销售模式
    报告期内，公司主要采取直销模式向国内外膳食营养补充领域品牌客户进行
    产品销售。公司广泛采用自主拓展、客户拜访、行业交流、口碑管理等多种形式
    开发客户资源。公司一般需要通过客户严格的资质认证后才能进入其合格供应商
    体系，且需要通过客户定期的考核、评审等。通常情况下，公司进入客户合格供
    应商体系后，即与客户保持长期稳定的合作关系。
    """
    result = gen_question.llm_api_get_question(company_name, reference_content)
    print(result)