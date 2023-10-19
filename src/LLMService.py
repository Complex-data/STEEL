import os
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import openai
import pandas as pd
import yaml

from Util import setup_logger, get_project_root, storage_cached
from website.sender import Sender, MSG_TYPE_SEARCH_STEP, MSG_TYPE_OPEN_AI_STREAM

logger = setup_logger('LLMService')


class LLMService(ABC):
    def __init__(self, config):
        self.config = config

    def clean_response_text(self, response_text: str):
        return response_text.replace("\n", "")

    def get_prompt(self, search_text: str, gpt_input_text_df: pd.DataFrame):
        logger.info(f"OpenAIService.get_prompt. search_text: {search_text}, gpt_input_text_df.shape: {gpt_input_text_df.shape}")
        prompt_length_limit = 3000  # obsolete
        is_use_source = self.config.get('source_service').get('is_use_source')
        if is_use_source:
            prompt_engineering = f"\n\nAnswer the question '{search_text}' using above information with about 100 words:"
            prompt = ""
            for index, row in gpt_input_text_df.iterrows():
                prompt += f"""{row['text']}\n"""
            # limit the prompt length
            prompt = prompt[:prompt_length_limit]
            return prompt + prompt_engineering
        else:
            return f"\n\nAnswer the question '{search_text}' with about 100 words:"

    def get_prompt_v2(self, search_text: str, gpt_input_text_df: pd.DataFrame):
        logger.info(f"OpenAIService.get_prompt_v2. search_text: {search_text}, gpt_input_text_df.shape: {gpt_input_text_df.shape}")
        context_str = ""
        gpt_input_text_df = gpt_input_text_df.sort_values('url_id')
        url_id_list = gpt_input_text_df['url_id'].unique()
        for url_id in url_id_list:
            context_str += f"Source ({url_id})\n"
            for index, row in gpt_input_text_df[gpt_input_text_df['url_id'] == url_id].iterrows():
                context_str += f"{row['text']}\n"
            context_str += "\n"
        prompt_length_limit = 3000 # obsolete
        context_str = context_str[:prompt_length_limit]
        prompt = \
            f"""
Answer with 100 words for the question below based on the provided sources using a scientific tone. 
If the context is insufficient, reply "I cannot answer".
Use Markdown for formatting code or text.
Source:
{context_str}
Question: {search_text}
Answer:
"""
        return prompt

    def get_prompt_v3(self, search_text: str, gpt_input_text_df: pd.DataFrame):
        language = self.config.get('general').get('language')
        if not self.config.get('source_service').get('is_use_source'):
            prompt = \
                f"""
Instructions: Write a comprehensive reply to the given query.  
If the context is insufficient, reply "I cannot answer".
Query: {search_text}
"""
            return prompt

        logger.info(f"OpenAIService.get_prompt_v3. search_text: {search_text}, gpt_input_text_df.shape: {gpt_input_text_df.shape}")
        context_str = ""
        for _, row_url in gpt_input_text_df[['url_id', 'url']].drop_duplicates().iterrows():
            domain = urlparse(row_url['url']).netloc.replace('www.', '')
            context_str += f"Source [{row_url['url_id']}] {domain}\n"
            for index, row in gpt_input_text_df[(gpt_input_text_df['url_id'] == row_url['url_id']) & gpt_input_text_df['in_scope']].iterrows():
                context_str += f"{row['text']}\n"
            context_str += "\n\n"
        prompt_length_limit = self.config.get('llm_service').get('openai_api').get('prompt').get('prompt_length_limit')
        context_str = context_str[:prompt_length_limit]
        prompt = \
            f"""
Web search result:
{context_str}

Instructions: Using the provided web search results, write a comprehensive reply to the given query. 
Make sure to cite results using [number] notation after the reference.
If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.
Answer in language: {language}
If the context is insufficient, reply "I cannot answer because my reference sources don't have related info" in language {language}.
Query: {search_text}
"""
        return prompt

    def get_prompt_fc(self, search_text: str, gpt_input_text_df: pd.DataFrame):
        language = self.config.get('general').get('language')
        if not self.config.get('source_service').get('is_use_source'):
            prompt = \
                f"""
            Instructions: Analyze the given query and use your own knowledge to determine if the claim is true or false. 
            Query: {search_text}
            Answer: [true/false]
            Try your best and only anwser "true" or "false" based on your knowledge. DO NOT reply any other words.
            """
            return prompt

        logger.info(
            f"OpenAIService.get_prompt_fc. search_text: {search_text}, gpt_input_text_df.shape: {gpt_input_text_df.shape}")
        context_str = ""
        for _, row_url in gpt_input_text_df[['url_id', 'url']].drop_duplicates().iterrows():
            domain = urlparse(row_url['url']).netloc.replace('www.', '')
            context_str += f"Source [{row_url['url_id']}] {domain}\n"
            for index, row in gpt_input_text_df[
                (gpt_input_text_df['url_id'] == row_url['url_id']) & gpt_input_text_df['in_scope']].iterrows():
                context_str += f"{row['text']}\n"
            context_str += "\n\n"
        prompt_length_limit = self.config.get('llm_service').get('openai_api').get('prompt').get('prompt_token_limit')
        context_str = context_str[:prompt_length_limit]
        prompt = \
            f"""
        

        Instructions: Based on the provided web search results, analyze the given query and determine whether the claim is true or false..

Claim: {search_text}
Web search result:
{context_str}


Let's think step by step and judge if the claim is true or false.
If you have no idea whether it is true or false, just guest one.
        """
        return prompt



    def get_prompt_fc_zh(self, search_text: str, gpt_input_text_df: pd.DataFrame):
        language = self.config.get('general').get('language')
        if not self.config.get('source_service').get('is_use_source'):
            prompt = \
                f"""
            指令: 分析给定的声明，并使用您自己的知识来确定声明是真是假。 
            声明: {search_text}
            Answer: [true/false]
            Try your best and only anwser "true" or "false" based on your knowledge. DO NOT reply any other words.
            """
            return prompt

        logger.info(
            f"OpenAIService.get_prompt_fc. search_text: {search_text}, gpt_input_text_df.shape: {gpt_input_text_df.shape}")
        context_str = ""
        for _, row_url in gpt_input_text_df[['url_id', 'url']].drop_duplicates().iterrows():
            domain = urlparse(row_url['url']).netloc.replace('www.', '')
            context_str += f"Source [{row_url['url_id']}] {domain}\n"
            for index, row in gpt_input_text_df[
                (gpt_input_text_df['url_id'] == row_url['url_id']) & gpt_input_text_df['in_scope']].iterrows():
                context_str += f"{row['text']}\n"
            context_str += "\n\n"
        prompt_length_limit = self.config.get('llm_service').get('openai_api').get('prompt').get(
            'prompt_token_limit')
        context_str = context_str[:prompt_length_limit]
        prompt = \
            f"""


        指令：根据提供的网络搜索结果，分析给定的声明，判断声明是真是假。


请记住答案需要按规定形式填写。
如果你认为信息足够，则如果该声明为真，则回复我“Answer：0”；如果该声明为假，则回复我“Answer：1”。
如果你认为需要更多信息，请回复我“Answer：1”，因为此次检索判断已失败。   
请根据以下格式提供对声明进行判断：

Answer：[0/1]
只需提供数字
声明: {search_text}
网络搜索结果:
{context_str}


请记住答案需要按规定形式填写。
        """

        return prompt

    def get_prompt_qg(self, search_text: str, gpt_input_text_df: pd.DataFrame):
        language = self.config.get('general').get('language')


        logger.info(
            f"OpenAIService.get_prompt_qg. search_text: {search_text}")



        context_str = ""
        for _, row_url in gpt_input_text_df[['url_id', 'url']].drop_duplicates().iterrows():
            domain = urlparse(row_url['url']).netloc.replace('www.', '')
            context_str += f"Source [{row_url['url_id']}] {domain}\n"
            for index, row in gpt_input_text_df[
                (gpt_input_text_df['url_id'] == row_url['url_id']) & gpt_input_text_df['in_scope']].iterrows():
                context_str += f"{row['text']}\n"
            context_str += "\n\n"
        prompt_length_limit = self.config.get('llm_service').get('openai_api').get('prompt').get(
            'prompt_token_limit')
        context_str = context_str[:prompt_length_limit]

        prompt = \
            f"""
Instruction: Judging what information needs to be searched according to the given content, and giving the query as a string array. The query you give is used to search engine to get incremental information.
"claim": {search_text}
"Previous source information:{context_str}"
query:["query_1"]
such as:query: ["Ohio worker retraining funding rankings over time"]
Remember query must be a string array.
        """


        return prompt

    def get_prompt_qg_zh(self, search_text: str, gpt_input_text_df: pd.DataFrame):
        language = self.config.get('general').get('language')

        logger.info(
            f"OpenAIService.get_prompt_qg. search_text: {search_text}")

        context_str = ""
        for _, row_url in gpt_input_text_df[['url_id', 'url']].drop_duplicates().iterrows():
            domain = urlparse(row_url['url']).netloc.replace('www.', '')
            context_str += f"Source [{row_url['url_id']}] {domain}\n"
            for index, row in gpt_input_text_df[
                (gpt_input_text_df['url_id'] == row_url['url_id']) & gpt_input_text_df['in_scope']].iterrows():
                context_str += f"{row['text']}\n"
            context_str += "\n\n"
        prompt_length_limit = self.config.get('llm_service').get('openai_api').get('prompt').get(
            'prompt_token_limit')
        context_str = context_str[:prompt_length_limit]

        prompt = \
            f"""
指令：根据给定的内容判断要验证声明还需要查找哪些信息，并将查询内容以字符串数组的形式给出。 所给的query将用于搜索引擎以获取增量信息。
“声明”：{search_text}
“以前的来源信息：{context_str}”
query：[“query_1”]
如：查询：[“俄亥俄州工人再培训资金随时间的排名”]
请记住查询必须是字符串数组。
        """

        return prompt

    # step check
    def get_prompt_sc(self, search_text: str, gpt_input_text_df=None, compressed_info_content=None):
        language = self.config.get('general').get('language')

        logger.info(
            f"OpenAIService.get_prompt_sc. search_text: {search_text}, gpt_input_text_df.shape: {gpt_input_text_df.shape}")
        context_str = ""
        for _, row_url in gpt_input_text_df[['url_id', 'url']].drop_duplicates().iterrows():
            domain = urlparse(row_url['url']).netloc.replace('www.', '')
            context_str += f"Source [{row_url['url_id']}] {domain}\n"
            for index, row in gpt_input_text_df[
                (gpt_input_text_df['url_id'] == row_url['url_id']) & gpt_input_text_df['in_scope']].iterrows():
                context_str += f"{row['text']}\n"
            context_str += "\n\n"

        prompt_length_limit = self.config.get('llm_service').get('openai_api').get('prompt').get('prompt_token_limit')

        context_str = context_str[:prompt_length_limit]
        prompt = \
            f"""

Instructions: Based on the provided web search results, analyze whether the information has enough evidence to
decide whether the claim is true or false.

If you think information is enough, respond "0" if this cliam is true, respond "1" if this claim is false.
If you think the claim is partially true, respond "0".
If you think you need more information, respond '2' this number, and tell me what you miss and what should you search;
and I need you to help me extract key evidence from the information obtained this time and compress it into one or two 
sentences for subsequent use.
Please provide your evaluation of the claim based on the following format:

Explanation: [Explain why you make this judgement.]
Answer: [0/1/2]
If Answer is 0 or 1:
Just provide the number
If Answer is 2:
Missing info: [Description of missing information]
Query: ["Query 1", "Query 2", ...]]
Compressed information: [Compressed information]

Claim: {search_text}
Compressed_info: {compressed_info_content}
Web search result:
{context_str}

Remember to fill in all required fields based on the Answer value.
If you think information is enough, respond "0" if this cliam is true, respond "1" if this claim is false.
If you think you need more information, respond '2' this number, and tell me what you miss and what should you search in the format I specified.
And explain how confident you are (0~100%) Confirm that you are confident in your answer, and reply "Confidence: [0~100%].               
                """
        return prompt

    # step check zh
    def get_prompt_sc_zh(self, search_text: str, gpt_input_text_df=None):
        language = self.config.get('general').get('language')

        logger.info(
            f"OpenAIService.get_prompt_fc. search_text: {search_text}, gpt_input_text_df.shape: {gpt_input_text_df.shape}")
        context_str = ""
        for _, row_url in gpt_input_text_df[['url_id', 'url']].drop_duplicates().iterrows():
            domain = urlparse(row_url['url']).netloc.replace('www.', '')
            context_str += f"Source [{row_url['url_id']}] {domain}\n"
            for index, row in gpt_input_text_df[
                (gpt_input_text_df['url_id'] == row_url['url_id']) & gpt_input_text_df['in_scope']].iterrows():
                context_str += f"{row['text']}\n"
            context_str += "\n\n"
        prompt_length_limit = self.config.get('llm_service').get('openai_api').get('prompt').get(
            'prompt_token_limit')
        context_str = context_str[:prompt_length_limit]
        prompt = \
            f"""

指令: 根据提供的网络搜索结果,分析是否有足够的证据来判断该声明是真还是假。

请记住答案需要按规定形式填写。
如果你认为信息足够，则如果该声明为真，则回复我“Answer：0”；如果该声明为假，则回复我“Answer：1”。
如果你认为需要更多信息，请回复我“Answer：2”，并告诉我你缺少了什么信息以及按格式返回应该搜索的内容。   
请根据以下格式提供对声明进行判断：

Answer：[0/1/2]
如果答案为 0 或 1：
只需提供数字
如果答案是 2：
缺失信息：[缺失信息描述]
查询：[“查询1”，“查询2”，...]

例如：
Query：["Query 1", "Query 2", ...]

声明：{search_text}
网络搜索结果：
{context_str}

请记住答案需要按规定形式填写。请再次确认要核查的声明是什么。
如果你认为信息足够，则如果该声明为真，则回复我“Answer：0”；如果该声明为假，则回复我“Answer：1”。
如果你认为需要更多信息，请回复我“Answer：2”，并告诉我你缺少了什么信息以及按格式返回应该搜索的内容。
并说明你有多大的把握（0~100%）确认你对你的回答的有信心，并回复“Confidence: [0~100%]"""
        return prompt

    # quadratic answer
    def get_prompt_qa(self, first_response_text: str):
        language = self.config.get('general').get('language')

        logger.info(
            f"OpenAIService.get_prompt_qa. first_response_text: {first_response_text}")
        prompt_length_limit = self.config.get('llm_service').get('openai_api').get('prompt').get(
            'prompt_token_limit')
        context_str = first_response_text[:prompt_length_limit]
        prompt = \
            f"""

Instruction: I will provide you with a paragraph that contains a judgment on a claim. 
Your task is to analyze the intention of this paragraph. If the paragraph considers the claim to be true, 
reply with ‘Answer: 0’. If the paragraph considers the statement to be false, reply with ‘Answer: 1’. 
If the paragraph considers the statement to be neither true nor false, reply with ‘Answer: 2’. 

Paragraph:{first_response_text}

Please remember to follow my instructions to reply.
"""
        return prompt

    # text summary
    def get_prompt_sm(self, first_response_text: str):
        language = self.config.get('general').get('language')

        logger.info(
            f"OpenAIService.get_prompt_sm. first_response_text: {first_response_text}")
        prompt_length_limit = self.config.get('llm_service').get('openai_api').get('prompt').get(
            'prompt_token_limit')
        context_str = first_response_text[:prompt_length_limit]
        prompt = \
            f"""
Please help me convert this news article into a concise statement that can be used to assess its authenticity. 
The output should be in the following format:
Claim:
Remember, the claim should be brief and to the point.
text: {first_response_text}
"""
        return prompt

    # response correction
    def get_prompt_rc(self, first_response_text: str):
        language = self.config.get('general').get('language')

        logger.info(
            f"OpenAIService.get_prompt_qa. first_response_text: {first_response_text}")
        prompt_length_limit = self.config.get('llm_service').get('openai_api').get('prompt').get(
            'prompt_token_limit')
        context_str = first_response_text[:prompt_length_limit]
        prompt = \
            f"""

I will give you the following information: [Explanation: [Explain why you make this judgment.]
Answer: [0/1/2]
If Answer is 0 or 1:
Just provide the number
If Answer is 2:
Missing info: [Description of missing information]
Query: ["Query 1", "Query 2", ...]] Your task is to help me determine whether the number after "Answer: " is consistent with the explanation, where "Answer: 0" means this cliam is true , "Answer: 1" means this claim is false. "Answer: 2" means need more information. If they are inconsistent, the explanation shall prevail and the Answer shall be corrected, and the answer with only Answer: changed shall be returned.
The paragraph that needs to be judged and changed: "{first_response_text}"
"""
        return prompt

    # vanilla
    def get_prompt_va(self, search_text: str, gpt_input_text_df=None):
        language = self.config.get('general').get('language')

        logger.info(
            f"OpenAIService.get_prompt_va. search_text: {search_text}")
        context_str = ""
        for _, row_url in gpt_input_text_df[['url_id', 'url']].drop_duplicates().iterrows():
            domain = urlparse(row_url['url']).netloc.replace('www.', '')
            context_str += f"Source [{row_url['url_id']}] {domain}\n"
            for index, row in gpt_input_text_df[
                (gpt_input_text_df['url_id'] == row_url['url_id']) & gpt_input_text_df['in_scope']].iterrows():
                context_str += f"{row['text']}\n"
            context_str += "\n\n"
        prompt_length_limit = self.config.get('llm_service').get('openai_api').get('prompt').get('prompt_token_limit')
        context_str = context_str[:prompt_length_limit]
        prompt = \
            f"""
Instructions: Based on the provided web search results, analyze whether the claim is true, false.


Claim: {search_text}
Web search result:
{context_str}


Answer: [true/false]

Remember to fill in all required fields based on your judgement. 
You must and can only choose one answer from true or false.
"""
        return prompt

    # vanilla
    def get_prompt_va_zh(self, search_text: str, gpt_input_text_df=None):
        language = self.config.get('general').get('language')

        logger.info(
            f"OpenAIService.get_prompt_va_zh. search_text: {search_text}")
        context_str = ""
        for _, row_url in gpt_input_text_df[['url_id', 'url']].drop_duplicates().iterrows():
            domain = urlparse(row_url['url']).netloc.replace('www.', '')
            context_str += f"Source [{row_url['url_id']}] {domain}\n"
            for index, row in gpt_input_text_df[
                (gpt_input_text_df['url_id'] == row_url['url_id']) & gpt_input_text_df['in_scope']].iterrows():
                context_str += f"{row['text']}\n"
            context_str += "\n\n"
        prompt_length_limit = self.config.get('llm_service').get('openai_api').get('prompt').get(
            'prompt_token_limit')
        context_str = context_str[:prompt_length_limit]
        prompt = \
            f"""
Instructions: Based on the provided web search results, analyze whether the claim is true, false.
指令：根据所提供的网页搜索结果，判断声明是真还是假

Claim: {search_text}
Web search result:
{context_str}


Answer: [true/false]

Remember to fill in all required fields based on your judgement. 
You must and can only choose one answer from true or false.
"""
        return prompt

    # compress
    def get_prompt_qa(self, search_text: str, first_response_text: str):
        language = self.config.get('general').get('language')

        logger.info(
            f"OpenAIService.get_prompt_qa. first_response_text: {first_response_text}")
        prompt_length_limit = self.config.get('llm_service').get('openai_api').get('prompt').get(
            'prompt_token_limit')
        context_str = first_response_text[:prompt_length_limit]
        prompt = \
            f"""

Instruction: The information obtained from this search is not enough to determine whether the claim is true or false, so I need you to help me extract key evidence from the information obtained this time and compress it into one or two sentences for subsequent use.
"Statement": {search_text}
"Web search results": {context_str}
"Compressed information": [Compressed information]
""
Remember to follow the format for output.
"""
        return prompt


    @abstractmethod
    def call_api(self, prompt):
        pass


class OpenAIService(LLMService):
    def __init__(self, config, sender: Sender = None):
        super().__init__(config)
        self.sender = sender
        open_api_key = config.get('llm_service').get('openai_api').get('api_key')
        if open_api_key is None:
            raise Exception("OpenAI API key is not set.")
        openai.api_key = open_api_key

    @storage_cached('openai', 'prompt')
    def call_api(self, prompt: str):
        if self.sender is not None:
            self.sender.send_message(msg_type=MSG_TYPE_SEARCH_STEP, msg='Calling OpenAI API ...')

        openai_api_config = self.config.get('llm_service').get('openai_api')
        model = openai_api_config.get('model')
        is_stream = openai_api_config.get('stream')
        logger.info(f"OpenAIService.call_api. model: {model}, len(prompt): {len(prompt)}")

        if model in ['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-16k']:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    temperature=openai_api_config.get('temperature'),
                    top_p=openai_api_config.get('top_p'),
                    messages=[
                        #{"role": "system", "content": "You are a helpful search engine."},      # role
                        {"role": "system", "content": "You are a fact-checker. Your task is collect enough information \
                        to determine whether a claim is true or false."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=is_stream
                )
            except Exception as ex:
                raise ex

            if is_stream:
                collected_messages = []
                # iterate through the stream of events
                for chunk in response:
                    chunk_message = chunk['choices'][0]['delta'].get("content", None)  # extract the message
                    if chunk_message is not None:
                        if self.sender is not None:
                            self.sender.send_message(msg_type=MSG_TYPE_OPEN_AI_STREAM, msg=chunk_message)
                        collected_messages.append(chunk_message)  # save the message

                full_reply_content = ''.join(collected_messages)
                return full_reply_content
            else:
                return response.choices[0].message.content
        else:
            try:
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=openai_api_config.get('max_tokens'),
                    temperature=openai_api_config.get('temperature'),
                    top_p=openai_api_config.get('top_p'),
                )
            except Exception as ex:
                raise ex
            return self.clean_response_text(response.choices[0].text)


class GooseAIService(LLMService):
    def __init__(self, config, sender: Sender = None):
        super().__init__(config)
        self.sender = sender
        goose_api_key = config.get('goose_ai_api').get('api_key')
        if goose_api_key is None:
            raise Exception("Goose API key is not set.")
        openai.api_key = goose_api_key
        openai.api_base = config.get('goose_ai_api').get('api_base')

    @storage_cached('gooseai', 'prompt')
    def call_api(self, prompt: str, sender: Sender = None):
        if self.sender is not None:
            self.sender.send_message(msg_type=MSG_TYPE_SEARCH_STEP, msg='Calling gooseAI API ...')
        logger.info(f"GooseAIService.call_openai_api. len(prompt): {len(prompt)}")
        goose_api_config = self.config.get('goose_ai_api')
        try:
            response = openai.Completion.create(
                engine=goose_api_config.get('model'),
                prompt=prompt,
                max_tokens=goose_api_config.get('max_tokens'),
                # stream=True
            )
        except Exception as ex:
            raise ex
        return self.clean_response_text(response.choices[0].text)


class LLMServiceFactory:
    @staticmethod
    def create_llm_service(config, sender: Sender = None) -> LLMService:
        provider = config.get('llm_service').get('provider')
        if provider == 'openai':
            return OpenAIService(config, sender)
        elif provider == 'goose_ai':
            return GooseAIService(config, sender)
        else:
            logger.error(f'LLM Service for {provider} is not yet implemented.')
            raise NotImplementedError(f'LLM Service - {provider} - not is supported')


if __name__ == '__main__':
    # Load config
    with open(os.path.join(get_project_root(), 'src/config/config.yaml'), encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        service_factory = LLMServiceFactory()
        service = service_factory.create_llm_service(config)
        prompt = """
        """
        # response_text = service.call_openai_api('What is ChatGPT')
        response_text = service.call_api(prompt)
        print(response_text)
