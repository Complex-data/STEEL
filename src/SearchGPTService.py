import os
from my_logging import logger

import pandas as pd
import yaml

from FrontendService import FrontendService
from LLMService import LLMServiceFactory
from SemanticSearchService import BatchOpenAISemanticSearchService
from SourceService import SourceService
from Util import setup_logger, get_project_root, storage_cached
from website.sender import Sender

# logger = setup_logger('SearchGPTService')


class SearchGPTService:
    """
    SearchGPT app->service->child-service structure
    - (Try to) app import service, child-service inherit service

    SearchGPT class
    - SourceService
    -- BingService
    -- Doc/PPT/PDF Service
    - SemanticSearchModule
    - LLMService
    -- OpenAIService
    -- GooseAPIService
    - FrontendService

    """

    def __init__(self, ui_overriden_config=None, sender: Sender = None):
        with open(os.path.join(get_project_root(), 'src/config/config.yaml'), encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.overide_config_by_query_string(ui_overriden_config)
        self.validate_config()
        self.sender = sender

    def overide_config_by_query_string(self, ui_overriden_config):
        if ui_overriden_config is None:
            return
        for key, value in ui_overriden_config.items():
            if value is not None and value != '':
                # query_string is flattened (one level) while config.yaml is nested (two+ levels)
                # Any better way to handle this?
                if key == 'bing_search_subscription_key':
                    self.config['source_service']['bing_search']['subscription_key'] = value
                elif key == 'openai_api_key':
                    self.config['llm_service']['openai_api']['api_key'] = value
                elif key == 'is_use_source':
                    self.config['source_service']['is_use_source'] = False if value.lower() in ['false', '0'] else True
                elif key == 'llm_service_provider':
                    self.config['llm_service']['provider'] = value
                elif key == 'llm_model':
                    if self.config['llm_service']['provider'] == 'openai':
                        self.config['llm_service']['openai_api']['model'] = value
                    elif self.config['llm_service']['provider'] == 'goose_ai':
                        self.config['llm_service']['goose_ai_api']['model'] = value
                    else:
                        raise Exception(f"llm_model is not supported for llm_service_provider: {self.config['llm_service']['provider']}")
                elif key == 'language':
                    self.config['general']['language'] = value
                else:
                    # invalid query_string but not throwing exception first
                    pass

    def validate_config(self):
        if self.config['source_service']['is_enable_bing_search']:
            assert self.config['source_service']['bing_search']['subscription_key'], 'bing_search_subscription_key is required'
        if self.config['llm_service']['provider'] == 'openai':
            assert self.config['llm_service']['openai_api']['api_key'], 'openai_api_key is required'

    @storage_cached('web', 'search_text')
    def query_and_get_answer(self, search_text, prompt_choice='fc_en', text_df=pd.DataFrame()):
        if not text_df.empty:
            text_df = text_df
        else:
            source_module = SourceService(self.config, self.sender)
            # Use gpt to generate the queries
            # Concat previous df


            bing_text_df = source_module.extract_bing_text_df(search_text)
            doc_text_df = source_module.extract_doc_text_df(bing_text_df)
            text_df = pd.concat([bing_text_df, doc_text_df], ignore_index=True)
        print(f"text_df: {text_df}")
        print(text_df.info())


        semantic_search_service = BatchOpenAISemanticSearchService(self.config, self.sender)
        gpt_input_text_df = semantic_search_service.search_related_source(text_df, search_text)
        gpt_input_text_df = BatchOpenAISemanticSearchService.post_process_gpt_input_text_df(gpt_input_text_df,
                                                                                            self.config.get('llm_service').get('openai_api').get('prompt').get('prompt_token_limit'))

        llm_service = LLMServiceFactory.create_llm_service(self.config, self.sender)

        # prompt choice  [v3, fc, fc_zh, qg, ]
        if prompt_choice == "qg":
            prompt = llm_service.get_prompt_qg(search_text, gpt_input_text_df)
        elif prompt_choice == "fc_en":
            prompt = llm_service.get_prompt_fc(search_text, gpt_input_text_df)
        elif prompt_choice == "fc_zh":
            prompt = llm_service.get_prompt_fc_zh(search_text, gpt_input_text_df)
        else:
            raise ValueError("Invalid prompt choice")

        #prompt = llm_service.get_prompt_v3(search_text, gpt_input_text_df)  # normal
        #prompt = llm_service.get_prompt_fc(search_text, gpt_input_text_df)  # fc_en
        #prompt = llm_service.get_prompt_fc_zh(search_text, gpt_input_text_df)  # fc_zh
        logger.info(f"prompt:{prompt}")
        response_text = llm_service.call_api(prompt=prompt)

        frontend_service = FrontendService(self.config, response_text, gpt_input_text_df)
        source_text, data_json = frontend_service.get_data_json(response_text, gpt_input_text_df)

        print('===========Prompt:============')
        print(prompt)
        print('===========Search:============')
        print(search_text)
        print('===========Response text:============')
        print(response_text)
        print('===========Source text:============')
        print(source_text)

        return response_text, source_text, data_json

        # Only use LLM service

    @storage_cached('web', 'search_text')
    def query_and_get_answer_llm(self, search_text, prompt_choice, text_df=None, query=None,
                                compressed_info_content=None):

        gpt_input_text_df = pd.DataFrame()
        # print("before judge text df")
        if text_df is not None and not text_df.empty:
            print("text df is not empty, searching...")
            semantic_search_service = BatchOpenAISemanticSearchService(self.config, self.sender)
            # gpt_input_text_df = semantic_search_service.search_related_source(text_df, search_text)
            if query is not None:
                logger.info(
                    f"SearchGPTService. search_query: {query})")
                gpt_input_text_df = semantic_search_service.search_related_source(text_df, query)
            else:
                gpt_input_text_df = semantic_search_service.search_related_source(text_df, search_text)
            gpt_input_text_df = BatchOpenAISemanticSearchService.post_process_gpt_input_text_df(gpt_input_text_df,
                                                                                                self.config.get(
                                                                                                    'llm_service').get(
                                                                                                    'openai_api').get(
                                                                                                    'prompt').get(
                                                                                                    'prompt_token_limit'))

        # if not compressed_info_content:
        #     print("compressed_info_content列表为空")
        # else:
        #     print("compressed_info_content列表不为空")

        llm_service = LLMServiceFactory.create_llm_service(self.config, self.sender)

        # prompt choice  [v3, fc, fc_zh, qg, sc, qa. rc ]
        if prompt_choice == "qg":
            prompt = llm_service.get_prompt_qg(search_text)
        elif prompt_choice == "fc":
            prompt = llm_service.get_prompt_fc(search_text, gpt_input_text_df)
        elif prompt_choice == "va":
            prompt = llm_service.get_prompt_va(search_text, gpt_input_text_df)
        elif prompt_choice == "va_zh":
            prompt = llm_service.get_prompt_va(search_text, gpt_input_text_df)
        elif prompt_choice == "sc":
            prompt = llm_service.get_prompt_sc(search_text, gpt_input_text_df, compressed_info_content)
        elif prompt_choice == "sc_zh":
            prompt = llm_service.get_prompt_sc_zh(search_text, gpt_input_text_df)
        elif prompt_choice == "fc_zh":
            prompt = llm_service.get_prompt_fc_zh(search_text, gpt_input_text_df)
        elif prompt_choice == "qa":
            prompt = llm_service.get_prompt_qa(search_text)
        elif prompt_choice == "rc":
            prompt = llm_service.get_prompt_rc(search_text)
        elif prompt_choice == "sm":
            prompt = llm_service.get_prompt_sm(search_text)
        else:
            raise ValueError("Invalid prompt choice")

        # prompt = llm_service.get_prompt_v3(search_text, gpt_input_text_df)  # normal
        # prompt = llm_service.get_prompt_fc(search_text, gpt_input_text_df)  # fc_en
        # prompt = llm_service.get_prompt_fc_zh(search_text, gpt_input_text_df)  # fc_zh
        logger.info(f"prompt:{prompt}")
        response_text = llm_service.call_api(prompt=prompt)


        #if gpt_input_text_df:
        #    frontend_service = FrontendService(self.config, response_text, gpt_input_text_df)
        #    source_text, data_json = frontend_service.get_data_json(response_text, gpt_input_text_df)

        #    print('===========Prompt:============')
        #    print(prompt)
        #    print('===========Search:============')
        #    print(search_text)
        #    print('===========Response text:============')
        #    print(response_text)
        #    print('===========Source text:============')
        #    print(source_text)

        #    return response_text, source_text, data_json

        print('===========Prompt:============')
        print(prompt)

        print('===========Response text:============')
        print(response_text)

        #if not gpt_input_text_df.empty:
        #    print("flag2")
        #    return response_text, gpt_input_text_df
        #else:
        #    print("flag3")
        #    return response_text
        return response_text

    # Use LLM service with
    @storage_cached('web', 'search_text')
    def query_and_get_answer_llm_qg(self, search_text, prompt_choice, text_df=None):

        gpt_input_text_df = pd.DataFrame()
        # print("before judge text df")
        if not text_df.empty:
            print("text df is not empty, searching...")
            semantic_search_service = BatchOpenAISemanticSearchService(self.config, self.sender)
            gpt_input_text_df = semantic_search_service.search_related_source(text_df, search_text)
            gpt_input_text_df = BatchOpenAISemanticSearchService.post_process_gpt_input_text_df(gpt_input_text_df,
                                                                                                self.config.get(
                                                                                                    'llm_service').get(
                                                                                                    'openai_api').get(
                                                                                                    'prompt').get(
                                                                                                    'prompt_token_limit'))

        llm_service = LLMServiceFactory.create_llm_service(self.config, self.sender)

        # prompt choice  [v3, fc, fc_zh, qg, ]
        if prompt_choice == "qg":
            prompt = llm_service.get_prompt_qg(search_text, gpt_input_text_df)
        elif prompt_choice == "qg_zh":
            prompt = llm_service.get_prompt_qg_zh(search_text, gpt_input_text_df)
        elif prompt_choice == "fc_en":
            prompt = llm_service.get_prompt_fc(search_text)
        elif prompt_choice == "fc_zh":
            prompt = llm_service.get_prompt_fc_zh(search_text)
        else:
            raise ValueError("Invalid prompt choice")

        # prompt = llm_service.get_prompt_v3(search_text, gpt_input_text_df)  # normal
        # prompt = llm_service.get_prompt_fc(search_text, gpt_input_text_df)  # fc_en
        # prompt = llm_service.get_prompt_fc_zh(search_text, gpt_input_text_df)  # fc_zh
        logger.info(f"prompt:{prompt}")
        response_text = llm_service.call_api(prompt=prompt)

        print('===========Prompt:============')
        print(prompt)

        print('===========Response text:============')
        print(response_text)

        return response_text

    # semantic research
    @storage_cached('web', 'search_text')
    def query_and_get_answer_llm_sr(self, search_text, prompt_choice, text_df=None, n=1):

        gpt_input_text_df = pd.DataFrame()
        #print("before judge text df")
        if not text_df.empty:
            print("text df is not empty, searching...")
            semantic_search_service = BatchOpenAISemanticSearchService(self.config, self.sender)
            gpt_input_text_df = semantic_search_service.search_related_source(text_df, search_text, n)
            gpt_input_text_df = BatchOpenAISemanticSearchService.post_process_gpt_input_text_df(gpt_input_text_df,
                                                                                                self.config.get(
                                                                                                    'llm_service').get(
                                                                                                    'openai_api').get(
                                                                                                    'prompt').get(
                                                                                                    'prompt_token_limit'))


        llm_service = LLMServiceFactory.create_llm_service(self.config, self.sender)

        # prompt choice  [v3, fc, fc_zh, qg, sc ]
        if prompt_choice == "qg":
            prompt = llm_service.get_prompt_qg(search_text)
        elif prompt_choice == "fc_en":
            prompt = llm_service.get_prompt_fc(search_text)
        elif prompt_choice == "sc":
            prompt = llm_service.get_prompt_sc(search_text, gpt_input_text_df)
        elif prompt_choice == "fc_zh":
            prompt = llm_service.get_prompt_fc_zh(search_text)
        else:
            raise ValueError("Invalid prompt choice")

        # prompt = llm_service.get_prompt_v3(search_text, gpt_input_text_df)  # normal
        # prompt = llm_service.get_prompt_fc(search_text, gpt_input_text_df)  # fc_en
        # prompt = llm_service.get_prompt_fc_zh(search_text, gpt_input_text_df)  # fc_zh
        logger.info(f"prompt:{prompt}")
        response_text = llm_service.call_api(prompt=prompt)


        #if gpt_input_text_df:
        #    frontend_service = FrontendService(self.config, response_text, gpt_input_text_df)
        #    source_text, data_json = frontend_service.get_data_json(response_text, gpt_input_text_df)

        #    print('===========Prompt:============')
        #    print(prompt)
        #    print('===========Search:============')
        #    print(search_text)
        #    print('===========Response text:============')
        #    print(response_text)
        #    print('===========Source text:============')
        #    print(source_text)

        #    return response_text, source_text, data_json

        print('===========Prompt:============')
        print(prompt)

        print('===========Response text:============')
        print(response_text)

        #if not gpt_input_text_df.empty:
        #    print("flag2")
        #    return response_text, gpt_input_text_df
        #else:
        #    print("flag3")
        #    return response_text
        return response_text