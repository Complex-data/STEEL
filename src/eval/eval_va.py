from SearchGPTService import SearchGPTService
from BingService import *
import json
from tqdm import tqdm
from my_logging import logger
import random
import time
import openai
from LLMService import LLMServiceFactory
from website.sender import Sender
import ast


# BingService.py import
import os
import re
import concurrent.futures
import pandas as pd
import requests
import yaml

from Util import setup_logger, get_project_root, storage_cached
from text_extract.html.beautiful_soup import BeautifulSoupSvc
from text_extract.html.trafilatura import TrafilaturaSvc
#####################


# SearchGPTService.py
from SemanticSearchService import BatchOpenAISemanticSearchService
from SourceService import SourceService
#################################


# Load config
with open(os.path.join(get_project_root(), 'src/config/config.yaml'), encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
bing_service = BingService(config)
options = ["true", "false"]

start_row = 0
upper_bound = 500

confidence_value = 100    # default
over_confidence = 0.8
is_web_problem_num = 0
steps_total = 1

search_gpt_service = SearchGPTService()

def self_correction(response):
    response_correction = search_gpt_service.query_and_get_answer_llm(search_text=response,
                                                                  prompt_choice="rc",
                                                                  text_df=None)
    return response_correction

def search_claims(search_text, steps_controls, text_df=None):
    global steps_total, is_web_problem_num, confidence_value

    if steps_controls > 0:
        steps_controls -= 1
        logger.info(f"Steps: {steps_total - steps_controls} . Searching query: {search_text}.")
        print(f"Steps: {steps_total - steps_controls} . Searching query: {search_text}.")
        #page_num = steps - steps_controls
        response = search_gpt_service.query_and_get_answer_llm(search_text=search_text,
                                                               prompt_choice='va_zh',  #sc_zh,sc,va,fc_zh
                                                               text_df=text_df)
        #response = search_gpt_service.query_and_get_answer_llm(search_text=search_text,
        #                                                       prompt_choice='sc',
        #                                                       text_df=text_df)
        #if len(response) == 2:
        #    response_text, gpt_input_text_df = response
        #else:
        #    response_text = response[0]
        print('===========Response text:============')
        print(response)
        logger.info('===========Response text:============')
        logger.info(response)

        # match_va = re.search(r'Answer[:：]\s*(\d)', response)
        # # 使用正则表达式匹配Confidence后面的数值
        # match_con = re.search(r"把握[:：]\s*(\d+)", response)
        # if match_con:
        #     number = match_con.group(1)
        #     print(number)
        # else:
        #     match_con = re.search(r'(?i)confidence[:：]\s*(\d+)', response)

        # response = self_correction(response)
        # print('===========Response text after rc:============')
        # print(response)
        # logger.info('===========Response text after rc:============')
        # logger.info(response)

        # match_rc = re.search(r'Answer[:：]\s*(\d)', response)
        #
        # # If match_rc is not None, use it
        # if match_rc:
        #     match = match_rc
        # # If match_rc is None, use match_va
        # else:
        #     match = match_va
        #
        # if match_con:
        #     confidence_value = float(match_con.group(1))
        #     confidence_value = confidence_value / 100 * over_confidence
        #     print(confidence_value)  # 输出提取的Confidence数值
        #     logger.info(f"Confidence: {confidence_value}")
        # else:
        #     print("未找到Confidence数值")
        #print("flag4")
        pattern = re.compile(r"(True|False|Uncertain)", re.IGNORECASE)
        match = pattern.search(response)

        if match is not None:
            print(match.group())
            answer = match.group()
        else:
            answer = 'false'

        if answer is not None:
            if answer == "True":
                response = "true"
                # if confidence_value > 0.5:
                #     print(f"The claim is {response}")
                #     return response
                # else:
                #     # read it twice
                #     search_claims(search_text, steps_controls, text_df)
            elif answer == "False":
                response = "false"
                # if confidence_value > 0.5:
                #     print(f"The claim is {response}")
                #     return response
                # else:
                #     # read it twice
                #     search_claims(search_text, steps_controls, text_df)
            elif answer == "Uncertain":
                # Confidence begin
                # =============================================================================================================
                # if confidence_value is not None:
                #     if confidence_value > 0.5:
                #         # Second web search
                #         # It is assumed here that the order of web pages returned for each search is fixed, but in fact it is not.
                #         # website_df_rest = bing_service.call_bing_search_api(search_text=search_text, return_rest=True)
                #         # website_df_rest = bing_service.call_urls_and_extract_sentences_concurrent(website_df=website_df_rest)
                #         # search_claims(search_text, steps_controls, website_df_rest)
                #         website_df_rest = bing_service.call_bing_search_api(search_text=search_text, return_rest=False)
                #         website_df_rest = bing_service.call_urls_and_extract_sentences_concurrent(website_df=website_df_rest)
                #         search_claims(search_text, steps_controls, website_df_rest)
                #     else:
                #         # read it twice
                website_df_rest = bing_service.call_bing_search_api(search_text=search_text, return_rest=False)
                website_df_rest = bing_service.call_urls_and_extract_sentences_concurrent(website_df=website_df_rest)
                search_claims(search_text, steps_controls, website_df_rest)
                # Confidence end
                # =============================================================================================================
                # Self-consistence begin
                # **************************************************************************************************************
                # is_web_problem_num += 1
                # logger.info(f"After {steps_total - steps_controls} search(s). Finding the problem... Is web problem count:{is_web_problem_num}, steps_total:{steps_total}")
                # print(
                #     f"After {steps_total - steps_controls} search(s). Finding the problem... Is web problem count:{is_web_problem_num}, steps_total:{steps_total}")
                # if is_web_problem_num/steps_total <= 0.5:
                #     logger.info("Retry to find out the problem")
                #     print("Retry to find out the problem")
                #     search_claims(search_text, steps_controls, text_df)
                # else:
                #     logger.info("It is web problem! Trying getting required urls")
                #     print("It is web problem! Trying getting required urls")
                #
                #     # Second web search
                #     website_df_rest = bing_service.call_bing_search_api(search_text=search_text,
                #                                                         return_rest=True)
                #     website_df_rest = bing_service.call_urls_and_extract_sentences_concurrent(
                #         website_df=website_df_rest)
                #
                #     search_claims(search_text, steps_controls, website_df_rest)
                # **************************************************************************************************************
                # Self-consistence end

                #query = search_text + "这是真的。"
                #query = "True"
                #search_claims(query, steps_controls, text_df)
                # First, check if key evidence is in other text within same urls.

                # code...
                # print(f"After {steps} search(s). Not enough information, need to search more.")
                # Second, check whether it is LLMs's mistake or a web search problem
                # Rejudge to confirm the reason

                    # base
                    # response = "false"
                    # return response


                    # print("Not follow the rule, assuming this claim is false.")
                    # logger.info("Not follow the rule, assuming this claim is false.")
                    # response = "false"
                    # Use LLMs to understand other cases

                    # response = search_gpt_service.query_and_get_answer_llm(search_text=search_text,
                    #                                                        prompt_choice='qa')

                    # return response
                    #queries_generated = search_gpt_service.query_and_get_answer_llm_qg(search_text=search_text,
                    #                                          prompt_choice='qg_zh',
                    #                                           text_df=text_df)
                    # if isinstance(queries_generated, str):
                    #     try:
                    #         queries = ast.literal_eval(queries_generated)
                    #     except (ValueError, SyntaxError):
                    #         print("Invalid input for ast.literal_eval")
                    #     except Exception as e:
                    #         print(f"An unexpected error occurred: {e}")
                    # else:
                    #     queries = queries_generated


                    # sc
                    # pattern = r'"(.*?)"'
                    # queries = re.findall(pattern, queries_generated)
                    # print(f"queries:{queries}")
                    # logger.info(f"queries:{queries}")
                    # web_df = pd.DataFrame()
                    # for query_local in queries:
                    #     print(f"query_local:{query_local}")
                    #     result = bing_service.call_bing_search_api(search_text=query_local)
                    #     web_df = pd.concat([web_df, result])
                    # text_df_re = bing_service.call_urls_and_extract_sentences_concurrent(website_df=web_df)
                    # text_df_str = text_df_re.to_string()
                    # logger.info(f"===========text df:============\n{text_df_str}")
                    # search_claims(search_text, steps_controls, text_df_re)


                # code...

                # match = re.search(r'Query:\s*\[(.+?)\]', response)
                # if match:
                #     query_str = match.group(1)
                #     queries = [query.strip(' "')
                #                for query in query_str.split(',')]
                #     #print(f"The queries are: {queries}")
                #     logger.info(f"The queries are: {queries}")
                #     # Recursively call the search_claims function for each query
                #     for query in queries:
                #         # Multi steps search
                #         search_claims(query, steps_controls, text_df)
        else:
            #print("Not follow the rule, assuming this claim is false.")
            #response = "false"
            #Quadratic Answer Extraction Using Large Language Models
            # print("Not follow the rule, assuming this claim is false.")
            # logger.info("Not follow the rule, assuming this claim is false.")
            # response = "false"

            # In case the answer is correct but not output according to the format
            # second_response = search_gpt_service.query_and_get_answer_llm(search_text=response,
            #                                                               prompt_choice="qa",
            #                                                               text_df=None)
            # match = re.search(r'Answer[:：]\s*(\d)', second_response)
            # # print("flag4")
            # if match:
            #     answer = int(match.group(1))
            #     if answer == 0:
            #         response = "true"
            #         print(f"The claim is {response}")
            #         return response
            #     elif answer == 1:
            #         response = "false"
            #         print(f"The claim is {response}")
            #         return response
            #     elif answer == 2:
            #         # response = "false"
            #         # print(f"The claim is {response}")
            #         # return response
            #         # Second web search
            #         website_df_rest = bing_service.call_bing_search_api(search_text=search_text,
            #                                                             return_rest=True)
            #         website_df_rest = bing_service.call_urls_and_extract_sentences_concurrent(
            #             website_df=website_df_rest)

            # search_claims(search_text, steps_controls, text_df)
            response = 'false'
            return response



    else:    # If out of steps, respond false (Do not work as wish)
        print("Out of steps, assuming this statement is false.")
        logger.info("Out of steps, assuming this statement is false.")
        response = "false"
        return response
    return response



def call_api_with_retry(search_text):
    max_retries = 10
    retry_delay = 10  # 重试延迟时间（秒）


    for _ in range(max_retries):
        try:
            if len(search_text) > 200:
                print(len(search_text))
                continue

            #print("falg1")
            # Step 1: Generate queries for search engine
            #queries = search_gpt_service.query_and_get_answer_llm(search_text=search_text, prompt_choice="qg")
            #queries = ast.literal_eval(queries)
            #print(f"qureies:{queries}")
            #print(f"type of qureies:{type(queries)}")

            website_df = pd.DataFrame()
            #for query in queries:
            #    result = bing_service.call_bing_search_api(search_text=query)
            #    website_df = pd.concat([website_df, result])

            # result = bing_service.call_bing_search_api(search_text=search_text)
            # website_df = pd.concat([website_df, result])


            #print(website_df.head())


            text_df = bing_service.call_urls_and_extract_sentences_concurrent(website_df=website_df)
            text_df_str = text_df.to_string()
            logger.info(f"===========text df:============\n{text_df_str}")
            #print(text_df.info())
            #print(text_df)


            #loop

            #not_enough_information = True
            steps_controls = 1
            # Call the search_claims function with the initial search_text
            response = search_claims(search_text, steps_controls,  text_df)



            print(f"response:{response}")
            print(f"type of response:{type(response)}")

            #predict, source_text, data_json = search_gpt_service.query_and_get_answer(search_text=search_text)
            # 在这里处理 API 调用的结果，可以根据需要进行进一步的逻辑处理
            predict = response
            #return predict, source_text, data_json
            return predict

        except openai.error.APIError as e:
            print(f"APIError occurred: {e}")
            # 在这里可以根据具体情况处理错误，例如记录日志、调整重试策略等

        except Exception as e:
            print(f"An error occurred: {e}")
            # 在这里可以处理其他异常情况，例如网络连接问题等

        # 等待一段时间后重试
        time.sleep(retry_delay)

    # 如果达到最大重试次数仍然失败，则在此处进行适当的处理
    print("Reached maximum retries, exiting...")
    return None, None, None

# 调用函数并获取结果
# predict_result, source_text_result, data_json_result = call_api_with_retry("your_search_text_here")

def eval(args):
    global start_row
    start_row_bk = start_row
    # ds = [json.loads(data_str) for data_str in open(args.evaluate_task_data_path).readlines()] fix
    with open(args.evaluate_task_data_path) as file:
        lines = file.readlines()

    # 读取指定行数的数据
    # ds = pd.read_csv(args.evaluate_task_data_path, skiprows=start_row - 1, nrows=nrows)

    # 将每行数据解析为字典对象
    ds = [json.loads(line) for line in lines]

    # 随机打乱数据顺序
    random.shuffle(lines)

    correct, total = 0, 0

    for ix, sample in enumerate(tqdm(ds)):
        if start_row_bk > 0:
            start_row_bk -= 1
            continue
        print(len(sample['question']))
        # if len(sample['question']) > 200:
        #    print(len(sample['question']))
        #    continue
        search_text = sample['question']

        time.sleep(1)
        if len(search_text) > 200:
            print(len(search_text))
            continue
        #predict, source_text_result, data_json_result = call_api_with_retry(search_text)
        predict = call_api_with_retry(search_text)
        #predict = model.query(sample['question'])['answer']
        label = sample['answer']  # 获取第一个单词   LIAR
        #label = sample['answer'][0]  # 获取第一个单词   WEIBO
        #print(f"predict type:{type(predict)}")
        print(f"predict before:{predict}")
        print(f"label:{label}")
        predict = str(predict)
        predict = re.sub(r"\b(not true|No)\b", "false", predict, count=1, flags=re.IGNORECASE)

        # Replace the first occurrence of "Yes" with "true"
        predict = re.sub(r"\bYes\b", "true", predict, count=1, flags=re.IGNORECASE)
        print(f"predict after:{predict}, type:{type(predict)}")
        match = re.search(r"\b(true|false|uncertain)\b", predict, re.IGNORECASE)
        print(f"match before:{match}")
        if match:
            Answer = match.group(0).lower()
            print(f"got it, answer:{Answer}")
        else:
            Answer = random.choice(options)
            print(f"fail it, answer:{Answer}")

        # Answer = predict.strip().split()[0]
        # if Answer.lower() == 'yes':
        #    Answer = 'true'
        # elif Answer.lower() == 'no':
        #    Answer = 'false'
        if label == Answer:
            correct += 1
        total += 1
        # logger.info(f"Q: {sample['question']}; A: {Answer}; GT: {label}; EX:{predict}")
        first_sentence = predict.split(".")[0] + "."
        logger.info(f"Q:{sample['question']};P:{predict}; A:{Answer};GT: {label}")
        logger.info(f"After {ix - start_row + 1} batch, correct:{correct},total:{total},Accuracy:{correct / total}")
        print(f"After {ix - start_row + 1} batch, correct:{correct},total:{total},Accuracy:{correct / total:.4f}")
        if total == upper_bound:
            break

    return correct / total


# def search_claims_sc(search_text, steps_controls, text_df=None, retry_choice=None):
#     global steps
#     search_gpt_service = SearchGPTService()
#     if steps_controls > 0:
#         if retry_choice is not None:
#             if retry_choice == "semantic_research":
#                 logger.info(f"Steps: {steps - steps_controls} . Searching query: {search_text}.")
#                 steps_controls -= 1
#                 response = search_gpt_service.query_and_get_answer_llm(search_text=search_text,
#                                                                        prompt_choice='sc',
#                                                                        text_df=text_df)
#                 ...
#             else:
#                 ...
#             logger.info(f"Steps: {steps - steps_controls} . Searching query: {search_text}.")
#             steps_controls -= 1
#             response = search_gpt_service.query_and_get_answer_llm(search_text=search_text,
#                                                                    prompt_choice='sc',
#                                                                    text_df=text_df)
#             #response = search_gpt_service.query_and_get_answer_llm(search_text=search_text,
#             #                                                       prompt_choice='sc',
#             #                                                       text_df=text_df)
#             #if len(response) == 2:
#             #    response_text, gpt_input_text_df = response
#             #else:
#             #    response_text = response[0]
#
#             match = re.search(r'Answer:\s*(\d)', response)
#             print("flag4")
#             if match:
#                 answer = int(match.group(1))
#                 if answer == 0:
#                     response = "true"
#                     print(f"The claim is {response}")
#                     return response
#                 elif answer == 1:
#                     response = "false"
#                     print(f"The claim is {response}")
#                     return response
#                 elif answer == 2:
#                     logger.info(f"After {steps - steps_controls} search(s). Not enough information, need to search more.")
#                     # First, check if key evidence is in other text within same urls.
#                     # code...
#                     # print(f"After {steps} search(s). Not enough information, need to search more.")
#                     # Second, check if not the right websites
#                     # code...
#                     match = re.search(r'Query:\s*\[(.+?)\]', response)
#                     if match:
#                         query_str = match.group(1)
#                         queries = [query.strip(' "')
#                                    for query in query_str.split(',')]
#                         #print(f"The queries are: {queries}")
#                         logger.info(f"The queries are: {queries}")
#                         # Recursively call the search_claims function for each query
#                         for query in queries:
#                             # Multi steps search
#                             search_claims(query, steps_controls, text_df)
#             else:
#                 print("Not follow the rule, assuming this claim is false.")
#                 response = "false"
#                 return response
#         else:
#             logger.info(f"Steps: {steps - steps_controls} . Searching query: {search_text}.")
#             steps_controls -= 1
#             response = search_gpt_service.query_and_get_answer_llm(search_text=search_text,
#                                                                    prompt_choice='sc',
#                                                                    text_df=text_df)
#             # response = search_gpt_service.query_and_get_answer_llm(search_text=search_text,
#             #                                                       prompt_choice='sc',
#             #                                                       text_df=text_df)
#             # if len(response) == 2:
#             #    response_text, gpt_input_text_df = response
#             # else:
#             #    response_text = response[0]
#
#             print('===========Response text:============')
#             print(response_text)
#             logger.info('===========Response text:============')
#             logger.info(response_text)
#             match = re.search(r'Answer:\s*(\d)', response)
#             #print("flag4")
#             if match:
#                 answer = int(match.group(1))
#                 if answer == 0:
#                     response = "true"
#                     print(f"The claim is {response}")
#                     return response
#                 elif answer == 1:
#                     response = "false"
#                     print(f"The claim is {response}")
#                     return response
#                 elif answer == 2:
#                     logger.info(
#                         f"After {steps - steps_controls} search(s). Not enough information, need to search more.")
#                     # First, check if key evidence is in other text within same urls.
#                     # code...
#                     # print(f"After {steps} search(s). Not enough information, need to search more.")
#                     # Second, check if not the right websites
#                     # code...
#                     match = re.search(r'Query:\s*\[(.+?)\]', response)
#                     if match:
#                         query_str = match.group(1)
#                         queries = [query.strip(' "')
#                                    for query in query_str.split(',')]
#                         # print(f"The queries are: {queries}")
#                         logger.info(f"The queries are: {queries}")
#                         # Recursively call the search_claims function for each query
#                         for query in queries:
#                             # Multi steps search
#                             search_claims(query, steps_controls, text_df)
#             else:
#                 print("Not follow the rule, assuming this claim is false.")
#                 response = "false"
#                 return response
#
#     # If out of steps, respond false
#     print("Out of steps, assuming this statement is false.")
#     response = "false"
#     return response