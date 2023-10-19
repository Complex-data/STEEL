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

start_row = 3   # Start row of dataset
upper_bound = 3   # Number of tests

confidence_value = 100    # Default
over_confidence = 0.8   # Discount
steps_total = 3   # Maximum of search steps

search_gpt_service = SearchGPTService()
compressed_info = ""  # established evidence

global TP, FP, TN, FN
TP = FP = TN = FN = 0

def self_correction(response):
    response_correction = search_gpt_service.query_and_get_answer_llm(search_text=response,
                                                                  prompt_choice="rc",
                                                                  text_df=None)
    return response_correction

global steps_control

def search_claims(search_text, steps_controls, text_df=None, query=None, over_confidence_a=False, compressed_info_content=None):
    global steps_total, confidence_value, compressed_info
    if steps_controls > 0:
        steps_controls -= 1
        if steps_controls < 2 and not over_confidence_a:
            logger.info(f"Steps: {steps_total - steps_controls} . Searching query: {query}.")
            # print(f"Steps: {steps_total - steps_controls} . Searching query: {search_text}.")
            # print(f"此处 compressed_info: {compressed_info}")  #
            logger.info(f"Here is compressed_info: {compressed_info}")
            response = search_gpt_service.query_and_get_answer_llm(search_text=search_text,
                                                                   prompt_choice='sc',  # sc_zh,sc,fc_zh,sm
                                                                   text_df=text_df,
                                                                   query=query,
                                                                   compressed_info_content=compressed_info)
        else:
            logger.info(f"Steps: {steps_total - steps_controls} . Searching query: {search_text}.")
            # print(f"Steps: {steps_total - steps_controls} . Searching query: {search_text}.")
            response = search_gpt_service.query_and_get_answer_llm(search_text=search_text,
                                                                   prompt_choice='sc',  #sc_zh,sc,fc_zh,sm
                                                                   text_df=text_df)
        logger.info('===========Response text:============')
        logger.info(response)
        # Use regular expressions to match the value after Answer
        match_va = re.search(r'Answer[:：]\s*(\d)', response)
        # Use regular expressions to match the value after Confidence
        match_con = re.search(r"把握[:：]\s*(\d+)", response)
        if match_con:
            number = match_con.group(1)
            print(number)
        else:
            match_con = re.search(r'(?i)confidence[:：]\s*(\d+)', response)
        match_rc = re.search(r'Answer[:：]\s*(\d)', response)
        # If match_rc is not None, use it
        if match_rc:
            match = match_rc
        # If match_rc is None, use match_va
        else:
            match = match_va
        if match_con:
            confidence_value = float(match_con.group(1))
            confidence_value_dis = "{:.2f}".format((confidence_value * 0.01) * over_confidence)
            logger.info(f"Here is Confidence: {confidence_value_dis}")
        else:
            print("Confidence value not found, set to 1")
            confidence_value = 1.0
        print(f"match={match}")
        if match:
            answer = int(match.group(1))
            if answer == 0:
                response = "true"
                if confidence_value > 0.5:
                    print(f"The claim is {response}")
                    return response
                else:
                    # read it twice
                    return search_claims(search_text, steps_controls, text_df, over_confidence_a=True)
            elif answer == 1:
                response = "false"
                if confidence_value > 0.5:
                    print(f"The claim is {response}")
                    return response
                else:
                    # read it twice
                    return search_claims(search_text, steps_controls, text_df, over_confidence_a=True)
            elif answer == 2:
                # Extract Query and Compressed information
                query_content = re.findall(r'Query: (\[.*?\]|".*?")', response)
                compressed_info += "".join(re.findall(r'Compressed information: (.*?)$', response, re.MULTILINE))
                # =============================================================================================================
                if confidence_value is not None:
                    if confidence_value > 0.5:
                        # print(f"Here is query_content: {query_content}")
                        for query in query_content:
                            # print(f"Here is query: {query}")
                            website_df_rest = bing_service.call_bing_search_api(search_text=query, return_rest=False)
                            website_df_rest = bing_service.call_urls_and_extract_sentences_concurrent(website_df=website_df_rest)
                            result = search_claims(search_text, steps_controls, website_df_rest, query=query, compressed_info_content=compressed_info)
                            if result == "true" or result == "false":
                                return result
                        return "false"
                    else:
                        # read it twice
                        return search_claims(search_text, steps_controls, text_df, over_confidence_a=True)
                else:
                    for query in query_content:
                        website_df_rest = bing_service.call_bing_search_api(search_text=query, return_rest=False)
                        website_df_rest = bing_service.call_urls_and_extract_sentences_concurrent(
                            website_df=website_df_rest)
                        search_claims(search_text, steps_controls, website_df_rest, query=query,
                                      compressed_info_content=compressed_info)
        else:
            # In case the answer is correct but not output according to the format
            second_response = search_gpt_service.query_and_get_answer_llm(search_text=response,
                                                                          prompt_choice="qa",
                                                                          text_df=None)
            match = re.search(r'Answer[:：]\s*(\d)', second_response)
            if match:
                answer = int(match.group(1))
                if answer == 0:
                    response = "true"
                    print(f"The claim is {response}")
                    return response
                elif answer == 1:
                    response = "false"
                    print(f"The claim is {response}")
                    return response
                elif answer == 2:
                    # response = "false"
                    # print(f"The claim is {response}")
                    # return response
                    # Second web search
                    website_df_rest = bing_service.call_bing_search_api(search_text=search_text,
                                                                        return_rest=True)
                    website_df_rest = bing_service.call_urls_and_extract_sentences_concurrent(
                        website_df=website_df_rest)

                    response = search_claims(search_text, steps_controls, website_df_rest)
                    return response

            return response

    else:    # If out of steps, respond false (Do not work as wish)
        print("Out of steps, assuming this statement is false.")
        logger.info("Out of steps, assuming this statement is false.")
        response = "false"
        return response
    return response



def call_api_with_retry(search_text):
    max_retries = 10
    retry_delay = 10  # Retry delay time (seconds)
    for _ in range(max_retries):
        try:
            if len(search_text) > 1200:
                print(len(search_text))
                continue
            website_df = pd.DataFrame()
            result = bing_service.call_bing_search_api(search_text=search_text)
            website_df = pd.concat([website_df, result])
            text_df = bing_service.call_urls_and_extract_sentences_concurrent(website_df=website_df)
            text_df_str = text_df.to_string()
            logger.info(f"===========text df:============\n{text_df_str}")
            steps_controls = steps_total
            # Call the search_claims function with the initial search_text
            response = search_claims(search_text=search_text, steps_controls=steps_controls,  text_df=text_df)
            # print(f"response:{response}")
            # print(f"type of response:{type(response)}")
            # The results of the API call are processed here, and further logical processing can be performed as needed.
            predict = response
            return predict

        except openai.error.APIError as e:
            print(f"APIError occurred: {e}")
            # Here you can handle errors according to specific circumstances, such as recording logs, adjusting retry
            # strategies, etc.

        except Exception as e:
            print(f"An error occurred: {e}")
            # Other exceptions can be handled here, such as network connection issues, etc.

        # Wait for some time and try again
        time.sleep(retry_delay)

    # If the maximum number of retries is reached and still fails, appropriate handling is done here
    print("Reached maximum retries, exiting...")
    return None, None, None

def metrics(TP,FP,TN,FN):
    eps = 0.01

    P_T = (TP + eps) / (TP + FP + 2*eps)
    R_T = (TP + eps) / (TP + FN + 2 * eps)
    F1_T = 2*(P_T * R_T) / (P_T + R_T)

    P_F = (TN + eps) / (TN + FN + 2 * eps)
    R_F = (TN + eps) / (TN + FP + 2 * eps)
    F1_F = 2 * (P_F * R_F) / (P_F + R_F)

    F1_Ma = (F1_T + F1_F) / 2
    F1_Mi = (TP + FN) / (TP + FN + TN + FP)

    return F1_Ma, F1_Mi, F1_T, P_T, R_T, F1_F, P_F, R_F


def eval(args):
    global start_row
    global TP, FP, TN, FN
    start_row_bk = start_row
    with open(args.evaluate_task_data_path) as file:
        lines = file.readlines()

    # Parse each row of data into a dictionary object
    ds = [json.loads(line) for line in lines]

    # Randomly shuffle data order
    random.shuffle(ds)

    correct, total = 0, 0

    for ix, sample in enumerate(tqdm(ds)):
        if start_row_bk > 0:
            start_row_bk -= 1
            continue
        print(len(sample['question']))
        search_text = sample['question']

        time.sleep(1)
        if len(search_text) > 500:
            print(len(search_text))
            continue
        #predict, source_text_result, data_json_result = call_api_with_retry(search_text)
        predict = call_api_with_retry(search_text)
        #predict = model.query(sample['question'])['answer']
        label = sample['answer']  # 获取第一个单词   LIAR
        #label = sample['answer'][0]  # 获取第一个单词
        #print(f"predict type:{type(predict)}")
        print(f"predict before:{predict}")
        print(f"label:{label}")
        predict = str(predict)

        # predict = re.sub(r"\b(not true|No)\b", "false", predict, count=1, flags=re.IGNORECASE)
        #
        # # Replace the first occurrence of "Yes" with "true"
        # predict = re.sub(r"\bYes\b", "true", predict, count=1, flags=re.IGNORECASE)
        # print(f"predict after:{predict}, type:{type(predict)}")

        match = re.search(r"\b(true|false)\b", predict, re.IGNORECASE)
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
        if (label == "true") and (Answer == "true"):
            TP += 1
            correct += 1
        elif (label == "true") and (Answer == "false"):
            FN += 1
        elif (label == "false") and (Answer == "false"):
            TN += 1
            correct += 1
        else:
            FP += 1
        total += 1
        # logger.info(f"Q: {sample['question']}; A: {Answer}; GT: {label}; EX:{predict}")
        # first_sentence = predict.split(".")[0] + "."
        print(f"Q:{sample['question']};P:{predict}; A:{Answer};GT: {label}")
        logger.info(f"Q:{sample['question']};P:{predict}; A:{Answer};GT: {label}")

        F1_Ma, F1_Mi, F1_T, P_T, R_T, F1_F, P_F, R_F = metrics(TP, FP, TN, FN)

        logger.info(f"After {ix - start_row + 1} batch, correct:{correct},total:{total},Accuracy:{correct / total}")
        print(f"After {ix - start_row + 1} batch, correct:{correct},total:{total},Accuracy:{correct / total:.4f}")
        if total % 10 == 0:
            print(f"After {ix - start_row + 1} batch, F1_Ma:{F1_Ma}, F1_Mi:{F1_Mi}, F1_T:{F1_T}, P_T:{P_T}, \
                        R_T:{R_T}, F1_F:{F1_F}, P_F:{P_F}, R_F:{R_F}, correct:{correct},total_test:{total}")
            logger.info(f"After {ix - start_row + 1} batch, F1_Ma:{F1_Ma}, F1_Mi:{F1_Mi}, F1_T:{F1_T}, P_T:{P_T}, \
            R_T:{R_T}, F1_F:{F1_F}, P_F:{P_F}, R_F:{R_F}, correct:{correct},total_test:{total}")
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