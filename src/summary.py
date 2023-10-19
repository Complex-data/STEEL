from SearchGPTService import SearchGPTService
import json
from tqdm import tqdm
from my_logging import logger
import re
import random
import time
import openai

options = ["true", "false"]

# upper_bound = 100



def call_api_with_retry(search_text):
    max_retries = 10
    retry_delay = 10  # 重试延迟时间（秒）
    search_gpt_service = SearchGPTService()

    for _ in range(max_retries):
        try:
            if len(search_text) > 20000:
                print(len(search_text))
                continue

            predict = search_gpt_service.query_and_get_answer_llm(search_text=search_text,
                                                                  prompt_choice="sm",
                                                                  text_df=None)
            # 在这里处理 API 调用的结果，可以根据需要进行进一步的逻辑处理
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

    # ds = [json.loads(data_str) for data_str in open(args.evaluate_task_data_path).readlines()] fix
    with open(args.evaluate_task_data_path) as file:
        lines = file.readlines()

    # 将每行数据解析为字典对象
    ds = [json.loads(line) for line in lines]

    ds = random.sample(ds, 5) if len(ds) > 1000 else ds

    all_data = []

    # 随机打乱数据顺序
    # random.shuffle(lines)

    correct, total = 0, 0

    for ix, sample in enumerate(tqdm(ds)):
        print(len(sample['question']))
        # if len(sample['question']) > 200:
        #    print(len(sample['question']))
        #    continue
        search_text = sample['question']

        # time.sleep(1)
        # if len(search_text)> 200:
        #     print(len(search_text))
        #     continue
        predict = call_api_with_retry(search_text)
        #predict = model.query(sample['question'])['answer']
        label = sample['answer']  # 获取第一个单词   LIAR
        #label = sample['answer'][0]  # 获取第一个单词   WEIBO
        # print(f"predict type:{type(predict)}")
        # print(f"predict before:{predict}")
        # print(f"label:{label}")
        predict = str(predict)
        if "Claim: " in predict:
            # 如果包含"Claim:"，则去掉它
            predict = predict.replace("Claim: ", "")
        print(f"predict:{predict}")
        new_data = {"question": predict.replace('\n', ''), "answer": label}
        all_data.append(new_data)

        # predict = re.sub(r"\b(not true|No)\b", "false", predict, count=1, flags=re.IGNORECASE)
        #
        # # Replace the first occurrence of "Yes" with "true"
        # predict = re.sub(r"\bYes\b", "true", predict, count=1, flags=re.IGNORECASE)
        # print(f"predict after:{predict}, type:{type(predict)}")
        # match = re.search(r"\b(true|false)\b", predict, re.IGNORECASE)
        # print(f"match before:{match}")
        # if match:
        #     Answer = match.group(0).lower()
        #     print(f"got it, answer:{Answer}")
        # else:
        #     Answer = random.choice(options)
        #     print(f"fail it, answer:{Answer}")
        #
        # # Answer = predict.strip().split()[0]
        # # if Answer.lower() == 'yes':
        # #    Answer = 'true'
        # # elif Answer.lower() == 'no':
        # #    Answer = 'false'
        # if label == Answer:
        #     correct += 1
        # total += 1
        # # logger.info(f"Q: {sample['question']}; A: {Answer}; GT: {label}; EX:{predict}")
        # first_sentence = predict.split(".")[0] + "."
        # logger.info(f"Q:{sample['question']};P:{predict}; A:{Answer};GT: {label}")
        # print(f"After {ix} batch, correct:{correct},total:{total},Accuracy:{correct / total}")
        # if total == upper_bound:
        #     break
    # 假设你已经有了一个包含所有数据的列表
    # all_data = [new_data]

    # 打开一个新的JSONL文件
    with open('./data/politifact_qa_claim.jsonl', 'w') as f:
        # 遍历所有数据
        for item in all_data:
            # 将每个数据项转换为JSON格式，并添加一个换行符，然后写入文件
            f.write(json.dumps(item) + '\n')
    return 1