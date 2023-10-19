import logging
import argparse
import os

from datetime import datetime
#args = parser.parse_args()

# Get the current timestamp
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

task_option = ['LIAR_test_qa_wo_search', 'LIAR_test_qa_wi_search', 'LIAR_test_qa_wi_search_enhanced',
               'LIAR_QA_RETR_WEB', 'LIAR_QA_VA', 'GOSSIPCOP_QA', 'POLITIFACT_QA',
               'POLITIFACT_qa_wo_search', 'FERVEROUS_QA',
               'WEIBO_qa_wo_search', 'WEIBO_qa_wi_search',
               'HOVER_qa_wo_search', 'HOVER_qa_wi_search',
               'CHEF_qa_wo_search', 'CHEF_qa_wi_search', 'CHEF_test_qa_wi_semantic_search_enhanced',
               'CHEF_QA_RETR_WEB', 'CHEF_QA_VA']


task = task_option[2]


log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)



#"""
# 创建一个logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建一个文件处理器
handler = logging.FileHandler(os.path.join(log_dir, task + '_' + timestamp + '.txt'), mode='w')
handler.setLevel(logging.INFO)

# 创建一个格式化器
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)

# 创建一个格式化器
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)

# 将处理器添加到logger
logger.addHandler(handler)
#"""
