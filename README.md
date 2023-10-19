Getting Started
---------------

### Prerequisites

To run `STEEL`, you'll need:

* [Python 3.10.8](https://www.python.org/downloads/)
* [OpenAI API Key](https://beta.openai.com/signup) or [GooseAI API Key](https://goose.ai/)
* [Azure Bing Search Subscription Key](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api/)

### Installation

1. Create your python or anaconda env and install python packages

Native
```
# using python=3.10.8
pip install -r requirements.txt
```

Anaconda
```
conda create --name STEEL python=3.10.8
conda activate STEEL
pip install -r requirements.txt
```

2. Input API keys (OpenAI/Azure Bing Search) in `STEEL/src/config/config.yaml` 
3. Run eval
```
cd STEEL/src
python evaluate.py --task LIAR_QA_RETR_WEB --evaluate_task_data_path ./data/LIAR_test_qa.jsonl 


