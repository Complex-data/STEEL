# -*- coding: utf-8 -*-
import os
import re
import concurrent.futures
import pandas as pd
import requests
import yaml

from Util import setup_logger, get_project_root, storage_cached
from text_extract.html.beautiful_soup import BeautifulSoupSvc
from text_extract.html.trafilatura import TrafilaturaSvc

from my_logging import logger

#logger = setup_logger('BingService')


class BingService:
    def __init__(self, config):
        self.config = config
        extract_svc = self.config.get('source_service').get('bing_search').get('text_extract')
        if extract_svc == 'trafilatura':
            self.txt_extract_svc = TrafilaturaSvc()
        elif extract_svc == 'beautifulsoup':
            self.txt_extract_svc = BeautifulSoupSvc()

    @storage_cached('bing_search_website', 'search_text')
    def call_bing_search_api(self, search_text: str, return_rest: bool=False) -> pd.DataFrame:
        logger.info("BingService.call_bing_search_api. query: " + search_text)
        subscription_key = self.config.get('source_service').get('bing_search').get('subscription_key')
        endpoint = self.config.get('source_service').get('bing_search').get('end_point') + "/v7.0/search"
        mkt = self.config.get('general').get('language')
        params = {'q': search_text, 'mkt': mkt}
        headers = {'Ocp-Apim-Subscription-Key': subscription_key}
        # Is there a filtering mechanism to get the required urls (if it exists)
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()

            columns = ['name', 'url', 'snippet']
            if response.json().get('webPages'):
                website_df = pd.DataFrame(response.json()['webPages']['value'])[columns]
                website_df['url_id'] = website_df.index + 1
                # Filter out inaccessible links and get the top n.
                website_df = website_df[website_df['url'].apply(self.is_url_accessible)]
                print(f"website_df_len:{len(website_df)}")
                website_df_string = website_df.to_string()
                logger.info(f"website_df:{website_df_string}")
                result_count = self.config.get('source_service').get('bing_search').get('result_count')
                website_df_rest = website_df[result_count:2*result_count]
                website_df = website_df[:result_count]

            else:
                website_df = pd.DataFrame(columns=columns + ['url_id'])
        except Exception as ex:
            raise ex
        if return_rest:
            return website_df_rest
        else:
            return website_df

    def call_urls_and_extract_sentences(self, website_df) -> pd.DataFrame:
        """
        :param:
            website_df: one row = one website with url
                name: website title name
                url: url
                snippet: snippet of the website given by BingAPI
        :return:
            text_df: one row = one website sentence
            columns:
                name: website title name
                url: url
                snippet: snippet of the website given by BingAPI
                text: setences extracted from the website
        """
        website_df_str = website_df.to_string()
        #logger.info(f"BingService.call_urls_and_extract_sentences. website_df.shape: {website_df.shape}")
        logger.info(f"website_df_str: {website_df_str}")
        name_list, url_list, url_id_list, snippet_list, text_list = [], [], [], [], []
        for index, row in website_df.iterrows():
            logger.info(f"Processing url: {row['url']}")
            sentences = self.extract_sentences_from_url(row['url'])
            for text in sentences:
                word_count = len(re.findall(r'\w+', text))  # approximate number of words
                if word_count < 8:
                    continue
                name_list.append(row['name'])
                url_list.append(row['url'])
                url_id_list.append(row['url_id'])
                snippet_list.append(row['snippet'])
                text_list.append(text)
        text_df = pd.DataFrame(data=zip(name_list, url_list, url_id_list, snippet_list, text_list),
                               columns=['name', 'url', 'url_id', 'snippet', 'text'])
        text_df_str = text_df.to_string()
        logger.info(f"text_df_str:{text_df_str}")
        return text_df

    def call_one_url(self, website_tuple):
        name, url, snippet, url_id = website_tuple
        logger.info(f"Processing url: {url}")
        sentences = self.extract_sentences_from_url(url)
        logger.info(f"  receive sentences: {len(sentences)}")
        return sentences, name, url, url_id, snippet

    @storage_cached('bing_search_website_content', 'website_df')
    def call_urls_and_extract_sentences_concurrent(self, website_df):
        logger.info(f"BingService.call_urls_and_extract_sentences_async. website_df.shape: {website_df.shape}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(self.call_one_url, website_df.itertuples(index=False)))
        name_list, url_list, url_id_list, snippet_list, text_list = [], [], [], [], []
        for result in results:
            sentences, name, url, url_id, snippet = result

            # Use all sentences
            # sentences = sentences[:self.config['source_service']['bing_search']['sentence_count_per_site']]  # filter top N only for stability
            for text in sentences:
                #print(f"text:{text}")
                #text.encode('gbk').decode('utf-8')
                #print(f"text:{text}")
                word_count = len(re.findall(r'\w+', text))  # approximate number of words
                # if word_count < 8:  # Is that rational when used in fact check?
                if word_count < 4:
                    continue
                name_list.append(name)
                url_list.append(url)
                url_id_list.append(url_id)
                snippet_list.append(snippet)
                text_list.append(text)
        text_df = pd.DataFrame(data=zip(name_list, url_list, url_id_list, snippet_list, text_list),
                               columns=['name', 'url', 'url_id', 'snippet', 'text'])
        return text_df

    def extract_sentences_from_url(self, url):
        # Fetch the HTML content of the page
        try:
            response = requests.get(url, timeout=3)
            encoding = response.apparent_encoding
            response.encoding = encoding
            print(f'网页内容使用的编码为: {encoding}')
        except:
            logger.error(f"Failed to fetch url: {url}")
            return []
        html_content = response.text


        # Use trafilatura to parse the HTML and extract the text
        extract_text = self.txt_extract_svc.extract_from_html(html_content)  # doc
        print(f"extract_text type:{type(extract_text)}")
        print(f"extract_text:{extract_text}")
        print("--------------------------------------------------------------")
        return extract_text

    def is_url_accessible(self, url):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except Exception as e:
            print(f"Error accessing {url}: {str(e)}")
        return False


if __name__ == '__main__':
    # Load config
    with open(os.path.join(get_project_root(), 'src/config/config.yaml'), encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        service = BingService(config)
        website_df = service.call_bing_search_api(search_text='What is ChatGPT')
        print("===========Website df:============")
        print(website_df)
        # text_df = service.call_urls_and_extract_sentences(website_df)
        text_df = service.call_urls_and_extract_sentences_concurrent(website_df=website_df)
        print("===========text df:============")
        print(text_df)
