from SearchGPTService import SearchGPTService

if __name__ == '__main__':
    search_text = 'Is the claim (This is michelle-obama statement. Only 2 percent of public high schools in the country offer PE classes.) true?'

    search_gpt_service = SearchGPTService()
    response_text, source_text, data_json = search_gpt_service.query_and_get_answer(search_text=search_text)
    print("response_text:", response_text)
    print("source_text:", source_text)
    print("data_json:", data_json)

