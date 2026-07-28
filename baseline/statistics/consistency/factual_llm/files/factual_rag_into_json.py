from factual_into_news_json import aggregate_tagged_csv_directory


if __name__ == "__main__":
    aggregate_tagged_csv_directory(
        "RAG_news_csvs_tag",
        "../../selected_news_for_llm/sinteticas_RAG_10_percent.json",
        "factual_RAG_by_news.json",
    )
