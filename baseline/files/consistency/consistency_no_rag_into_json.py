from consistency_into_news_json import aggregate_tagged_csv_directory
if __name__ == "__main__":
    aggregate_tagged_csv_directory(
        "tagged/NO_RAG_news_csvs",
        "../../../metrics/selected_news_for_llm/sinteticas_NORAG_10_percent.json",
        "consistency_NO_RAG_by_news.json",
    )