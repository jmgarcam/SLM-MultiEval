from coherence_into_news_json import aggregate_tagged_csv_directory


if __name__ == "__main__":
    aggregate_tagged_csv_directory(
        "tagged/RAG_news_csvs",
        "../../../metrics/selected_news_for_llm/sinteticas_RAG_10_percent.json",
        "coherence_RAG_by_news.json",
    )


