"""Run every pipeline step, from downloading the datasets to loading the database.

The ranking model is trained separately with pipeline.ranker, between expand and load_db,
because it needs PyTorch.
"""

from pipeline import catalog, download, embed, enrich, expand, load_db, posters, review_scores


def main() -> None:
    download.main()
    catalog.main()
    enrich.main()
    embed.main()
    expand.main()
    posters.main()
    review_scores.main()
    load_db.main()


if __name__ == "__main__":
    main()
