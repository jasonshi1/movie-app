"""Run every pipeline step, from downloading the datasets to loading the database."""

from pipeline import catalog, download, embed, enrich, load_db


def main() -> None:
    download.main()
    catalog.main()
    enrich.main()
    embed.main()
    load_db.main()


if __name__ == "__main__":
    main()
