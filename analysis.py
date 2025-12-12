from database import PostgresConfig
from pipeline import VGSalesPipeline
from settings import CLEAN_CSV_PATH, DEFAULT_ARTIFACTS, OUTPUT_DIR


def main() -> None:
    pipeline = VGSalesPipeline(
        CLEAN_CSV_PATH,
        OUTPUT_DIR,
        artifacts=DEFAULT_ARTIFACTS,
        db_config=PostgresConfig(),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
