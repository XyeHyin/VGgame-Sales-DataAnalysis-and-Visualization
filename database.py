from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from settings import LOGGER, POSTGRES_DEFAULTS


@dataclass
class PostgresConfig:
    host: str = POSTGRES_DEFAULTS["host"]
    port: int = POSTGRES_DEFAULTS["port"]
    database: str = POSTGRES_DEFAULTS["database"]
    user: str = POSTGRES_DEFAULTS["user"]
    password: str = POSTGRES_DEFAULTS["password"]
    table_name: str = POSTGRES_DEFAULTS["table_name"]

    def alchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}?connect_timeout=5"
        )


class PostgresWriter:
    def __init__(self, config: PostgresConfig) -> None:
        self.config = config
        self.engine: Optional[Engine] = None

    def write(self, df: pd.DataFrame) -> None:
        try:
            if self.engine is None:
                self.engine = create_engine(self.config.alchemy_url(), future=True)
            df.to_sql(
                self.config.table_name,
                self.engine,
                if_exists="replace",
                index=False,
                method="multi",
                chunksize=1000,
            )
            LOGGER.info("清洗数据已写入 PostgreSQL 表 %s", self.config.table_name)
        except SQLAlchemyError as exc:
            LOGGER.warning("写入 PostgreSQL 失败：%s", exc)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("连接 PostgreSQL 时出现非预期错误：%s", exc)


class PostgresReader:
    def __init__(self, config: PostgresConfig) -> None:
        self.config = config
        self.engine: Optional[Engine] = None

    def read(self) -> pd.DataFrame:
        try:
            if self.engine is None:
                self.engine = create_engine(self.config.alchemy_url(), future=True)
            with self.engine.connect() as connection:
                df = pd.read_sql_table(self.config.table_name, con=connection)
                # SQLAlchemy may return quoted_name objects; convert for sklearn compatibility
                df.columns = df.columns.astype(str)
            LOGGER.info(
                "已从 PostgreSQL 载入表 %s，共 %d 行", self.config.table_name, len(df)
            )
            return df
        except SQLAlchemyError as exc:
            LOGGER.error("读取 PostgreSQL 数据失败：%s", exc)
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("连接 PostgreSQL 时出现非预期错误：%s", exc)
            raise
