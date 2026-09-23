import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

database_url = os.environ["DATABASE_URL"]

# explicitly set through psycopg3:
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql+psycopg://", 1)


engine = create_engine(
    database_url,
    pool_pre_ping=True,
    pool_size=3,
    max_overflow=2
)


from sqlalchemy import text


from database.session import engine


with engine.connect() as connection:
    result = connection.execute(text("SELECT 1"))
    print(result.scalar_one())

    