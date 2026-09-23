from datetime import datetime,date


from sqlalchemy import (
    String, Date, Numeric, DateTime, Integer,
    UniqueConstraint, Index, func,
)

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass



class DailyPrice(Base):
    __tablename__ = "daily_prices"

    id:             Mapped[int]  = mapped_column(Integer, primary_key=True)
    provider:       Mapped[str]  = mapped_column(String(32), nullable=False, default="yahoo")
    symbol:         Mapped[str]  = mapped_column(String(32), nullable=False)
    trade_date:     Mapped[date] = mapped_column(Date, nullable=False)
    adjusted_close: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)

    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("provider", "symbol", "trade_date",
                         name="uq_daily_prices_provider_symbol_date"),
        # Speeds up your dominant query: prices for a set of symbols in a date range.
        Index("ix_daily_prices_symbol_date", "symbol", "trade_date"),
    )