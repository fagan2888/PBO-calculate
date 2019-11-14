
# created by zhaoliyuan

# all .py files in shell/db stands for different tables in database

from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class mz_markowitz_nav(Base):
    __tablename__ = 'mz_markowitz_nav'

    mz_markowitz_id = Column(String, primary_key = True)
    mz_date = Column(Date, primary_key = True)
    mz_nav = Column(Float)
    mz_inc = Column(Float)

