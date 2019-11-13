
# created by zhaoliyuan

# all .py files in shell/db stands for different tables in database

from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class rank_z_score_descriptor_return_1021(Base):
    __tablename__ = 'rank_z_score_descriptor_return_1021'

    descirptor = Column(Integer, primary_key = True)
     = Column(String)
    fi_wind_id = Column(String)
    fi_name = Column(String)
    fi_first_raise = Column(Float)
    fi_laste_raise = Column(Float)

