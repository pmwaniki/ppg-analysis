import os

import warnings
import json
import pandas as pd

import sqlite3
from settings import base_dir

database_file=os.path.join(base_dir,"results.db")





# Function for saving results


def create_table3():
    conn=sqlite3.connect(database_file)
    c=conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS table3 (
        model TEXT NOT NULL,
        precision NUMERIC,
        recall NUMERIC ,
        specificity NUMERIC ,
        auc NUMERIC,
        details TEXT NOT NULL ,
        other TEXT,
        PRIMARY KEY (model,details)
    )
    """)
    conn.commit()
    conn.close()


def save_table3(model,precision,recall,specificity,auc,details,other=None):
    create_table3()

    conn = sqlite3.connect(database_file)
    c = conn.cursor()
    c.execute("""
                    INSERT OR REPLACE INTO table3 (

                        model,
                        precision,
                        recall,
                        specificity,
                        auc,
                        details,
                        other
                    ) VALUES(?,?,?,?,?,?,?)
                    """, (model, precision, recall,specificity, auc, details, other))

    conn.commit()
    conn.close()




