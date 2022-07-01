import random
import sqlite3

con = sqlite3.connect("tf_tmp.db", check_same_thread=False)
cur = con.cursor()

def create_table(table):
    sql = "CREATE TABLE IF NOT EXISTS " + table + "(id TEXT PRIMARY KEY, origin_text TEXT,standard_text TEXT)"
    cur.execute(sql)


def insert_into_table(table, uuidlist, all_text, all_label):
    text = "INSERT OR IGNORE INTO " + table + " VALUES (?,?,?)"
    for index, item in enumerate(all_text):
        cur.execute(text, [str(uuidlist[index]), item, all_label[index]])
    con.commit()


def search_content_db(table, uuid):
    text = "SELECT origin_text, standard_text from " + table + " where id=(?)"
    content = cur.execute(text, [uuid])
    return content

def get_uuid(text_list, seed=42):
    uuidlist = []
    if seed:
        random.seed(seed)
    for _ in range(len(text_list)):
        uuidlist.append(random.getrandbits(32))
    return uuidlist
