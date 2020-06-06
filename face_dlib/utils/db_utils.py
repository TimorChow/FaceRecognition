"""
数据库交互模块
"""

import psycopg2
import json


class PostgresDB():
    def __init__(self):
        self.conn = psycopg2.connect("dbname=service "
                                     "user=postgres "
                                     "password=qq467003500")
        self.cur = self.conn.cursor()

    def save_record_to_db(self, _id, name, encoding):
        """
        新增用户信息
        :param _id:
        :param name:
        :param encoding:
        :return:
        """
        encoding = json.dumps(encoding.tolist())
        sql = "INSERT INTO store1 (_id, name, encoding) VALUES ({}, '{}' ,'{}')".format(_id, name, encoding)
        self.cur.execute(sql)
        self.conn.commit()

    def get_record(self):
        """
        获取已知用户的特征
        :return:
        """
        sql = "SELECT _id, name, encoding, temp_id FROM store1 limit 1000"

        self.cur.execute(sql)
        try:
            rows = self.cur.fetchall()
        except:
            rows = None

        self.conn.commit()
        return rows

    def get_info_by_id(self, _id):
        """
        根据_id获取用户的详细信息
        :param _id:
        :return:
        """
        sql = "SELECT _id,name,visit_times,age,last_visit FROM store1_details WHERE _id={}".format(_id)
        self.cur.execute(sql)
        try:
            rows = self.cur.fetchall()
        except:
            rows = None
        # self.conn.commit()
        return rows

    def get_info_by_encoding(self, encoding):
        """
        根据encoding获取用户的详细信息
        :param _id:
        :return:
        """
        sql = "SELECT _id,name,encoding, temp_id FROM store1 WHERE encoding='{}'".format(encoding)
        # sql = "SELECT _id,name,encoding, temp_id FROM store1 WHERE temp_id='{}'".format(encoding)
        self.cur.execute(sql)
        try:
            rows = self.cur.fetchall()
        except:
            rows = None
        # self.conn.commit()
        return rows


    def close(self):
        self.cur.close()
        self.conn.close()


postgres_db = PostgresDB()


def get_known_id_name_encoding():
    """
    通过访问数据库, 获取所有已知的用户的id, name和encoding特征
    :return:
    """
    result = postgres_db.get_record()
    know_face_id = [item[0] for item in result]
    known_face_names = [item[1] for item in result]
    known_face_encodings = [json.loads(item[2]) for item in result]
    known_face_tmp_id = [item[3] for item in result]

    postgres_db.close()

    return know_face_id, known_face_names, known_face_encodings, known_face_tmp_id


def get_known_id_name_encoding_from_txt():
    """
    读取文件中的数据
    :return: [id1, id2..], [name1, name2...], [encoding1, encoding2...] 返回三个分别包括id, name, encoding 的列表
    """
    face_ids = []
    face_names = []
    face_encodings = []
    with open("data/features_all.txt") as file:
        all_lines = file.readlines()
        for line in all_lines:
            data = json.loads(line)
            face_ids.append(data['_id'])
            face_names.append(data['name'])
            face_encodings.append(data['encoding'])

    return face_ids, face_names, face_encodings


def save_to_txt(_id, name, encoding):
    """
    存储数据到文件中
    :param _id:
    :param name:
    :param encoding:
    :return:
    """
    data = {'_id': _id, 'name': name, 'encoding': list(encoding)}
    path_feature = "data/features_all.txt"
    with open(path_feature, "a+") as file:
        file.writelines(json.dumps(data))
        file.writelines("\n")


if __name__ == "__main__":
    results = postgres_db.get_record()
    print(results)
    results = postgres_db.get_info_by_id(3)
    print(results)

    know_face_id, known_face_names, known_face_encodings = get_known_id_name_encoding()
    print(know_face_id, known_face_names, known_face_encodings)

    postgres_db.close()
