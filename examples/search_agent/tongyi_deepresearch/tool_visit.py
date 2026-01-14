import pymysql
from typing import Type, List, Tuple, Dict
from pydantic import BaseModel, Field
import sys
import os
import json


DB_CONFIG = {
    "host": '10.239.2.30',
    "user": 'topology_access',
    "password": 'TOpo%%14gy',
    "database": 'circuits',
    "table_name": 'circuit_info_cxmt'
}
# Database connection details
HOST = DB_CONFIG['host']
USER = DB_CONFIG['user']
PASSWORD = DB_CONFIG['password']
DATABASE = DB_CONFIG['database']
TABLE_NAME = DB_CONFIG['table_name']


class GetCircuitSpecsByName():
    name: str = "get_circuit_specs_by_name"
    description: str = (
        "Retrieve the reference specification of circuits by name"
    )

    def call(self, names) -> List[Tuple[str, str]]:
        """Perform circuit information query.

        Args:
            List[str]: contain the circuits' names which we want to query

        Returns:
            List[Tuple[str, str]]: contain (name, reference_specification)
        """
        try:
            connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD,
                                         database=DATABASE, charset='utf8mb4')
            cursor = connection.cursor()

            # 使用参数化查询防止SQL注入
            placeholders = ','.join(['%s'] * len(names))
            query = f"""
            SELECT name, reference_specification
            FROM {TABLE_NAME}
            where selection_evidence != '' and name IN ({placeholders})
            """

            cursor.execute(query, names)
            results = cursor.fetchall()
            results =  [{"name": row[0], "reference_specification": row[1]} for row in results]
            return json.dumps(results)
        except pymysql.Error as e:
            print(f"数据库查询错误: {e}")
            return f'ERROR: {e}'
        finally:
            if 'connection' in locals():
                connection.close()

if __name__ == '__main__':
   tool = GetCircuitSpecsByName()
   result = tool.call(["CM_LDO", "CM_OTA"])
   print(result)
