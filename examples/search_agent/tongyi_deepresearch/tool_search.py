import pymysql
from typing import Type, List, Tuple, Dict
from pydantic import BaseModel, Field
from qwen_agent.tools.base import BaseTool, register_tool
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


class GetAllCircuitSummaries():
    name: str = "get_all_circuit_summaries"
    description: str = (
        "Retrieve the name and description of all circuits."
    )

    def call(self) -> List[Tuple[str, str]]:
        """Perform circuit information query.

        Returns:
            List[Tuple[str]]: contain (name, description)
        """
        try:
            connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD,
                                         database=DATABASE, charset='utf8mb4')
            cursor = connection.cursor()

            # 使用参数化查询防止SQL注入
            query = f"""
            SELECT name, description
            FROM {TABLE_NAME}
            where selection_evidence != ''
            """

            cursor.execute(query, [])
            results = cursor.fetchall()
            results = [{"name": row[0], "description": row[1]} for row in results]
            return json.dumps(results)
        except pymysql.Error as e:
            print(f"数据库查询错误: {e}")
            return f'ERROR: {e}'
        finally:
            if 'connection' in locals():
                connection.close()

if __name__ == '__main__':
    tool = GetAllCircuitSummaries()
    result = tool.call()
    print(result)
