import pymysql, io
conn = pymysql.connect(host='127.0.0.1', user='root', password='123456', database='df_rag_v4', charset='utf8mb4')
cursor = conn.cursor()
cursor.execute("SELECT short_name FROM company_info WHERE short_name NOT LIKE '%%指数%%' AND short_name NOT LIKE '%%等权%%' AND short_name NOT LIKE '%%国债%%' AND short_name NOT LIKE '%%公司债%%' AND short_name NOT LIKE '%%转债%%'")
all_companies = [row[0] for row in cursor.fetchall() if row[0] and len(row[0])>=2]
conn.close()
print(f'加载了{len(all_companies)}家上市公司')

file_path = r'F:\small_project\aiagent\ailearn\rag\Itcast_qa_system\rag_qa\core\query_classifier.py'
with io.open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old_init = 'self.device = device'
new_init = f'self.device = device\n        # 全A股上市公司简称列表\n        self.company_list = {all_companies}'
content = content.replace(old_init, new_init, 1)

old_predict = '    def predict(self, query: str) -> tuple[str, float]:'
new_predict = '''    def predict(self, query: str) -> tuple[str, float]:
        # 规则匹配上市公司
        query_clean = query.strip()
        matched = [c for c in self.company_list if c in query_clean]
        if matched:
            logger.info(f'规则匹配到上市公司: {matched}，直接走RAG')
            return '股票财经', 1.0'''
content = content.replace(old_predict, new_predict, 1)

with io.open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('✅ 规则匹配加完了')
