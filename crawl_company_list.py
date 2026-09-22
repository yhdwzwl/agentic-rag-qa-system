# -*- coding: utf-8 -*-
"""
用baostock爬全A股上市公司列表，存MySQL，不会被反爬
"""
import pymysql
import baostock as bs
import pandas as pd

# 登录baostock
lg = bs.login()
print("登录baostock:", lg.error_msg)

# 连接MySQL
conn = pymysql.connect(
    host='localhost', port=3306, user='root', password='123456',
    database='df_rag_v3', charset='utf8mb4'
)
cursor = conn.cursor()

# 建表
cursor.execute("""
CREATE TABLE IF NOT EXISTS company_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stock_code VARCHAR(10) UNIQUE COMMENT '股票代码',
    short_name VARCHAR(50) COMMENT '股票简称',
    industry VARCHAR(50) COMMENT '所属行业',
    list_date VARCHAR(20) COMMENT '上市日期',
    created_time DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='全A股上市公司信息表'
""")
conn.commit()

print("开始拉取全A股列表...")
rs = bs.query_stock_basic()
print(f"拉取完成，错误码: {rs.error_code}, 错误信息: {rs.error_msg}")

# 读数据
data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())

df = pd.DataFrame(data_list, columns=rs.fields)
print(f"总共拉了{len(df)}条数据")
print(df.head())

# 存数据库
count = 0
for _, row in df.iterrows():
    try:
        bs_code = row['code']  # 格式是sh.600000 / sz.000001
        # 转成纯6位代码
        if bs_code.startswith('sh.'):
            code = bs_code.split('.')[1]
            exchange = '上交所'
        elif bs_code.startswith('sz.'):
            code = bs_code.split('.')[1]
            exchange = '深交所'
        else:
            continue
        
        cursor.execute("""
            INSERT IGNORE INTO company_info (stock_code, short_name, industry, list_date)
            VALUES (%s, %s, %s, %s)
        """, (
            code,
            row['code_name'],
            row.get('industry', ''),
            row.get('listingDate', ''),
        ))
        count += 1
    except Exception as e:
        print(f"存{row['code']}失败: {e}")

conn.commit()
bs.logout()

print(f"成功存了{count}家公司！")
cursor.execute("SELECT COUNT(*) FROM company_info")
total = cursor.fetchone()[0]
print(f"表中现在总共有{total}家上市公司")
conn.close()
