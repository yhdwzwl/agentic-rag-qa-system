# -*- coding: utf-8 -*-
"""
用baostock爬全A股上市公司列表，存MySQL，不会被反爬

2026-09-23 修复：
  1. 只保留 type=1(股票) 且 status=1(上市中)，过滤掉可转债/ETF/指数/退市股
  2. industry 原来永远为空（query_stock_basic 不返回行业），
     现在改调 query_stock_industry() 单独拉行业分类后按 code join
  3. INSERT IGNORE -> ON DUPLICATE KEY UPDATE，支持改名/换行业更新
  4. 入库后清理历史污染行（可转债 11/12 开头、ETF 51/56/58 开头）
"""
import pymysql
import baostock as bs
import pandas as pd

# 登录baostock
lg = bs.login()
print("登录baostock:", lg.error_msg)

# 连接MySQL：先不指定库，建库后再切过去
DB_NAME = 'df_rag_v4'
conn = pymysql.connect(
    host='localhost', port=3306, user='root', password='123456',
    charset='utf8mb4'
)
cursor = conn.cursor()
cursor.execute(
    f"CREATE DATABASE IF NOT EXISTS {DB_NAME} "
    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
)
conn.commit()
conn.select_db(DB_NAME)
print(f"已连接数据库: {DB_NAME}")

# 建表（已存在则跳过；company_name/exchange 是 v4 已有字段，这里补齐）
cursor.execute("""
CREATE TABLE IF NOT EXISTS company_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stock_code VARCHAR(10) UNIQUE COMMENT '股票代码',
    short_name VARCHAR(50) COMMENT '股票简称',
    company_name VARCHAR(200) COMMENT '公司全称',
    exchange VARCHAR(20) COMMENT '交易所',
    industry VARCHAR(50) COMMENT '所属行业',
    list_date VARCHAR(20) COMMENT '上市日期',
    created_time DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='全A股上市公司信息表'
""")
conn.commit()

def _fetch_all(rs):
    """兼容新版 pandas：不用 get_data()（内部用了已删除的 df.append），手动 while 循环。"""
    rows = []
    while (rs.error_code == '0') & rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)

# ---------- 1. 拉基本信息 ----------
print("拉取股票基本信息...")
rs_basic = bs.query_stock_basic()
df_basic = _fetch_all(rs_basic)
print(f"  basic 共 {len(df_basic)} 条，字段: {list(df_basic.columns)}")

# type 1=股票 2=指数 3=指数 4=可转债 5=ETF...  status: 1=上市 0=退市
df_basic = df_basic[(df_basic['type'] == '1') & (df_basic['status'] == '1')]
print(f"  过滤 type=1 & status=1 后: {len(df_basic)} 条")

# ---------- 2. 单独拉行业分类 ----------
print("拉取行业分类...")
rs_ind = bs.query_stock_industry()
df_ind = _fetch_all(rs_ind)
print(f"  industry 共 {len(df_ind)} 条，字段: {list(df_ind.columns)}")

# 按 code join：basic 为主表，左连行业
df = df_basic.merge(
    df_ind[['code', 'industry', 'industryClassification']],
    on='code', how='left'
)
print(f"  join 后 {len(df)} 条，industry 非空 {df['industry'].notna().sum()} 条")

# ---------- 3. 入库 ----------
count = 0
for _, row in df.iterrows():
    try:
        bs_code = row['code']  # sh.600000 / sz.000001
        if bs_code.startswith('sh.'):
            code = bs_code.split('.')[1]
            exchange = '上交所'
        elif bs_code.startswith('sz.'):
            code = bs_code.split('.')[1]
            exchange = '深交所'
        else:
            continue  # 北交所 bj.* 跳过

        cursor.execute("""
            INSERT INTO company_info (stock_code, short_name, exchange, industry, list_date)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                short_name = VALUES(short_name),
                exchange   = VALUES(exchange),
                industry   = VALUES(industry),
                list_date  = VALUES(list_date)
        """, (
            code,
            row['code_name'],
            exchange,
            row.get('industry') or '',
            row.get('ipoDate') or '',
        ))
        count += 1
    except Exception as e:
        print(f"存 {row['code']} 失败: {e}")

conn.commit()
print(f"UPSERT 完成: {count} 家公司")


def cleanup_pollution(cur):
    """一次性清理历史污染行（可转债/ETF/上证指数系列）。

    这个函数只在表有旧脏数据时手动调一次，不需要每次跑批都执行。
    用法：取消下面三行注释即可。
    """
    cur.execute("""
        DELETE FROM company_info
        WHERE stock_code LIKE '11%' OR stock_code LIKE '12%'
           OR stock_code LIKE '51%' OR stock_code LIKE '56%' OR stock_code LIKE '58%'
           OR stock_code LIKE '08%' OR stock_code LIKE '09%'
    """)
    # 上证指数系列代码也是 000xxx 开头，和深市股票重叠，没法按前缀删；
    # 它们 industry 全空，直接清空 industry 为空的
    cur.execute("DELETE FROM company_info WHERE industry='' OR industry IS NULL")
    return cur.rowcount


# 首次跑或怀疑有污染时取消下一行注释，跑一次后再注释回去
# deleted = cleanup_pollution(cursor)
# conn.commit()
# print(f"清理历史污染行: 删除 {deleted} 行")

bs.logout()

cursor.execute("SELECT COUNT(*) FROM company_info")
total = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM company_info WHERE industry != ''")
ind_ok = cursor.fetchone()[0]
print(f"\n表中现在共 {total} 家上市公司，其中 industry 非空 {ind_ok} 家")
conn.close()
