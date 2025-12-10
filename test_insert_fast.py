#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## 这个作为实际的插入测试，插入200000条数据，然后创建索引

import os
import pymysql
import numpy as np
import argparse
from mldr_common_tools import load_corpus, fvecs_read_yield, check_languages, get_default_query_result_dir

def test_simple_insert(lang: str):
    """插入数据测试"""
    
    # 数据库配置
    DB_CONFIG = {
        'host': '127.0.0.1',
        'port': 2881,
        'user': 'root',
        'password': '',
        'database': 'test',
        'charset': 'utf8mb4',
        'connect_timeout': 300,  # 连接超时时间（秒）
        'read_timeout': 1800,    # 读取超时时间（秒），批量插入可能需要较长时间
        'write_timeout': 1800,   # 写入超时时间（秒），批量插入可能需要较长时间
        'autocommit': False     # 手动控制事务提交
    }
    
    # 连接数据库
    print("连接数据库...")
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # 创建测试表
    print("创建测试表...")
    table_name = f"test_insert_{lang}"
    
    # 先删除已存在的表
    drop_sql = f"DROP TABLE IF EXISTS {table_name}"
    cursor.execute(drop_sql)
    
    # 创建表SQL - 先不创建索引，等数据插入完成后再创建
    create_table_sql = f"""
    CREATE TABLE {table_name} (
        id BIGINT AUTO_INCREMENT,
        base_id VARCHAR(255) NOT NULL,
        docid_col VARCHAR(255) NOT NULL,
        fulltext_col LONGTEXT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    cursor.execute(create_table_sql)
    print("测试表创建成功")
    
    # 使用流式加载语料库 - 真正逐条处理，不缓存数据
    print(f"使用流式加载语料库: {lang}")
    corpus_stream = load_corpus(lang, streaming=True)
    
    # 选择插入数量
    test_count = 200000  # 插入200000条用于测试
    print(f"将插入最多 {test_count} 条记录（流式处理）")
    
    # 批量插入SQL
    insert_sql = f"""
    INSERT INTO {table_name} (base_id, docid_col, fulltext_col)
    VALUES (%s, %s, %s)
    """
    
    # 流式处理：批量插入，提高性能
    batch_size = 1000  # 每1000条批量插入一次
    inserted_count = 0
    processed_count = 0  # 处理的数据条数（包括失败的）
    batch_data = []  # 批量数据缓存
    batch_num = 0
    
    try:
        for i, data in enumerate(corpus_stream):
            if i >= test_count:
                print(f"达到目标数量 {test_count}，停止处理")
                break
            
            processed_count = i + 1
            try:
                # 从流式数据中获取
                docid = data["docid"]
                text = data["text"]
                
                # 生成base_id 
                base_group_size = 500
                base_id_number = (i // base_group_size) + 1
                base_id = f"base_id_{base_id_number}"
                
                # 添加到批量数据列表
                batch_data.append((base_id, docid, text))
                
                # 立即释放当前数据引用
                del data, docid, text, base_id
                
                # 达到批次大小时，执行批量插入
                if len(batch_data) >= batch_size:
                    batch_num += 1
                    batch_start = (batch_num - 1) * batch_size
                    batch_end = min(batch_start + batch_size, processed_count)
                    
                    try:
                        cursor.executemany(insert_sql, batch_data)
                        conn.commit()
                        inserted_count += len(batch_data)
                        
                        if batch_num % 10 == 0:
                            print(f"✓ 批次 {batch_num} 插入成功: {len(batch_data) * 10} 条，总进度: {batch_end}/{test_count} ({batch_end*100//test_count}%)")
                        
                        # 清空批量数据，释放内存
                        batch_data.clear()
                    except Exception as e:
                        print(f"✗ 批量插入失败（批次 {batch_num}, {batch_start}-{batch_end}）: {e}")
                        import traceback
                        traceback.print_exc()
                        conn.rollback()  # 回滚失败的批次
                        batch_data.clear()  # 清空失败的数据
                        # 失败时直接抛出异常，不再继续
                        raise RuntimeError(f"批量插入失败（批次 {batch_num}, {batch_start}-{batch_end}）: {e}") from e
                
            except Exception as e:
                print(f"处理第 {i+1} 条记录失败: {e}")
                import traceback
                traceback.print_exc()
                # 失败时直接抛出异常，不再继续
                raise RuntimeError(f"处理第 {i+1} 条记录失败: {e}") from e
        
        # 插入剩余的数据
        if len(batch_data) > 0:
            batch_num += 1
            batch_start = (batch_num - 1) * batch_size
            try:
                print(f"插入最后一批数据，共 {len(batch_data)} 条")
                cursor.executemany(insert_sql, batch_data)
                conn.commit()
                inserted_count += len(batch_data)
                batch_data.clear()
                print(f"✓ 最后一批数据插入成功")
            except Exception as e:
                print(f"✗ 批量插入剩余数据失败: {e}")
                import traceback
                traceback.print_exc()
                conn.rollback()
                # 失败时直接抛出异常
                raise RuntimeError(f"批量插入剩余数据失败: {e}") from e
        
        print(f"处理完成：共处理 {processed_count} 条数据，成功插入 {inserted_count} 条记录")
        
    finally:
        # 显式释放流式数据
        del corpus_stream
        import gc
        gc.collect()
    
    # 最终提交（虽然每批次已提交，但确保所有数据都已提交）
    conn.commit()
    
    print("\n" + "="*60)
    print("验证插入结果...")
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f" 表中总记录数: {count}")
    print(f" 程序计数: {inserted_count} 条")
    print(f" 处理数量: {processed_count} 条")
    print(f" 目标数量: {test_count} 条")
    
    # 验证插入数量是否一致
    if count != inserted_count:
        print(f"\n⚠️  警告：数据库中的记录数 ({count}) 与插入计数 ({inserted_count}) 不一致！")
        print(f"   差异: {abs(count - inserted_count)} 条")
        if count < inserted_count:
            print(f"   可能原因: 进程被中断（如OOM kill），部分数据已提交但计数不准确")
        else:
            print(f"   可能原因: 有重复插入或其他异常情况")
    else:
        print(f"\n✓ 验证通过：数据库记录数与插入计数一致 ({inserted_count} 条)")
    
    if count < test_count:
        print(f"\n⚠️  注意：实际插入 {count} 条，少于目标 {test_count} 条")
        print(f"   完成度: {count*100//test_count}%")
    
    # cursor.execute(f"SELECT id, base_id, docid_col, fulltext_col FROM {table_name} LIMIT 5")
    # records = cursor.fetchall()
    # print("前5条记录:")
    # for record in records:
    #     print(f"  ID: {record[0]}, BaseID: {record[1]}, DocID: {record[2]}, Text: {record[3]}...")
    
    print("创建普通索引...")
    try:
        cursor.execute(f"CREATE INDEX idx_base_id ON {table_name}(base_id)")
        print("base_id索引创建成功")
    except Exception as e:
        print(f"base_id索引创建失败: {e}")
    
    try:
        cursor.execute(f"CREATE INDEX idx_docid ON {table_name}(docid_col)")
        print("docid索引创建成功")
    except Exception as e:
        print(f"docid索引创建失败: {e}") 
    
    # 2. 创建全文索引
    print("创建全文索引...")
    try:
        cursor.execute("SET ob_query_timeout = 3600000000")
        cursor.execute("SET ob_trx_timeout = 864000000")
        cursor.execute(f"CREATE /*+ parallel(90) */ FULLTEXT INDEX ft_fulltext ON {table_name}(fulltext_col)")
        print("全文索引创建成功")
    except Exception as e:
        print(f"全文索引创建失败: {e}")
    
    print("所有索引创建完成！")
    
    # 清理
    cursor.close()
    conn.close()
    print("测试完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='快速插入测试脚本')
    parser.add_argument('--lang', nargs='+', default=['en'], 
                       help='要处理的语言列表，支持: ar de en es fr hi it ja ko pt ru th zh')
    parser.add_argument('--all', action='store_true', 
                       help='处理所有支持的语言')
    
    args = parser.parse_args()
    
    # 确定要处理的语言列表
    if args.all:
        languages = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']
        print(f"使用 --all 参数，将处理所有 {len(languages)} 种语言")
    else:
        languages = check_languages(args.lang)
        print(f"将处理指定的 {len(languages)} 种语言: {languages}")
    
    # 逐个处理每种语言
    for lang in languages:
        print(f"\n{'='*60}")
        print(f"开始处理语言: {lang}")
        print(f"{'='*60}")
        
        try:
            test_simple_insert(lang)
            print(f"语言 {lang} 处理完成！")
        except Exception as e:
            print(f"语言 {lang} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            print(f"继续处理下一个语言...")
            continue
    
    print(f"\n{'='*60}")
    print("所有语言处理完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 