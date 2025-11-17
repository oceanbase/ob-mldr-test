# MLDR 数据集测试
基于 https://github.com/infiniflow/infinity/tree/main/python/benchmark/mldr_benchmark 修改

## 环境说明

1. python版本建议在3.11及以上，不低于3.9
2. pip install datasets==2.19.0
3. 测试会自动下载 mldr 语料数据集，并在本地进行缓存
4. java 版本建议不低于 11

## 安装 OpenJDK 11

```bash
sudo dnf install java-11-openjdk-devel -y
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-11.0.22.0.7-0.al8.x86_64
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
```

## 使用方法

**只需运行一个文件即可完成所有测试：**

```bash
python mldr_data_test.py --lang en --query_types bm25
```

### 参数说明

- `--lang`: 测试语言，默认 `en`（支持: ar de en es fr hi it ja ko pt ru th zh）
- `--query_types`: 查询类型，默认 `bm25`（支持: bm25, dense, sparse）
- `--db_port`: 数据库端口，默认 `2881`
- `--query_result_dir`: 查询结果保存目录，如果为 None 则使用临时目录

### 示例

```bash
# 使用默认参数（英文，bm25查询）
python mldr_data_test.py

# 指定语言和查询类型
python mldr_data_test.py --lang en --query_types bm25

# 指定结果保存目录
python mldr_data_test.py --lang en --query_types bm25 --query_result_dir ./my-results
```

## 测试流程

`mldr_data_test.py` 会自动执行以下步骤：

1. **数据插入**：插入200000条数据并创建索引
2. **预热测试**：执行一次搜索和评估（不算成绩）
3. **正式测试**：执行3次搜索和评估，取平均值
4. **结果输出**：显示 recall@10 和 QPS 指标

## 分别测试

如果需要分别运行各个测试步骤，可以按照以下方式操作：

### 步骤 1: 测试时导入数据并构建索引

```bash
python3 test_insert_fast.py --lang en
```

### 步骤 2: 进行一组测试，获得平均查询时间

```bash
python3 get_search_rrf_oceanbase.py --languages en \
  --query_types bm25 --query_result_dave_dir ./query-results
```

### 步骤 3: 计算召回率

```bash
python3 evaluate_results_oceanbase.py --languages en \
  --metrics recall@10 --query_result_dave_dir ./query-results
```

### 一体化运行（推荐）

如果需要一次预热+三组查询并自动计算平均值，可以使用一体化测试：

```bash
python3 mldr_data_test.py --lang en --query_types bm25
```

**注意**：一体化测试会自动执行上述三个步骤，并进行预热和多次测试取平均值，推荐使用此方式获得更准确的测试结果。

## 注意事项

- 确保 OceanBase 数据库已启动并可以连接（默认 127.0.0.1:2881）
- 确保已下载 qrels 文件到 `./qrels/` 目录
- 测试过程可能需要较长时间，请耐心等待