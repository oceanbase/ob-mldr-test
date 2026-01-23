# MLDR 数据集测试框架

基于 [Infinity MLDR Benchmark](https://github.com/infiniflow/infinity/tree/main/python/benchmark/mldr_benchmark) 修改的多语言文档检索（MLDR）测试框架。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 项目简介

本项目是一个完整的 MLDR（Multi-Lingual Document Retrieval）数据集测试框架，用于评估和测试多语言文档检索系统的性能。支持多种查询类型（BM25、Dense Vector、混合检索等）。

## 功能特性

### 🗄️ 支持的数据库后端
- **OceanBase**：支持全文检索、向量检索、混合检索
- **seekdb**：支持全文检索、向量检索、混合检索

### 🔍 支持的查询类型

| 查询类型 | 说明 | 支持后端 |
|---------|------|---------|
| `bm25` | BM25 全文检索 | OceanBase, seekdb |
| `dense` | Dense 向量检索 | OceanBase, seekdb |
| `hybrid_dense_bm25` | Dense+BM25 混合检索 | OceanBase, seekdb |

### 📊 评估指标
- **Recall@10**：前10个结果的召回率
- **NDCG@10**：归一化折损累积增益
- **平均查询时间**：单次查询的平均响应时间

## 环境要求

### 系统要求
- **Python**：3.11 及以上
- **Java**：JDK 11 及以上（用于 pyserini）

### 依赖服务
- **OceanBase 数据库**：支持向量检索和全文检索的版本
- **seekdb 数据库**：支持向量检索和全文检索

---

## 安装指南

### 1. 安装 Java 环境

**Linux (Alibaba Cloud Linux / CentOS):**
```bash
sudo dnf install java-11-openjdk-devel -y
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
```

### 2. 创建 Python 虚拟环境

```bash
# 下载安装 Conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# 重新打开终端，初始化 Conda
source ~/miniconda3/bin/activate
conda init --all

# 新建和初始化 mdlr 所需的 Python 环境
conda create -n test python=3.11
conda activate test
```

### 3. 安装 Python 依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 4. 准备数据库

**启动 OceanBase 或 seekdb 数据库：**
```bash
# 确保数据库已启动并配置好连接信息
# 在 config.yaml 中配置对应数据库的连接参数
```

---

## 快速开始

> **⚠️ 重要提示：使用前请先配置数据库信息**
> 
> 在运行测试之前，请先完成以下配置步骤：
> 
> ```bash
> # 1. 复制示例配置文件
> cp config.yaml.example config.yaml
> 
> # 2. 编辑配置文件，将数据库信息替换为自己的配置
> vim config.yaml  
> ```
> 
> 需要修改的主要配置项：
> - `oceanbase.host`：数据库主机地址
> - `oceanbase.port`：数据库端口号
> - `oceanbase.user`：数据库用户名
> - `oceanbase.password`：数据库密码
> - `oceanbase.database`：数据库名称
> - `embedding.vector_download_url`：向量文件下载 URL（如果使用向量检索）

### 一键运行完整测试

```bash
# 英文混合检索测试
python mldr_test_runner.py --lang en --query-type hybrid_dense_bm25

# 跳过数据插入（数据已存在）
python mldr_test_runner.py --lang en --query-type hybrid_dense_bm25 --skip-insert
```

### 测试流程说明

测试框架会自动执行以下步骤：

1. **数据插入**：插入 200,000 条测试数据并创建索引
2. **预热测试**（可选）：执行一次搜索预热
3. **正式测试**：执行指定次数的搜索和评估
4. **结果输出**：显示平均 Recall@10，NDCG@10，平均时延

---

## 详细使用

### 命令行参数

```bash
python mldr_test_runner.py [OPTIONS]
```

**基本参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lang` | str | `en` | 测试语言（如：`en`、`zh` 等） |
| `--backend` | str | `oceanbase` | 数据库后端（目前仅支持 `oceanbase`） |
| `--query-type` | str | `bm25` | 查询类型，可选值：`hybrid_dense_bm25`（混合检索）、`dense`（向量检索）、`bm25`（全文检索） |
| `--skip-insert` | flag | `False` | 跳过数据插入步骤（数据已存在时使用） |
| `--result-dir` | str | `临时目录` | 结果保存目录 |
| `--config` | str | `None` | 配置文件路径（YAML 格式），默认使用当前目录下的 `config.yaml` |