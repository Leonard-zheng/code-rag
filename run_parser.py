# your_project_root/run_parser.py
import os
import sys
from pathlib import Path

# 确保能找到我们复制过来的模块
# 这行代码很重要，它将 'code_parser' 的父目录添加到Python的搜索路径中
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from code_parser.graph_updater import GraphUpdater
from code_parser.parser_loader import load_parsers
from code_parser.services.graph_service import MemgraphIngestor

def parse_and_ingest_repository(
    repo_path: str,
    memgraph_host: str = "localhost",
    memgraph_port: int = 7687,
    clean_db: bool = False
):
    """
    一个简单的封装函数，用于解析代码库并将其数据存入Memgraph。

    Args:
        repo_path (str): 您想解析的代码库的路径。
        memgraph_host (str): Memgraph数据库的主机地址。
        memgraph_port (int): Memgraph数据库的端口。
        clean_db (bool): 是否在入库前清空数据库。首次运行时建议设为True。
    """
    print(f"--- Starting Codebase Analysis for: {repo_path} ---")
    
    # 1. 加载所有语言的解析器
    parsers, queries = load_parsers()
    
    # 2. 连接到Memgraph
    with MemgraphIngestor(host=memgraph_host, port=memgraph_port) as ingestor:
        if clean_db:
            print("Cleaning database...")
            ingestor.clean_database()
        
        # 确保数据库约束存在
        ingestor.ensure_constraints()
        
        # 3. 初始化并运行GraphUpdater
        updater = GraphUpdater(
            ingestor=ingestor,
            repo_path=Path(repo_path),
            parsers=parsers,
            queries=queries,
        )
        
        print("Running parser and ingesting data into Memgraph...")
        updater.run() # 这会执行所有解析步骤并自动写入数据库
        
    print(f"--- Successfully parsed and ingested codebase into Memgraph! ---")