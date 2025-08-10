# your_project_root/your_main_app.py
from run_parser import parse_and_ingest_repository

if __name__ == "__main__":
    # 确保你的Memgraph Docker容器正在运行
    # docker-compose up -d

    # 指定你要分析的项目路径
    TARGET_REPO = "/Users/zhengzilong/code-graph/code-graph-rag"
    
    # 调用我们的封装函数，首次运行时清空数据库
    parse_and_ingest_repository(repo_path=TARGET_REPO, clean_db=True)
    
    # 解析完成后，您可以打开Memgraph Lab (http://localhost:3000)
    # 查看已经构建好的代码知识图谱！
    
    # 接下来，您可以在这里继续编写您项目的其他逻辑...