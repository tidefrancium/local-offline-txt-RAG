# local-offline-txt-RAG
【适合新手】一个单机离线私有化RAG知识库，语义分块+BM25混合检索，并封装Docker一键部署，数据不出本地，适配内网无外网场景。（不断更新中）

✨ 项目简介
语义分块 + BM25混合检索，Docker一键部署，所有数据完全不出本地，适配内网无外网环境。
区别于简单Demo：加入容错降级、向量库备份、自动化评测脚本、分层工程化架构。

## ✨核心特性
- 分层架构：core底层能力 / service业务层 / utils工具集，UI与业务解耦
- 混合检索：向量检索 + BM25，提升文档召回效果
- 健壮容错：LLM服务离线、损坏文档、超大文件均做异常降级，服务不崩溃
- 增量知识库更新：基于文件MD5，未改动文件不会重复向量化
- 运维工具：向量库定时备份脚本、批量导入文档脚本
- 自动化评测：tests目录内置评测脚本，可复现召回率、性能对比实验
- Docker Compose一键部署，离线打包交付

git clone https://github.com/tidefrancium/local-offline-txt-RAG/
cd local-offline-txt-rag
pip install -r requirements.txt
# 修改.env配置
python app.py
