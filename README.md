# 医疗药品规范智能咨询 Demo

## 运行

Windows PowerShell 在本目录运行 `./start.ps1`，然后访问 http://127.0.0.1:8000。
也可以 `pip install -r requirements.txt` 后运行 `python -m uvicorn app:app --host 127.0.0.1 --port 8000`。
Docker 可用时运行 `docker compose up --build`。Docker 配置已提供，本机演示使用 Python 启动。

## 演示路径

1. 点击“扑热息痛有哪些重复成分注意事项？”：查看名称标准化、召回分数与引用。
2. 点击“布洛芬与华法林有什么相互作用？”：查看多来源证据。
3. 点击诊断处方示例：查看前置拦截。
4. 输入“阿莫西林的注意事项”：查看证据不足拒答。
5. 左侧查看知识库和运行说明。

## 实现边界

这是轻量流程演示，不是医疗服务。真实实现 FastAPI、SSE、BM25、字符二元组余弦向量、RRF、规则精排、别名映射和规则审查。字符向量不是 BGE 语义嵌入；规则精排不是 BGE-Reranker。默认回答由检索证据摘录组成，不是 Qwen 生成。

4 条资料为已核对的 FDA/NHS 网页中文摘要，链接和核对日期保存在 knowledge.json。不是中国药典或国家药品标准，也不具备中国临床适用性验证。演示规则可能漏检或误拦，不保证医疗合规。审查完整回答后才开始 SSE 推送，避免未经审查内容先展示。

未接入 Qwen 权重、qLoRA、AWQ、Milvus、BGE、Redis、MySQL、RAGAS。没有真实模型性能或评估数据，不声称显存降低 60% 或吞吐提升 3 倍。没有持久化会话，刷新页面会清空记录。

## 可选 vLLM 接口

将 VLLM_BASE_URL 设为已有推理服务的 OpenAI 兼容基地址（通常以 /v1 结尾），可选 VLLM_MODEL 和 VLLM_API_KEY，然后重启服务。默认模型名称 Qwen/Qwen2.5-7B-Instruct。此路径需要用户已有模型服务，当前演示未实测该路径；规则审查不能证明生成忠实度或临床安全。

## API

- GET /api/health：运行模式
- GET /api/knowledge：证据集合
- POST /api/chat：JSON {"question":"问题"}；返回 stage、sources、token、done SSE 事件
- GET /docs：FastAPI 交互式接口文档

测试：在安装依赖后运行 `python test_demo.py`。
