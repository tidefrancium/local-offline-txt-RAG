from service.kb_service import build_kb, hybrid_search
from core.llm_client import ollama_client
from utils.logger import logger
import gradio as gr

def get_retrieve_context(question: str):
    """仅执行RAG检索，提取逻辑分离"""
    if not question.strip():
        return ""
    ctx = hybrid_search(question)
    if ctx.startswith("【提示】"):
        return ""
    return ctx

def stream_chat_handle(user_msg: str, chat_history: list):
    """
    流式对话生成器
    1. 发送瞬间立刻渲染用户消息
    2. AI逐字流式输出，不用等全部生成完成
    """
    # 第一步：立刻把用户提问塞进对话，前端马上显示用户气泡
    chat_history.append({"role": "user", "content": user_msg})
    yield chat_history, ""

    # 执行RAG召回文档
    retrieve_ctx = get_retrieve_context(user_msg)
    if not retrieve_ctx:
        # 无检索内容直接返回固定话术
        chat_history.append({"role": "assistant", "content": "知识库无相关信息"})
        yield chat_history, ""
        return

    # 构建约束Prompt
    prompt_template = """你是专业文本问答助手，严格仅使用下方提供的文档信息作答。
文档中没有对应内容时，统一回复「我们换个话题聊聊吧」，禁止编造内容。
【参考文档】
{context}
【用户问题】
{question}
"""
    full_prompt = prompt_template.format(context=retrieve_ctx, question=user_msg)

    ai_total_text = ""
    # 流式迭代大模型输出，逐字更新AI气泡
    for chunk_text in ollama_client.chat_stream(full_prompt):
        ai_total_text += chunk_text
        # 更新/新增assistant消息
        if chat_history and chat_history[-1]["role"] == "assistant":
            chat_history[-1]["content"] = ai_total_text
        else:
            chat_history.append({"role": "assistant", "content": ai_total_text})
        # 实时推送界面更新 + 同步展示检索原文
        yield chat_history, retrieve_ctx


# 页面布局
with gr.Blocks(title="本地基础RAG知识库问答") as demo:
    gr.Markdown("# “蘑菇与南瓜”知识库问答系统")

    # 知识库管理折叠面板
    with gr.Accordion("⚙️ 知识库管理", open=False):
        with gr.Row():
            btn_rebuild_all = gr.Button("🔨 重建全部知识库", variant="primary")
            btn_inc_update = gr.Button("♻️ 增量更新知识库")
        build_log_box = gr.Textbox(label="执行日志", interactive=False)

    # 绑定知识库构建按钮
    btn_rebuild_all.click(fn=lambda: build_kb(incremental=False), outputs=[build_log_box])
    btn_inc_update.click(fn=lambda: build_kb(incremental=True), outputs=[build_log_box])

    # 聊天窗口
    chatbot_ui = gr.Chatbot(
    height=520,
    buttons=["copy"],
    avatar_images=("13.png", "mushroom.png")
)


    # 输入栏 + 发送按钮
    with gr.Row():
        input_box = gr.Textbox(
            label="请输入你的问题",
            placeholder="在此输入问题，回车发送",
            scale=9,
            lines=2
        )
        btn_send = gr.Button("发送", scale=1, variant="primary")

    # 检索原文折叠面板
    with gr.Accordion("🔍 检索参考上下文（RAG召回切片）", open=False):
        ref_text_box = gr.Textbox(
            label="召回文档片段",
            interactive=False,
            lines=12
        )

    btn_clear = gr.Button("清空全部对话历史")

    # 绑定发送点击事件，流式生成后清空输入框
    submit_flow = btn_send.click(
        fn=stream_chat_handle,
        inputs=[input_box, chatbot_ui],
        outputs=[chatbot_ui, ref_text_box]
    ).then(lambda: "", outputs=[input_box])

    # 回车提交绑定
    input_box.submit(
        fn=stream_chat_handle,
        inputs=[input_box, chatbot_ui],
        outputs=[chatbot_ui, ref_text_box]
    ).then(lambda: "", outputs=[input_box])

    # 清空对话
    btn_clear.click(lambda: [], None, chatbot_ui, queue=False)


if __name__ == "__main__":
    logger.info("RAG网页服务启动地址：http://127.0.0.1:7860")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft()
    )

