import uvicorn
from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
import sys
import os

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("chatglm3-6b", trust_remote_code=True)

# 根据命令行参数选择加载不同精度的模型
match sys.argv[1]:
    case "FP16":
        model = AutoModel.from_pretrained("chatglm3-6b", trust_remote_code=True).half().cuda()
    case "INT8":
        model = AutoModel.from_pretrained("chatglm3-6b", trust_remote_code=True).half().quantize(8).cuda()
    case "INT4":
        model = AutoModel.from_pretrained("chatglm3-6b", trust_remote_code=True).half().quantize(4).cuda()
    case "CPU32":
        model = AutoModel.from_pretrained("chatglm3-6b", trust_remote_code=True).float()
    case _:
        model = AutoModel.from_pretrained("chatglm3-6b", trust_remote_code=True).bfloat16()
model = model.eval()

# 自定义postprocess来渲染Markdown为HTML
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

gr.Chatbot.postprocess = postprocess

# 解析输入文本中的Markdown格式并转为HTML
def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text










def refine_answer(input_text, chatbot, max_length, top_p, temperature, history, past_key_values):

    chatbot.append((input_text, ""))  # 初始用户输入
    # 初始输入保持不变
    current_input = input_text + "总结下面的对话，必须包含你扮演的角色性格所有角色及其语句的总结，不超过100字。"

    # 读取 SUMMARIZE.txt 内容
    with open("SUMMARIZE1.txt", "r", encoding="utf-8") as f:
        summarize_content = f.read()

    # 将 SUMMARIZE.txt 的内容作为提示的一部分
    combined_input =  current_input + "对话记录：" + summarize_content

    # 使用模型生成响应
    inputs = tokenizer(combined_input, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
        do_sample=True
    )

    # 解码模型生成的响应
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 更新聊天记录
    chatbot.append((combined_input, response))

    # 仅将当前的 AI 回复记录到文件中
    with open("HISTORY.txt", "a", encoding="utf-8") as f:
        f.write(f"{response}\n")

    # 先清空SUMMARIZE2.txt的内容（如果文件存在）
    with open('SUMMARIZE2.txt', 'w', encoding='utf-8') as file2:
        pass  # 以写模式打开文件，会自动清空文件内容

        # 打开SUMMARIZE1.txt并读取内容
    with open('SUMMARIZE1.txt', 'r', encoding='utf-8') as file1:
        content = file1.read()

    # 将内容写入SUMMARIZE2.txt
    with open('SUMMARIZE2.txt', 'w', encoding='utf-8') as file2:
        file2.write(content)

    
    # 删除 SUMMARIZE.txt 文件
    if os.path.exists("SUMMARIZE1.txt"):
        os.remove("SUMMARIZE1.txt")  

    # 返回更新后的聊天记录和模型状态
    return chatbot, history, past_key_values





# 用于处理用户输入的主要预测函数
def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values):
    # 确保 HISTORY.txt和SUMMARIZE.txt 文件存在，如果不存在则创建
    if not os.path.exists("HISTORY.txt"):
        with open("HISTORY.txt", "w", encoding="utf-8") as f:
            f.write("")  # 创建一个空文件
    # 确保 HISTORY.txt和SUMMARIZE.txt 文件存在，如果不存在则创建
    if not os.path.exists("SUMMARIZE2.txt"):
        with open("SUMMARIZE2.txt", "w", encoding="utf-8") as f:
            f.write("")  # 创建一个空文件

    chatbot, history, past_key_values = refine_answer(input, chatbot, max_length, top_p, temperature, history, past_key_values)
    
    return chatbot, history, past_key_values










# 清空用户输入框
def reset_user_input():
    return gr.update(value='')

# 清空聊天记录和状态
def reset_state():
    return [], [], None  # 重置为初始状态

# Gradio界面构建
with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM3-6B</h1>""")
    
    # 聊天组件
    chatbot = gr.Chatbot()
    
    # 输入区
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 30000, value=8000, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
    
    # 状态存储历史对话和模型的状态
    history = gr.State([])
    past_key_values = gr.State(None)
    
    # 提交按钮事件
    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                    [chatbot, history, past_key_values], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])
    
    # 清空聊天记录按钮事件
    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

# 启动Gradio应用
demo.queue().launch(share=False, inbrowser=True)