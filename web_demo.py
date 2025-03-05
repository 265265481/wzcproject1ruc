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










def refine_answer(input_text, chatbot, max_length, top_p, temperature, history, past_key_values, num_iterations=5):

    chatbot.append((input_text, ""))  # 初始用户输入
    # 初始输入保持不变
    current_input = input_text + "首先提醒你自己你扮演的角色是什么，自己的人设是什么，你应该以什么说话风格回应他们，根据以上问题生成一个总结以指导你之后的回答，答案总字数不得超过100字。"

    # 每次迭代时，拼接历史对话记录和从 REMEMBER.txt 读取的内容
    with open("REMEMBER.txt", "r", encoding="utf-8") as f:
        remember_content = f.read()

    # 读取 HISTORY.txt 内容
    with open("HISTORY.txt", "r", encoding="utf-8") as f:
        history_content = f.read()

    # 读取 SUMMARIZE.txt 内容
    with open("SUMMARIZE2.txt", "r", encoding="utf-8") as f:
        summarize_content = f.read()

    # 进行 num_iterations 轮交互
    for i in range(num_iterations):
        
        # 将 SUMMARIZE.txt 的内容作为提示的一部分
        combined_input = "对话记录：" + remember_content +  current_input

        # 根据迭代次数调整 current_input，这里会在每轮迭代中更新
        if i == 0:
            current_input = "现在请你根据对话历史回答：请总结刚刚的角色扮演过程中对话的内容，哪些人给你说了什么样的话？答案总字数不得超过300字"
        elif i == 1:
            current_input =  "上次对话总结：" + summarize_content + history_content + "结合之前的对话，思考：与你对话的这些人都是什么样的人，你对他们的印象如何？答案总字数不得超过200字"
        elif i == 2:
            current_input =  "上次对话总结：" + summarize_content + history_content + "结合之前的对话，思考：你在这次对话中感受如何？答案总字数不得超过150字"
        elif i == 3:
            current_input = "以下面的格式B总结这次角色扮演：1.扮演的角色。2.对话的内容总结。3.对话中的人物及其对话内容。4.对人物的印象。5.感受。答案总字数不得超过500字。"

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
        with open("REMEMBER.txt", "a", encoding="utf-8") as f:
            f.write(f"AI (Iteration {i}): {response}\n")

        # 读取更新后的对话历史
        with open("REMEMBER.txt", "r", encoding="utf-8") as f:
            remember_content = f.read()  

    if not os.path.exists("SUMMARIZE1.txt"):
        with open("SUMMARIZE1.txt", "w", encoding="utf-8") as f:
            f.write("") # 创建一个空文件
 
    # 查找和提取 "B" 后面的内容
    b_index = remember_content.find('B')
    if b_index != -1:
        content_after_b = remember_content[b_index + 1:]
    else:
        content_after_b = ""

    # 更新 SUMMARIZE.txt 文件
    with open("SUMMARIZE1.txt", "w", encoding="utf-8") as summarize_file:
        summarize_file.write(content_after_b)
    
    # 删除 REMEMBER.txt 文件
    if os.path.exists("REMEMBER.txt"):
        os.remove("REMEMBER.txt")  

    # 返回更新后的聊天记录和模型状态
    return chatbot, history, past_key_values





# 用于处理用户输入的主要预测函数
def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values):
    # 确保 REMEMBER.txt和SUMMARIZE1.txt 文件存在，如果不存在则创建
    if not os.path.exists("REMEMBER.txt"):
        with open("REMEMBER.txt", "w", encoding="utf-8") as f:
            f.write("")  # 创建一个空文件
    if not os.path.exists("SUMMARIZE1.txt"):
        with open("SUMMARIZE1.txt", "w", encoding="utf-8") as f:
            f.write("")  # 创建一个空文件

    # 获取用户输入的前两个字符
    user_input_prefix = input[:2]
    
    if user_input_prefix == "对话":
        # 如果前两个字符为 "RC"，执行 refine_answer
        chatbot, history, past_key_values = refine_answer(input, chatbot, max_length, top_p, temperature, history, past_key_values, num_iterations=5)
    elif user_input_prefix == "提问":
    # 读取 SUMMARIZE.txt 的内容
        try:
            with open("SUMMARIZE2.txt", "r", encoding="utf-8") as f:
                summarize_content = f.read().strip()
        except FileNotFoundError:
            summarize_content = ""  # 如果 SUMMARIZE2.txt 不存在，则为空字符串
    # 读取 HISTORY.txt 的内容
        try:
            with open("HISTORY.txt", "r", encoding="utf-8") as f:
                history_content = f.read().strip()
        except FileNotFoundError:
            history_content = ""  # 如果 HISTORY.txt 不存在，则为空字符串

        # 将用户输入和 SUMMARIZE.txt 内容合并为新的输入
        combined_input = summarize_content + history_content + input

        # 执行初次预测
        chatbot.append((parse_text(combined_input), ""))  # 初始用户输入
    
        for response, history, past_key_values in model.stream_chat(tokenizer, combined_input, history, 
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True,
                                                                    max_length=max_length, top_p=top_p,
                                                                    temperature=temperature):
            chatbot[-1] = (parse_text(combined_input), parse_text(response))
    
        with open("SUMMARIZE1.txt", "a", encoding="utf-8") as f:
            f.write(f"{parse_text(input)}\n")  # 追加用户输入 A
            f.write(f"{parse_text(response)}\n")  # 追加生成的响应
            
    else:
        # 如果输入不为 "RC" 或 "AQ"，可以根据需要选择默认处理方式或返回错误
        pass
    
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
            #修改max_length的上下限和初始值，只需在此处做一次修改即可(下限，上限，value=初始值，step=步长)
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