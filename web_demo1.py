import uvicorn
from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
import sys
import os
tokenizer = AutoTokenizer.from_pretrained("chatglm3-6b", trust_remote_code=True)
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

    chatbot.append((input_text, ""))
    current_input = input_text + "___"

    with open("SUMMARIZE1.txt", "r", encoding="utf-8") as f:
        summarize_content = f.read()

    combined_input =  current_input + "___" + summarize_content

    inputs = tokenizer(combined_input, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    chatbot.append((combined_input, response))

    with open("HISTORY.txt", "a", encoding="utf-8") as f:
        f.write(f"{response}\n")

    with open('SUMMARIZE2.txt', 'w', encoding='utf-8') as file2:
        pass 

    with open('SUMMARIZE1.txt', 'r', encoding='utf-8') as file1:
        content = file1.read()

    with open('SUMMARIZE2.txt', 'w', encoding='utf-8') as file2:
        file2.write(content)

    if os.path.exists("SUMMARIZE1.txt"):
        os.remove("SUMMARIZE1.txt")  

    return chatbot, history, past_key_values

def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values):

    if not os.path.exists("HISTORY.txt"):
        with open("HISTORY.txt", "w", encoding="utf-8") as f:
            f.write("") 

    if not os.path.exists("SUMMARIZE2.txt"):
        with open("SUMMARIZE2.txt", "w", encoding="utf-8") as f:
            f.write("")

    chatbot, history, past_key_values = refine_answer(input, chatbot, max_length, top_p, temperature, history, past_key_values)
    
    return chatbot, history, past_key_values





def reset_user_input():
    return gr.update(value='')





def reset_state():
    return [], [], None 

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM3-6B</h1>""")
    
    chatbot = gr.Chatbot()
    
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
    
    history = gr.State([])
    past_key_values = gr.State(None)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                    [chatbot, history, past_key_values], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])
    
    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

demo.queue().launch(share=False, inbrowser=True)