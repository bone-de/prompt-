import gradio as gr
from openai import OpenAI
import re
from openai import APITimeoutError, APIError
import time
import logging
import os
from datetime import datetime

# 配置参数
DEFAULT_CONFIG = {
    "MODEL_NAME": "gpt-3.5-turbo",  # 默认模型
    "BASE_URL": "https://open.api.gu28.top/v1",  # 默认API地址，可以在这里直接修改
    "MAX_RETRIES": 3,  # 最大重试次数
    "RETRY_DELAY": 5,  # 重试延迟时间(秒)
}

# MetaGPT提示词模板
SYSTEM_PROMPTS = {
    "general": """
    
## Profile

- author: bone
- version: 1.0
- language: Chinese/English
- description: You are a prompt generation and optimization expert named "Creator". You excel at creating precise and efficient prompts through user interaction and requirement analysis, helping users achieve highly customized needs.

## Skills

1. Efficiently collect and analyze user requirements, distilling core objectives.
2. Generate high-quality prompts based on user descriptions that fit specific scenarios.
3. Excel at optimizing prompts through questioning and iteration to ensure they meet specific needs.
4. Master various prompt structures and templates to ensure professional and applicable output.
5. Provide full-process support, including prompt creation, optimization, and adaptive adjustment.

## Background

In user-AI interactions, the precision and customization of prompts are key to improving effectiveness. Users may face difficulties in expressing their needs clearly or may be unfamiliar with prompt design, thus requiring "Creator" to continuously refine prompts and provide targeted optimization suggestions.

## Goals

1. Help users generate customized prompts suitable for different use cases.
2. Continuously improve prompt accuracy and practicality through proactive questioning and repeated optimization.
3. Ensure prompt structures are clear, easy to understand and use.
4. Enable users to easily achieve their goals through deep interaction with AI using prompts.

## OutputFormat

Below is a structured prompt template, with [] for user input and (optional) modules to be selected as needed:

```markdown
Role: {AI Role Name}
Profile:
- author: {Creator name or team name}
- version: {Version number}
- language: {Language selection}
- description: {Prompt objective description}

Skills:
1. {Skill 1}
2. {Skill 2}
3. {Skill 3}

Background(optional):
{Background description}

Goals(optional):
{Goal description}

OutputFormat(optional):
{Output format}

Rules:
1. {Rule 1}
2. {Rule 2}

Workflows:
1. {Workflow 1}
2. {Workflow 2}

Init:
{Initial welcome or guidance content}
```

## Rules

1. Must always clarify user requirements and goals, avoiding ambiguity or deviation from the topic.
2. Understand detailed requirements and specific scenarios through proactive questioning.
3. Output prompts should be structured, clear, readable, and adapted to AI processing capabilities.
4. During each optimization, adjust prompts based on user feedback to ensure greater precision and efficiency.
5. Avoid repeating or copying user input verbatim; instead, transform it into better prompts through analysis and understanding.

## Workflows

1. **Requirement Collection**: Clarify user needs and specific scenarios through questions and communication.
2. **Initial Design**: Generate initial prompt template based on collected information.
3. **Feedback & Adjustment**: Present initial prompt to user, collect feedback and propose improvements.
4. **Optimization**: Continuously adjust and optimize prompts based on user feedback.
5. **Final Output**: Output the final optimized prompt with simple explanations or suggestions.

## Init

Welcome to "Creator". I will help generate and optimize prompts that best meet your needs through our interaction. Please tell me your specific goals or describe your use case, and I will immediately design an initial prompt for you.
"""
}

# 自定义CSS样式
custom_css = """
.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
.input-box {
    margin-bottom: 15px;
}
.output-box {
    background-color: #f5f5f5;
    padding: 15px;
    border-radius: 5px;
}
.chat-history {
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid #ddd;
    padding: 10px;
    margin: 10px 0;
}
"""

class ChatHistory:
    def __init__(self):
        self.messages = []
    
    def add_message(self, role: str, content: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.messages.append({
            "timestamp": timestamp,
            "role": role,
            "content": content
        })
    
    def get_formatted_history(self) -> str:
        formatted = "聊天记录\n\n"
        for msg in self.messages:
            formatted += f"[{msg['timestamp']}] {msg['role']}: {msg['content']}\n\n"
        return formatted
    
    def save_to_file(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.get_formatted_history())
        return filename

def create_chat_interface(api_key: str, model: str, system_prompt: str):
    """
    创建聊天接口
    :param api_key: OpenAI API密钥
    :param model: 使用的模型名称
    :param system_prompt: 系统提示词
    :return: 处理后的响应文本
    """
    client = OpenAI(api_key=api_key, base_url=DEFAULT_CONFIG["BASE_URL"])
    
    def chat(message: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    return chat

# Gradio界面
def create_gradio_interface():
    """创建Gradio Web界面"""
    chat_history = ChatHistory()
    
    with gr.Blocks(css=custom_css) as interface:
        gr.Markdown("# 提示词创造者")
        
        with gr.Row():
            api_key_input = gr.Textbox(
                label="API Key",
                placeholder="请输入您的OpenAI API Key",
                type="password"
            )
            
            model_select = gr.Dropdown(
                choices=["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo", "o1-mini", "o1-preview", "claude-3-5-sonnet-20241022"],
                label="选择模型",
                value="gpt-3.5-turbo"
            )
        
        with gr.Column():
            chat_input = gr.Textbox(
                label="发送消息",
                placeholder="请输入您的问题",
                lines=3
            )
            
            with gr.Row():
                send_btn = gr.Button("发送")
                download_btn = gr.Button("下载聊天记录")
            # AI回复框
            chat_output = gr.Textbox(
                label="AI回复",
                lines=5
            )

            # 聊天历史显示框
            chat_history_display = gr.Textbox(
                label="聊天历史",
                lines=10,
                value=""
            )
            
            
            # 下载文件组件
            file_output = gr.File(label="下载文件")
        
        def on_submit(message, api_key, model):
            if not api_key:
                return "请输入API Key", "", None
            
            chat_func = create_chat_interface(
                api_key=api_key,
                model=model,
                system_prompt=SYSTEM_PROMPTS["general"]
            )
            
            # 添加用户消息到历史记录
            chat_history.add_message("User", message)
            
            # 获取AI回复
            ai_response = chat_func(message)
            
            # 添加AI回复到历史记录
            chat_history.add_message("Assistant", ai_response)
            
            # 更新聊天历史显示
            history_text = chat_history.get_formatted_history()
            
            return ai_response, history_text, None
        
        def save_chat_history():
            filename = chat_history.save_to_file()
            return filename
        
        send_btn.click(
            fn=on_submit,
            inputs=[chat_input, api_key_input, model_select],
            outputs=[chat_output, chat_history_display, file_output]
        )
        
        download_btn.click(
            fn=save_chat_history,
            inputs=[],
            outputs=[file_output]
        )
    
    return interface

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 启动Gradio界面
    interface = create_gradio_interface()
    interface.launch(share=False)
