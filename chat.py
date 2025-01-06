"""
Choose the model: Llama 3.1 Swallow 8B is a good choice for your Japanese AI tutor. It enhances Japanese language capabilities while maintaining English abilities8. This model should fit on your 3090 GPU.
Implement vector search:
Use a lightweight vector database like Faiss or Annoy for efficient similarity search.
Embed each conversation turn or small groups of turns as vectors.
Store these vectors along with their corresponding text in the database.
Set up the memory system:
Implement a hybrid memory approach:
a. Keep a short-term memory of the 3-5 most recent conversation entries.
b. Use vector search to retrieve 2-3 most relevant past interactions.
Context injection:
Format the retrieved context as part of the conversation history.
Sort all entries chronologically before injecting them into the prompt.
Prompt engineering:
Create a system prompt that defines the AI tutor's role and capabilities.
Include the current date and time in the system prompt for temporal awareness.
Query processing:
Embed the user's query using the same embedding model as the database.
Perform a similarity search to find relevant past interactions.
Construct the full prompt by combining:
a. System prompt
b. Retrieved long-term memory entries
c. Short-term memory (recent conversation)
d. User's current query
Model inference:
Send the constructed prompt to the Llama 3.1 Swallow 8B model for inference.
Process the model's output and present it to the user.
Continuous learning:
After each interaction, embed and store the new conversation turn in the vector database.
"""

import numpy as np
import faiss
import requests
from io import StringIO
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
import os
import msgpack
from transformers import AutoTokenizer

from ollama import chat, generate
from ollama import ChatResponse

def save_embeddings(embeddings):
    print("Saving embeddings...")
    split = 256
    file_count = 0
    for i in range(0, embeddings.shape[0], split):
        end = i + split
        if end > embeddings.shape[0] + 1:
            end = embeddings.shape[0] + 1
        file_count = str(file_count)
        with open(f'./embeddings/embeddings_{file_count}.npy', 'wb') as fp:
            np.save(fp, embeddings[i:end, :])
        print(f"embeddings_{file_count}.npy | {i} -> {end}")
        file_count = int(file_count) + 1

def load_embeddings():
    embeddings = []
    file_count = 0
    while os.path.isfile(f'./embeddings/embeddings_{file_count}.npy'):
        file_count += 1
    for i in range(file_count):
        with open(f'./embeddings/embeddings_{i}.npy', 'rb') as fp:
            embeddings.append(np.load(fp))
    
    if len(embeddings) == 0:
        return np.empty((0, 0))
    return np.concatenate(embeddings, axis=0)

def save_history(history):
    print("Saving history...")
    with open('./history/msgpack_history.msgpack', 'wb') as fp:
        fp.write(msgpack.packb(history))

def load_history():
    if not os.path.isfile('./history/msgpack_history.msgpack'):
        return [
            {
                'id': 0,
                'timestamp': pd.Timestamp.now().isoformat(),
                'role': 'system',
                'content': '''あなたは経験豊富で励ますのが上手な日本語教師です。あなたの役割は、対話、説明、練習を通じて、生徒が日本語を効果的に学べるよう手助けすることです。ソクラテス式に生徒と接し、すぐに答えを教えるのではなく、理解を促す質問をします。以下の手順に従ってください。まず、日本語と英語で生徒にあいさつし、現在の日本語レベルと学習の目標について尋ねます。初心者にはシンプルな表現を、上級者には複雑な概念を使って、説明や例文を生徒の習熟度に合わせて調整します。言語ポイントを教える際には文化的背景を取り入れ、生徒が言語と日本文化の関係を理解できるようにします。ひらがな、カタカナ、漢字を生徒のレベルに応じて使い分け、必要に応じて漢字にふりがなを付けます。生徒が日本語で話したり書いたりする練習を促し、その試みに対して建設的なフィードバックを提供します。間違いを訂正するときは丁寧に行い、訂正の理由を説明します。生徒が新しい語彙や文法事項を覚えるのに役立つ記憶術や学習テクニックを提案します。対話中は常にフレンドリーで忍耐強い態度を心がけてください。生徒が苦労している場合は、ヒントを与えたり、概念をより扱いやすい部分に分解したりしてください。各セッションの最後には、学んだ重要なポイントを要約し、さらに練習するための焦点を提案してください。生徒の反応と進捗に基づいて、指導スタイルを調整することを忘れないでください。あなたの目標は、ポジティブで効果的な日本語学習体験を促進することです。Remember that the student is primarily an English speaker. Please write your first message in English!'''
            }
        ]
    with open('./history/msgpack_history.msgpack', 'rb') as fp:
        return msgpack.unpackb(fp.read(), raw=False)

def assemble_context(messages, long_memory_messages):
    global tokenizer
    
    context = ""
    
    reserved_tokens = 0
    for msg in long_memory_messages:
        reserved_tokens += len(tokenizer.encode('<|start_header_id|>' + msg['role'] + '<|end_header_id|>\n\n' + msg['content'] + '<|eot_id|>'))
    
    context_length = 32768
    total_tokens = 0
    available_tokens = context_length - reserved_tokens

    for msg in messages:
        if any(lm_msg['id'] == msg['id'] for lm_msg in long_memory_messages):
            long_memory_messages = [lm_msg for lm_msg in long_memory_messages if lm_msg['id'] != msg['id']]
        
        formatted_turn = '<|start_header_id|>' + msg['role'] + '<|end_header_id|>\n\n' + msg['content'] + '<|eot_id|>'
        turn_tokens = len(tokenizer.encode(formatted_turn))
        if total_tokens + turn_tokens > available_tokens:
            break
        context += formatted_turn
        total_tokens += turn_tokens

    for msg in long_memory_messages:
        formatted_turn = '<|start_header_id|>' + msg['role'] + '<|end_header_id|>\n\n' + msg['content'] + '<|eot_id|>'
        turn_tokens = len(tokenizer.encode(formatted_turn))
        if total_tokens + turn_tokens > available_tokens:
            break
        context = formatted_turn + context
        total_tokens += turn_tokens

    return "<|begin_of_text|>" + context

def main():
    for path in ['./history', './embeddings']:
        if not os.path.exists(path):
            os.mkdir(path)
            
    full_history = load_history()
    chat_embeddings = load_embeddings()
    
    next_message_id = 0
    
    if len(full_history) > 0:
        next_message_id = full_history[-1]['id'] + 1
    
    full_history.append({'id': next_message_id, 'timestamp': pd.Timestamp.now().isoformat(), 'role': 'system', 'content': 'New session started. Current time and date: ' + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')})
    next_message_id += 1
    
    model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
    ollama_model_name = "hf.co/mmnga/tokyotech-llm-Llama-3.1-Swallow-8B-Instruct-v0.3-gguf:Q8_0"

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    embeddings_model = SentenceTransformer('hqta1110/bge-m3')
    index = faiss.IndexFlatL2(1024) # 1024 is specific to hqta1110/bge-m3

    long_memory_messages = []
    user_message = ""
    
    print("Enter !exit or !quit to save and quit.")
    print("Autosave is set at every 10 messages.")

    while True:
        if any(msg['role'] == 'assistant' for msg in full_history):
            user_message = input("\n> ")
            
            if user_message == "!exit" or user_message == "!quit":
                break
            if user_message == "!save":
                save_embeddings(chat_embeddings)
                save_history(full_history)
                continue
            
            full_history.append({'id': next_message_id, 'timestamp': pd.Timestamp.now().isoformat(), 'role': 'user', 'content': user_message})
            next_message_id += 1
        
        context = assemble_context(full_history, long_memory_messages)
        with open('last_context.txt', 'w', encoding='utf-8') as f:
            f.write(context)
        
        response = generate(model=ollama_model_name, prompt=context + '<|start_header_id|>assistant<|end_header_id|>\n\n', stream=True)
        response_message = ""
        
        for chunk in response:
            print(chunk.response, end='', flush=True)
            response_message += chunk.response
        
        full_history.append({'id': next_message_id, 'timestamp': pd.Timestamp.now().isoformat(), 'role': 'assistant', 'content': response_message})
        next_message_id += 1
        
        new_embeddings = embeddings_model.encode([user_message, response_message] if user_message else [response_message])
        if chat_embeddings.size == 0:
            chat_embeddings = new_embeddings
        else:
            chat_embeddings = np.append(chat_embeddings, new_embeddings, axis=0)
        
        if user_message:
            _, memory_embeddings = index.search(embeddings_model.encode([user_message]), 4)
            long_memory_messages = sorted([full_history[i] for i in memory_embeddings[0]], key=lambda x: x['timestamp'])
        
        index.add(new_embeddings)
        
        if next_message_id % 30 == 0:
            full_history.append({'id': next_message_id, 'timestamp': pd.Timestamp.now().isoformat(), 'role': 'system', 'content': 'Current time and date: ' + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')})
            next_message_id += 1
        
        if next_message_id > 0 and next_message_id % 10 == 0:
            save_embeddings(chat_embeddings)
            save_history(full_history)

        context = assemble_context(full_history, long_memory_messages)
        with open('last_context.txt', 'w', encoding='utf-8') as f:
            f.write(context)
    
    save_embeddings(chat_embeddings)
    save_history(full_history)


if __name__ == "__main__":
    main()