import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_index import ROOT, load_model, search

LLM_NAME = "unsloth/Qwen3-0.6B-GGUF"
LLM_FILE = "Qwen3-0.6B-Q4_K_M.gguf"

SYSTEM_PROMPT = """Ты справочный помощник. Отвечай только по фрагменту из последнего сообщения.
Примеры показывают формат, но не содержат ответ на новый вопрос.
Если в найденном фрагменте нет ответа, напиши только: Я не знаю
Это правило действует даже для общеизвестных фактов. Не используй знания вне фрагмента.
Если ответ есть, кратко объясни его в два шага:
1. В источнике: «точная цитата из фрагмента на языке оригинала» [1].
2. Ответ: конкретный ответ на вопрос по-русски [1].
Не переводи цитату. Сохраняй имена и коды без изменений.
"""


def load_llm():
    cache = str(ROOT / ".cache/models")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", cache_dir=cache)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME, gguf_file=LLM_FILE, cache_dir=cache, dtype=torch.float32,
    ).eval()
    return tokenizer, model


def make_messages(question, chunk):
    example = (ROOT / "knowledge_base/nerik_valdor.md").read_text(encoding="utf-8")
    example_context = f"Фрагменты:\n[1]\n{example}\nВопрос: "
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example_context + "Какой код архива у Nerik Valdor?"},
        {"role": "assistant", "content": "1. В источнике: «Archive code: 58374F133751.» [1]\n2. Ответ: код архива Nerik Valdor — 58374F133751 [1]."},
        {"role": "user", "content": example_context + "Сколько весит Nerik Valdor?"},
        {"role": "assistant", "content": "Я не знаю"},
        {"role": "user", "content": f"Фрагмент:\n[1]\n{chunk['text']}\n\nВопрос: {question}"},
    ]


def answer_question(question, encoder, tokenizer, llm):
    chunk, score = search(encoder, question)[0]
    if score < 0.75:
        return "Я не знаю"
    inputs = tokenizer.apply_chat_template(
        make_messages(question, chunk), add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt", return_dict=True,
    )
    with torch.inference_mode():
        output = llm.generate(**inputs, max_new_tokens=220, do_sample=False)
    answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if "я не знаю" in answer.casefold():
        return "Я не знаю"
    cited = set(re.findall(r"\[(\d+)\]", answer))
    quotes = re.findall(r"«([^»]+)»", answer)
    if cited != {"1"} or not quotes or any(quote not in chunk["text"] for quote in quotes):
        return "Я не знаю"
    source = (f"[1] {chunk['source']}, фрагмент {chunk['id']}, "
              f"символы {chunk['start_char']}:{chunk['end_char']}")
    return answer + "\n\nИсточник:\n" + source


def main():
    parser = argparse.ArgumentParser(description="Консольный RAG-бот")
    parser.add_argument("question", nargs="?", help="Вопрос; без него запускается диалог")
    args = parser.parse_args()
    print("Загружаю модели...", flush=True)
    encoder = load_model()
    tokenizer, llm = load_llm()
    if args.question:
        print(answer_question(args.question, encoder, tokenizer, llm))
        return
    print("Задайте вопрос. Для выхода введите /exit.")
    while True:
        try:
            question = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question == "/exit":
            break
        if question:
            print("Бот:", answer_question(question, encoder, tokenizer, llm))


if __name__ == "__main__":
    main()
