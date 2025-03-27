import sqlite3
from datetime import datetime

def cleanup_data(data: str):
    cleaned_apos = data.replace("’", "'")
    return cleaned_apos.replace("—", "-")

def format_to_jsonl():
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    c.execute("SELECT questions, answers FROM question_answers")
    rows = c.fetchall()
    c.close()
    conn.close()

    today = datetime.now()
    today = today.strftime("%m-%d-%Y")

    with open(f"{today}_train.jsonl", "w") as f:
        for row in rows:
            cleaned_question = cleanup_data(row[0])
            cleaned_answer = cleanup_data(row[1])
            f.write(f'{{"instruction": "{cleaned_question}", "context": "","response": "{cleaned_answer}"}}\n')

if __name__ == "__main__":
    format_to_jsonl()
