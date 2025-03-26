graph = "nice"

from dotenv import load_dotenv
import sqlite3

load_dotenv()

conn = sqlite3.connect('data.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM subtopics WHERE done = 0")
subtopics = cursor.fetchall()
for subtopic in subtopics:
    print(f"ID: {subtopic[0]}, Topic: {subtopic[1]}, Subtopic: {subtopic[2]}")

# with open('questions_answers.csv', 'r') as file:
#     today = datetime.today()
#     reader = csv.reader(file)
#     next(reader)
#     for row in reader:
#         cursor.execute("INSERT INTO question_answers (questions, answers, created_at, subtopic_id) VALUES (?, ?, ?, ?)", (row[0], row[1], today.strftime('%m-%d-%Y'), 0))

cursor.close()
conn.commit()
conn.close()

