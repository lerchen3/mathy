%pip install -U -q "google-generativeai>=0.8.3"
from IPython.display import HTML, Markdown, display
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')
from google.api_core import retry
retry_policy = {
    "retry": retry.Retry(predicate=retry.if_transient_error, initial=10, multiplier=1.5, timeout=300)
}
import pandas as pd
math = pd.read_csv('/kaggle/input/un-parsed-soluions-file/math_competitions_solutions.csv')

import re
import ast
import pandas as pd

All_questions = []

for pdf_text in math['text']:
    # Split pdf_text into chunks of around 4096 words
    words = pdf_text.split()
    chunk_size = 4096
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    for chunk in chunks:
        prompt = (
            "Extract and write in LaTeX all of the question-answers pairs of the following text "
            "in a python list of tuples. Skip any questions that you cannot write LaTeX for, or do "
            "not make sense with what you are given. Return only the python list of tuples.\n\n"
            "Requirements:\n"
            "- Questions should be self-contained (no context needed)\n"
            "- All answers must be integers\n"
            "- Include any necessary integer extraction instructions in the question\n"
            "- Use proper LaTeX formatting\n\n"
            "Format: [(r'question_in_latex', integer_answer)]\n"
            "Example: [(r'What is $\sqrt{50} + \sqrt{50}$? If your answer can be expressed in the form $a\sqrt{b}$, where "
            "$a$ and $b$ are positive integers such that $b$ is not divisible by any perfect square, compute $a+b$.', 12)]\n\n"
            "Text to process: " + chunk
        )
        
        response = model.generate_content(prompt, request_options=retry_policy)
        text = response.text
        
        # Extract text between the first '[' and the last ']'
        start_index = text.find('[')
        end_index = text.rfind(']')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            code_block = text[start_index:end_index+1].strip()
            # Attempt to safely evaluate the code block
            try:
                # Use regex to find all tuples in the code block
                pattern = re.compile(r"\(r?'(.*?)',\s*r?'(.*?)'\)")
                matches = pattern.findall(code_block)
                # Convert matches to a list of tuples
                qs_pairs = [(q, a) for q, a in matches]
                All_questions.extend(qs_pairs)
            except Exception as e:
                print("Error evaluating response:")
                print(code_block)
                print(f"Exception: {e}")
        else:
            print("No list found in response:")
            print(text)

# Save All_questions to CSV
df = pd.DataFrame(All_questions, columns=['Question', 'Answer'])
df.to_csv('allquestionanswers.csv', index=False)
print("Questions and answers saved!")