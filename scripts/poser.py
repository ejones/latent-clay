
import argparse
import os.path
import re
import readline
import shlex
import time

import openai

base_path = os.path.dirname(os.path.abspath(__file__))

oai = openai.Client(api_key='sk-x', base_url='http://localhost:8080')

messages = []

while line := input('> '):
    if not line.strip():
        continue
    messages.append({'role': 'user', 'content': line.strip()})

    completion = oai.chat.completions.create(
        model='foo',
        messages=[
            {'role': 'system', 'content': open(os.path.join(base_path, 'system.txt')).read()},
            *messages,
        ],
        stream=True,
    )

    ass_content = ''
    for chunk in completion:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content is None:
            break
        ass_content += chunk_content
        print(chunk_content, end='', flush=True)

    print()
    messages.append({'role': 'assistant', 'content': ass_content})

    ass_parts = re.split(r'^---.*$', ass_content, 1, re.M)
    if len(ass_parts) < 2:
        continue
