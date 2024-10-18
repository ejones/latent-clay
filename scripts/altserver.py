import http.server
import os.path
import time

import openai


base_path = os.path.dirname(os.path.abspath(__file__))

oai = openai.Client(api_key='sk-x', base_url='http://localhost:8080')
messages = []


class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server, directory='public')


    def do_POST(self):
        length = int(self.headers['content-length'])
        content = self.rfile.read(length).decode('utf8')

        self.send_response(200)
        self.end_headers()

        messages.append({'role': 'user', 'content': content})

        completion = oai.chat.completions.create(
            model='foo',
            messages=[
                {'role': 'system', 'content': open(os.path.join(base_path, 'system.txt')).read()},
                *messages,
            ],
            stream=True,
            extra_body={'cache_prompt': True},
        )

        ass_content = ''
        for chunk in completion:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is None:
                break
            ass_content += chunk_content
            self.wfile.write(chunk_content.encode('utf8'))
            self.wfile.flush()

        messages.append({'role': 'assistant', 'content': ass_content})


def main():
    port = 8000
    server = http.server.HTTPServer(('127.0.0.1', port), RequestHandler)
    print(f'serving on http://localhost:{port}')
    server.serve_forever()


if __name__ == '__main__':
    main()

