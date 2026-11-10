"""Deterministic text projection for line references into archived arXiv HTML."""
from html.parser import HTMLParser
from pathlib import Path


class PaperText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []

    def handle_starttag(self, tag, attrs):
        if tag in ['p', 'section', 'h1', 'h2', 'h3', 'h4', 'tr', 'li', 'pre', 'br']:
            self.out.append('\n')

    def handle_data(self, data):
        self.out.append(data)


if __name__ == '__main__':
    source = Path(__file__).resolve().parent / 'sources/arxiv/2511.21631v1.html'
    parser = PaperText()
    parser.feed(source.read_text())
    source.with_suffix('.txt').write_text(''.join(parser.out))
