"""Chunk OpenStax crawler output and generate Alpaca-style JSONL for fine-tuning.

Heuristics used:
- Overlapping character chunks (~800 chars, overlap 200)
- Extract "Example" / "Worked Example" blocks as problem-solution pairs when nearby solution text is found
- Use section headings as concept Q&A seeds
- For explanation pairs, instruction is "Explain: {excerpt}", output is the original excerpt (OpenStax CC BY 4.0)

Usage:
python chunk_and_generate.py --input data_extraction/openstax_physics_vol1_ch1_6.json --out data_extraction/finetune_dataset.jsonl --target 5000
"""

import argparse
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup


def load_sections(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    L = len(text)
    if L <= chunk_size:
        return [text.strip()]
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks


def extract_examples_from_html(html):
    soup = BeautifulSoup(html, 'lxml')
    examples = []
    # find headings or strong labels that contain 'Example'
    for tag in soup.find_all(['h2', 'h3', 'h4', 'p', 'div']):
        text = tag.get_text(' ', strip=True)
        if re.search(r'\bExample\b|\bWorked example\b', text, re.IGNORECASE):
            # collect following sibling paragraphs as example content
            parts = [text]
            sib = tag.find_next_sibling()
            steps = 0
            while sib and steps < 6:
                t = sib.get_text(' ', strip=True)
                if t:
                    parts.append(t)
                sib = sib.find_next_sibling()
                steps += 1
            examples.append('\n'.join(parts))
    return examples


def extract_headings(html):
    soup = BeautifulSoup(html, 'lxml')
    headings = []
    for h in soup.find_all(['h1', 'h2', 'h3']):
        t = h.get_text(' ', strip=True)
        if t and len(t) < 120:
            headings.append(t)
    return headings


def is_solution_text(text):
    # heuristic: contains 'answer', 'solution', '=' or numeric with units
    if re.search(r'answer[:]?|solution[:]?|soln[:]?|=', text, re.IGNORECASE):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*(m/s|m/s\^2|N|kg|s|m)\b", text):
        return True
    return False


def generate_pairs_from_sections(sections, target_count=5000):
    dataset = []
    seen = set()

    for sec in sections:
        text = sec.get('text','')
        html = sec.get('html','')
        # 1) explanation chunks
        chunks = chunk_text(text, chunk_size=800, overlap=200)
        for c in chunks:
            instr = f"Explain: {c}"
            out = c
            key = (instr, out)
            if key not in seen:
                dataset.append({'instruction': instr, 'input': '', 'output': out, 'source': sec.get('url'), 'license': sec.get('license')})
                seen.add(key)
            if len(dataset) >= target_count:
                return dataset
        # 2) extract examples (problem -> solution)
        if html:
            examples = extract_examples_from_html(html)
            for ex in examples:
                # try to split into problem and solution heuristically
                lines = [l.strip() for l in ex.split('\n') if l.strip()]
                problem = lines[0]
                solution = '\n'.join(lines[1:]) if len(lines) > 1 else lines[0]
                # find explicit solution lines
                for l in lines:
                    if is_solution_text(l):
                        solution = l
                        break
                instr = f"Question: {problem}"
                out = solution
                key = (instr, out)
                if key not in seen:
                    dataset.append({'instruction': instr, 'input': '', 'output': out, 'source': sec.get('url'), 'license': sec.get('license')})
                    seen.add(key)
                if len(dataset) >= target_count:
                    return dataset
        # 3) headings -> concept Q&A
        if html:
            headings = extract_headings(html)
            for h in headings:
                # create simple Q/A
                q = h
                if ':' in q or len(q.split()) > 10:
                    continue
                instr = f"Q: What is {q}?"
                out = f"Definition and explanation as in OpenStax: {q}. (See source)"
                key = (instr, out)
                if key not in seen:
                    dataset.append({'instruction': instr, 'input': '', 'output': out, 'source': sec.get('url'), 'license': sec.get('license')})
                    seen.add(key)
                if len(dataset) >= target_count:
                    return dataset
    return dataset


def write_jsonl(dataset, outpath):
    p = Path(outpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        for item in dataset:
            # Alpaca-style JSONL
            rec = {'instruction': item['instruction'], 'input': item.get('input',''), 'output': item['output']}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data_extraction/openstax_physics_vol1_ch1_6.json')
    parser.add_argument('--out', default='data_extraction/finetune_dataset.jsonl')
    parser.add_argument('--target', type=int, default=5000)
    args = parser.parse_args()

    sections = load_sections(args.input)
    dataset = generate_pairs_from_sections(sections, target_count=args.target)
    write_jsonl(dataset, args.out)
    print(f"Wrote {len(dataset)} examples to {args.out}")
