from datasets import load_dataset
import json
from tqdm import tqdm
import os
from pathlib import Path
import re
import multiprocessing as mp
import argparse

PROMPT = "<image>Please generate accurate HTML code that represents the table structure shown in input image, including any merged cells."


def td_to_th_in_thead(html_string):
    # Find the thead section
    thead_pattern = r'(<thead>.*?</thead>)'
    thead_match = re.search(thead_pattern, html_string, re.DOTALL)
    
    if thead_match:
        thead_content = thead_match.group(1)
        # Replace td with th in the thead content
        modified_thead = thead_content.replace('<td>', '<th>').replace('</td>', '</th>')
        # Replace the original thead with the modified one
        html_string = html_string.replace(thead_content, modified_thead)

    return html_string

def clean_html_table(html_content):
    # Remove head tag
    pattern_head = re.compile(r'(?s)<head>.*?</head>')
    html_content = pattern_head.sub('', html_content)

    # Remove leading whitespace at the beginning of each line
    html_content = re.sub(r'^\s+', '', html_content, flags=re.MULTILINE)

    # Remove empty lines
    html_content = re.sub(r'\n\s*\n', '\n', html_content)

    # Remove trailing whitespace at the end of each line
    html_content = re.sub(r'\s+$', '', html_content, flags=re.MULTILINE)

    # Add html and body tags
    if not re.search(r'<html.*?>.*</html>', html_content, re.DOTALL | re.IGNORECASE):
        if not re.search(r'<body.*?>.*</body>', html_content, re.DOTALL | re.IGNORECASE):
            html_content = f'<html>\n<body>\n{html_content}\n</body>\n</html>'
        else:
            html_content = f'<html>\n{html_content}\n</html>'

    return html_content.strip()


def process_sample(sample):
    messages = [
        {
            "role": "user",
            "content": PROMPT,
        },
        {
            "role": "assistant",
            "content": td_to_th_in_thead(clean_html_table(sample['html_table'])),
        }
    ]
    image_path = images_folder / f"{sample['imgid']}.jpg"
    sample['image'].save(image_path)
    return {"messages": messages, "images": [str(image_path)]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process PubTabNet dataset')
    parser.add_argument('--output_dir', type=str, default='./data/pubtabnet',
                        help='Directory to store output files')
    parser.add_argument('--num_processes', type=int, default=32,
                        help='Number of parallel processes to use')

    args = parser.parse_args()

    #ds = load_dataset("apoidea/pubtabnet-html")['train']
    #ds = ds.remove_columns('image')
    ds = load_dataset("apoidea/pubtabnet-html", split="train").select(range(2000))

    data_folder = Path(args.output_dir)
    images_folder = data_folder / 'images'
    data_folder.mkdir(parents=True, exist_ok=True)
    images_folder.mkdir(parents=True, exist_ok=True)

    output_file = data_folder / f'pubtabnet.json'

    # Create a pool of worker processes
    with mp.Pool(processes=args.num_processes) as pool:
        # Process samples in parallel
        results = list(tqdm(pool.imap(process_sample, ds), total=len(ds)))

    # Write results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"Processing complete. Output saved to {output_file}")