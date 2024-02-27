import json
import argparse

def convert_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    with open(output_file, 'w') as f:
        for item in data['data']:
            for paragraph in item['paragraphs']:
                qa_list = []
                for qa in paragraph['qas']:
                    question = qa['question']
                    answers = [qa['answers'][0]['text']]
                    qa_list.append({"question": question, "answers": answers})
                line = {"context": paragraph['context'], "qas": qa_list}
                f.write(json.dumps(line) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert JSON to JSONL.')
    parser.add_argument('-i', '--input', help='Input JSON file path', required=True)
    parser.add_argument('-o', '--output', help='Output JSONL file path', required=True)
    args = parser.parse_args()

    convert_to_jsonl(args.input, args.output)