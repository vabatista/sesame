from nltk.tokenize import sent_tokenize
import json
import csv

class SentenceExtractor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_dataset(self):
        with open(self.dataset_path, 'r') as file:
            json_list = list(file)
        
        data = []
        # [1:] removes the header
        for item in json_list[1:]:
            item = json.loads(item)
            data.append(item)
        return data

    def get_sentences(self, data):
        sentences = []
        for item in data:
            context = item['context']
            sentences.extend(sent_tokenize(context))
        return sentences

    def save_sentences_to_csv(self, sentences, output_file):
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for sentence in sentences:
                writer.writerow([sentence])

    def run(self):
        data = self.load_dataset()
        sentences = self.get_sentences(data)
        return sentences

if __name__ == "__main__":
    extractor = SentenceExtractor('qa-datasets/originals/DROP-dev_modified.jsonl')
    sentences = extractor.run()
    extractor.save_sentences_to_csv(sentences, 'qa-datasets/azure/DROP-dev_modified_sentences.csv')
    print(len(sentences))