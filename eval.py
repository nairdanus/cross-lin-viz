import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import re

def get_metrics(path: str):
    
    data = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip().strip('()')
            elements = [re.sub(r'\(.*?\)', '', item).strip().strip("'").strip() for item in line.split(',', maxsplit=3)]
            data.append(tuple(elements))
    
    # Extract the true labels (gold) and predicted labels
    y_true = np.array([row[2] for row in data])  # 3rd column (gold labels)
    y_pred = np.array([row[3] for row in data])  # 4th column (predicted labels)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro') 
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, conf_matrix


def write_metrics(path):
    accuracy, precision, recall, f1, conf_matrix = get_metrics(path)

    out_file = path.replace(".txt", "_RESULT.txt")

    with open(out_file, "w") as f:
        f.writelines([
            f"Accuracy: {accuracy}\n",
            f"Precision: {precision}\n",
            f"Recall: {recall}\n",
            f"F1 Score: {f1}\n\n",
            f"Confusion Matrix:\n{conf_matrix}"
        ])

def test_all():
    for model in os.listdir("results"):
        model = os.path.join("results", model)
        for lan in os.listdir(model):
            lan = os.path.join(model, lan)
            for file in os.listdir(lan):
                file = os.path.join(lan, file)
                if file.endswith("RESULT.txt") or file.endswith("RESULTS.txt"):
                    continue
                write_metrics(file)


def main():
    test_all()
    for model in os.listdir("results"):
        model = os.path.join("results", model)
        for lan in os.listdir(model):
            lan = os.path.join(model, lan)
            for file in os.listdir(lan):
                file = os.path.join(lan, file)
                if not file.endswith("RESULT.txt"):
                    continue

                total_accuracy = 0
                total_precision = 0
                total_recall = 0
                total_f1 = 0

                file_count = 0

                with open(file, 'r') as f:
                    # Read the file line by line and extract the metrics
                    for line in f:
                        if line.startswith('Accuracy:'):
                            total_accuracy += float(line.split(': ')[1].strip())
                        elif line.startswith('Precision:'):
                            total_precision += float(line.split(': ')[1].strip())
                        elif line.startswith('Recall:'):
                            total_recall += float(line.split(': ')[1].strip())
                        elif line.startswith('F1 Score:'):
                            total_f1 += float(line.split(': ')[1].strip())
                    
                # Increment the file counter
                file_count += 1

                if file_count > 0:
                    avg_accuracy = total_accuracy / file_count
                    avg_precision = total_precision / file_count
                    avg_recall = total_recall / file_count
                    avg_f1 = total_f1 / file_count

                out_file = os.path.join(os.path.dirname(file), "0_RESULTS.txt")

                with open(out_file, 'w') as f: 
                    f.writelines([
                        f"Avg Accuracy: {avg_accuracy}\n",
                        f"Avg Precision: {avg_precision}\n",
                        f"Avg Recall: {avg_recall}\n",
                        f"Avg F1 Score: {avg_f1}\n\n"
                    ])
                

if __name__ == "__main__":
    main()