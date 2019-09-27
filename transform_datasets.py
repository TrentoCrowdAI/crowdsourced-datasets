import os
import pandas as pd
import re
import platform

def recursive_walk(folder, delimeter):
    for folderName, subfolders, filenames in os.walk(folder):
        dest = os.path.join(os.getcwd(), folderName.split('data-raw')[0]) + 'transformed_dataset.csv'
        if folderName == 'classification' + delimeter + 'binary-classification' + delimeter + 'Blue Birds' + delimeter + 'data-raw':
            processBlueBirds(filenames, folderName)
        elif folderName == 'classification' + delimeter + 'binary-classification' + delimeter + 'HITspam-UsingCrowdflower' + delimeter + 'data-raw':
            processGoldAndLabelFiles(filenames, folderName, dest, None)
        elif folderName == 'classification' + delimeter + 'binary-classification' + delimeter + 'HITspam-UsingMTurk' + delimeter + 'data-raw':
            processGoldAndLabelFiles(filenames, folderName, dest, None)
        elif folderName == 'classification' + delimeter + 'binary-classification' + delimeter + 'Recognizing Textual Entailment' + delimeter + 'data-raw':
            processWithSeperateText(filenames, folderName, 'rte.standardized.tsv', 'rte1.tsv', dest, True)
        elif folderName == 'classification' + delimeter + 'binary-classification' + delimeter + 'Sentiment popularity - AMT' + delimeter + 'data-raw':
            processSentiment(filenames, folderName, dest)
        elif folderName == 'classification' + delimeter + 'binary-classification' + delimeter + 'Temporal Ordering' + delimeter + 'data-raw':
            processWithSeperateText(filenames, folderName, 'temp.standardized.tsv', 'all.tsv', dest, True)
        elif folderName == 'classification' + delimeter + 'multi-label-classification' + delimeter + '2010 Crowdsourced Web Relevance Judgments' + delimeter + 'data-raw':
            processTopicDocument(filenames, folderName, dest)
        elif folderName == 'classification' + delimeter + 'multi-label-classification' + delimeter + 'AdultContent2' + delimeter + 'data-raw':
            processGoldAndLabelFiles(filenames, folderName, dest, None)
        elif folderName == 'classification' + delimeter + 'multi-label-classification' + delimeter + 'AdultContent3' + delimeter + 'data-raw':
            processGoldAndLabelFiles(filenames, folderName, dest, None)
        elif folderName == 'classification' + delimeter + 'multi-label-classification' + delimeter + 'Weather Sentiment - AMT' + delimeter + 'data-raw':
            processSentiment(filenames, folderName, dest)
        elif folderName == 'rating' + delimeter + 'Emotion' + delimeter + 'data-raw':
            processEmotionDataset(filenames, folderName)
        elif folderName == 'rating' + delimeter + 'Toloka Aggregation Relevance 2' + delimeter + 'data-raw':
            processGoldAndLabelFiles(filenames, folderName, dest, 'Toloka')
        elif folderName == 'rating' + delimeter + 'Toloka Aggregation Relevance 5' + delimeter + 'data-raw':
            processGoldAndLabelFiles(filenames, folderName, dest, 'Toloka')
        elif folderName == 'rating' + delimeter + 'Word Pair Similarity' + delimeter + 'data-raw':
            processWithSeperateText(filenames, folderName, 'wordsim.standardized.tsv', None, dest, False)
        else:
            for subfolder in subfolders:
                recursive_walk(subfolder, delimeter)

def processBlueBirds(filenames, folderName):
    gt_dict = {}
    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent']
    df = pd.DataFrame([], columns=columns)
    for file in sorted(filenames):
        if file == 'gt.yaml':
            for title, block in blueBirdBlocks(os.path.join(folderName, file)):
                block = block.replace('{', '')
                block = block.replace('}', '')
                block = block.replace('\n', '')
                block = block.replace(' ', '')
                gt_dict = dict(item.split(":") for item in block.split(","))
        if file == 'labels.yaml':
            for title, block in blueBirdBlocks(os.path.join(folderName, file)):
                block = block.replace('\n', '')
                splitted_block = block.split(": {")
                splitted_block[1] = splitted_block[1].replace('}', '')
                new_dict = dict(item.split(": ") for item in splitted_block[1].split(","))
                for key in new_dict.keys():
                    row = [splitted_block[0], key.strip(), new_dict.get(key), gt_dict.get(key.strip()), None]
                    dfRow = pd.DataFrame([row], columns=columns)
                    df = df.append(dfRow)

    df.to_csv(os.path.join(os.getcwd(), folderName.split('data-raw')[0]) + 'transformed_dataset.csv', index=None, header=True)


def blueBirdBlocks(filename):
    title, block = '', None
    with open(filename) as fp:
        for line in fp:
            if '{' in line:
                block = line
            elif block is not None:
                block += line
            else:
                title = line
            if '}' in line:
                yield title, block
                title, block = '', None

def processGoldAndLabelFiles(filenames, folderName, dest, dataset):
    if dataset == 'Toloka':
        filenames = sorted(filenames, reverse=True)
    else:
        filenames = sorted(filenames)
    gt_dict = {}
    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent']
    df = pd.DataFrame([], columns=columns)
    for file in filenames:
        if file == 'gold.txt' or file == 'golden_labels.tsv':
            with open(os.path.join(folderName, file)) as fp:
                rows = ( line.split('\t') for line in fp )
                gt_dict = { row[0]:row[1] for row in rows }
        if file == 'labels.txt' or file == 'crowd_labels.tsv':
            with open(os.path.join(folderName, file)) as fp:
                for line in fp:
                    label_list = re.split(r'\t+', line)
                    goldLabel = None
                    if gt_dict.get(label_list[1]) != None:
                        goldLabel = gt_dict.get(label_list[1]).strip()
                    row = [label_list[0], label_list[1], label_list[2].strip(), goldLabel, None]
                    dfRow = pd.DataFrame([row], columns=columns)
                    df = df.append(dfRow)
    df.to_csv(dest, index=None, header=True)

def processWithSeperateText(filenames, folderName, file1, file2, dest, isTextExist):
    if file2 == 'rte1.tsv':
        filenames = sorted(filenames)
    else:
        filenames = sorted(filenames, reverse=True)
    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent']
    df = pd.DataFrame([], columns=columns)
    for file in filenames:
        with open(os.path.join(folderName, file)) as fp:
            next(fp)
            for line in fp:
                label_list = re.split(r'\t+', line)
                if file == file1:
                    row = [label_list[1], label_list[2], label_list[3], label_list[4].strip(), None]
                    dfRow = pd.DataFrame([row], columns=columns)
                    df = df.append(dfRow)
                if file == file2 and isTextExist:
                    if file2 == 'rte1.tsv':
                        df.loc[df['taskID'] == label_list[0], ['taskContent']] = label_list[3]
                    else:
                        df.loc[df['taskID'] == label_list[0], ['taskContent']] = label_list[4]
    df.to_csv(dest, index=None, header=True)

def processWithSeperateText(filenames, folderName, file1, file2, dest, isTextExist):
    if file2 == 'rte1.tsv':
        filenames = sorted(filenames)
    else:
        filenames = sorted(filenames, reverse=True)
    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent']
    df = pd.DataFrame([], columns=columns)
    for file in filenames:
        with open(os.path.join(folderName, file)) as fp:
            next(fp)
            for line in fp:
                label_list = re.split(r'\t+', line)
                if file == file1:
                    row = [label_list[1], label_list[2], label_list[3], label_list[4].strip(), None]
                    dfRow = pd.DataFrame([row], columns=columns)
                    df = df.append(dfRow)
                if file == file2 and isTextExist:
                    if file2 == 'rte1.tsv':
                        df.loc[df['taskID'] == label_list[0], ['taskContent']] = label_list[3]         
                    else:
                        df.loc[df['taskID'] == label_list[0], ['taskContent']] = label_list[4]         
    df.to_csv(dest, index=None, header=True)

def processSentiment(filenames, folderName, dest):
    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent', 'timeSpent']
    df = pd.DataFrame([], columns=columns)
    for file in sorted(filenames):
        with open(os.path.join(folderName, file)) as fp:
            for line in fp:
                label_list = re.split(r',', line)
                row = [label_list[0], label_list[1], label_list[2], label_list[3], None, label_list[4]]
                dfRow = pd.DataFrame([row], columns=columns)
                df = df.append(dfRow)
    df.to_csv(dest, index=None, header=True)
    
def processTopicDocument(filenames, folderName, dest):
    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent']
    df = pd.DataFrame([], columns=columns)
    for file in sorted(filenames):
        if file == 'trec-rf10-data.txt':
            with open(os.path.join(folderName, file)) as fp:
                next(fp)
                for line in fp:
                    label_list = re.split(r'\t+', line)
                    task = [label_list[0], label_list[2]]
                    row = [label_list[1], '_'.join(task), label_list[4].strip(), label_list[3], None]
                    dfRow = pd.DataFrame([row], columns=columns)
                    df = df.append(dfRow)
    df.to_csv(dest, index=None, header=True)

def processEmotionDataset(filenames, folderName):
    text_dict = {}
    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent']
    df = pd.DataFrame([], columns=columns)
    for file in sorted(filenames):
        if file == "affect.tsv":
            with open(os.path.join(folderName, file)) as fp:
                rows = ( line.split('\t') for line in fp )
                text_dict = { row[0]:row[1] for row in rows }
        else:
            dest = os.path.join(os.getcwd(), folderName.split('data-raw')[0]) + 'transformed_dataset_' + file.split('.')[0] + '.csv'
            with open(os.path.join(folderName, file)) as fp:
                next(fp)
                for line in fp:
                    label_list = re.split(r'\t+', line)
                    row = [label_list[1], label_list[2], label_list[3], label_list[4].strip(), text_dict.get(label_list[2])]
                    dfRow = pd.DataFrame([row], columns=columns)
                    df = df.append(dfRow)
            df.to_csv(dest, index=None, header=True)

if __name__ == '__main__':
    # get the current path
    path = '.'

    # define the delimeter based on the operating system
    delimeter = '/'
    if platform.system() == 'Windows':
        delimeter = '\\'

    # walk through the directories and transform all the datasets
    recursive_walk(path, delimeter)



