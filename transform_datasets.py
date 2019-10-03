'''
By using this tool you agree to acknowledge the original datasets and to check their terms and conditions.
Some data providers may require authentication, filling forms, etc.
We include a link to the original source of each dataset in our repository, please cite the appropriate sources in your work.
'''

import os
import pandas as pd
import re
import platform
import csv
from itertools import islice

def recursive_walk(folder, delimeter):
    for folderName, subfolders, filenames in os.walk(folder):
        dest = os.path.join(os.getcwd(), folderName.split('data-raw')[0]) + 'transformed_dataset.csv'
        if folderName == 'binary-classification' + delimeter + 'Blue Birds' + delimeter + 'data-raw':
            processBlueBirds(filenames, folderName)
        if folderName == 'binary-classification' + delimeter + 'Crowdsourced Amazon Sentiment' + delimeter + 'data-raw':
            processCrowdsourcedAmazonSentimentDataset(filenames, folderName)
        if folderName == 'binary-classification' + delimeter + 'Crowdsourced loneliness-slr' + delimeter + 'data-raw':
            processCrowdsourcedLonelinessDataset(filenames, folderName)
        elif folderName == 'binary-classification' + delimeter + 'HITspam-UsingCrowdflower' + delimeter + 'data-raw':
            processGoldAndLabelFiles(filenames, folderName, dest, None)
        elif folderName == 'binary-classification' + delimeter + 'HITspam-UsingMTurk' + delimeter + 'data-raw':
            processGoldAndLabelFiles(filenames, folderName, dest, None)
        elif folderName == 'binary-classification' + delimeter + 'Recognizing Textual Entailment' + delimeter + 'data-raw':
            processWithSeperateText(filenames, folderName, 'rte.standardized.tsv', 'rte1.tsv', dest, True)
        elif folderName == 'binary-classification' + delimeter + 'Sentiment popularity - AMT' + delimeter + 'data-raw':
            processSentiment(filenames, folderName, dest)
        elif folderName == 'binary-classification' + delimeter + 'Temporal Ordering' + delimeter + 'data-raw':
            processWithSeperateText(filenames, folderName, 'temp.standardized.tsv', 'all.tsv', dest, True)
        elif folderName == 'binary-classification' + delimeter + 'Text Highlighting' + delimeter + 'data-raw':
            processTextHighlightingDataset(filenames, folderName)
        elif folderName == 'multi-class-classification' + delimeter + '2010 Crowdsourced Web Relevance Judgments' + delimeter + 'data-raw':
            processTopicDocument(filenames, folderName, dest)
        elif folderName == 'multi-class-classification' + delimeter + 'AdultContent2' + delimeter + 'data-raw':
            processGoldAndLabelFiles(filenames, folderName, dest, None)
        elif folderName == 'multi-class-classification' + delimeter + 'AdultContent3' + delimeter + 'data-raw':
            processGoldAndLabelFiles(filenames, folderName, dest, None)
        elif folderName == 'multi-class-classification' + delimeter + 'Weather Sentiment - AMT' + delimeter + 'data-raw':
            processSentiment(filenames, folderName, dest)
        elif folderName == 'multi-class-classification' + delimeter + 'Emotion' + delimeter + 'data-raw':
            processEmotionDataset(filenames, folderName)
        elif folderName == 'binary-classification' + delimeter + 'Toloka Aggregation Relevance 2' + delimeter + 'data-raw':
            processGoldAndLabelFiles(filenames, folderName, dest, 'Toloka')
        elif folderName == 'multi-class-classification' + delimeter + 'Toloka Aggregation Relevance 5' + delimeter + 'data-raw':
            processGoldAndLabelFiles(filenames, folderName, dest, 'Toloka')
        elif folderName == 'multi-class-classification' + delimeter + 'Word Pair Similarity' + delimeter + 'data-raw':
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

def processTextHighlightingDataset(filenames, folderName):
    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent']
    df = pd.DataFrame([], columns=columns)
    for file in sorted(filenames):
        minus = 0
        if file == 'crowdsourced_highlights.csv':
            minus = 1
        with open(os.path.join(folderName, file), encoding="utf8") as csv_file:
            dest = os.path.join(os.getcwd(), folderName.split('data-raw')[0]) + 'transformed_dataset_' + file.split('.')[0] + '.csv'
            next(csv_file)
            csv_reader = csv.reader(csv_file, delimiter=',')
            for line in csv_reader:
                if line[11 - minus] == 'True':
                    row = [line[12 - minus], line[0], line[15 - minus], line[15 - minus], line[2]]
                else:
                    row = [line[12 - minus], line[0], line[15 - minus], None, line[2]]
                dfRow = pd.DataFrame([row], columns=columns)
                df = df.append(dfRow)
            df.to_csv(dest, index=None, header=True)


def processCrowdsourcedAmazonSentimentDataset(filenames, folderName):
    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent']
    for file in sorted(filenames):
        for i in range(2):
            df = pd.DataFrame([], columns=columns)
            with open(os.path.join(folderName, file), encoding="utf8") as csv_file:
                filePostfix = 'is_book'
                responseIndex = 14
                goldenIndex = 23
                if (i == 1):
                    filePostfix = 'is_negative'
                    responseIndex = 15
                    goldenIndex = 25
                dest = os.path.join(os.getcwd(),
                                    folderName.split('data-raw')[0]) + 'transformed_dataset_' + filePostfix + '.csv'
                next(csv_file)
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    row = [line[9], line[0], line[responseIndex], line[goldenIndex], line[27]]
                    dfRow = pd.DataFrame([row], columns=columns)
                    df = df.append(dfRow)
                df.to_csv(dest, index=None, header=True)


def processCrowdsourcedLonelinessDataset(filenames, folderName):
    gt_dict = {}
    for file in sorted(filenames):
        with open(os.path.join(folderName, file), encoding="utf8") as csv_file:
            next(csv_file)
            csv_reader = csv.reader(csv_file, delimiter=',')
            gt_dict = {row[0] + 'intervention': row[14] for row in islice(csv_reader, 500)}

    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent']
    for file in sorted(filenames):
        for i in range(0, 9, 4):
            df = pd.DataFrame([], columns=columns)
            with open(os.path.join(folderName, file), encoding="utf8") as csv_file:
                filePostfix = 'intervention'
                goldenIndex = 14
                if (i == 4):
                    filePostfix = 'use_of_tech'
                    goldenIndex = 15
                elif (i == 8):
                    filePostfix = 'older_adult'
                    goldenIndex = 16
                dest = os.path.join(os.getcwd(),
                                    folderName.split('data-raw')[0]) + 'transformed_dataset_' + filePostfix + '.csv'
                next(csv_file)
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    row = []
                    if (i == 0):
                        row = [line[i + 1], line[i], line[i + 2], gt_dict.get(line[i] + 'intervention'), None]
                    else:
                        row = [line[i + 1], line[i], line[i + 2], line[goldenIndex], None]
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



