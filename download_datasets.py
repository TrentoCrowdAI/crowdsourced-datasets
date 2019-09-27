import os
import wget
import zipfile
import tarfile
import re
import platform
from shutil import copyfile, rmtree

def download(folderName, urlDict, key):
    print("downloading ", key)
    directoryPath = os.path.join(folderName, 'data-raw')
    try:
        os.mkdir(directoryPath)
    except OSError:
        print("Creation of the directory %s failed" % directoryPath)
    url = urlDict.get(key)
    if url is None:
        url = urlDict.get('NLP Annotations')
    for file in url:
        wget.download(file, directoryPath)

def download_folders(folder, urlDict):
    for folderName, subfolders, filenames in os.walk(folder):
        for subfolder in subfolders:
            if subfolder == 'rating':
                for currentFolder, currentSubfolders, currentFiles in os.walk(os.path.join(folderName, subfolder)):
                    for dataset in currentSubfolders:
                        if dataset != "text-highlighting":
                            download(os.path.join(currentFolder, dataset), urlDict, dataset)
                    break
            elif subfolder == 'classification':
                for f, sf, files in os.walk(os.path.join(folderName, subfolder)):
                    for sub in sf:
                        for currentFolder, currentSubfolders, currentFiles in os.walk(os.path.join(f, sub)):
                            for dataset in currentSubfolders:
                                if dataset != "text-highlighting":
                                    download(os.path.join(currentFolder, dataset), urlDict, dataset)
                            break
                    break
            else:
                continue
        break

def extract_nested_archives(archivedFile, toFolder):
    if archivedFile.endswith('.zip'):
        with zipfile.ZipFile(archivedFile, 'r') as zfile:
            zfile.extractall(path=toFolder)
    elif archivedFile.endswith('.tgz'):
        tar = tarfile.open(archivedFile, "r:gz")
        tar.extractall(path=toFolder)
        tar.close()
    os.remove(archivedFile)

    for root, dirs, files in os.walk(toFolder):
        if '__MACOSX' not in root:
            for filename in files:
                if re.search(r'\.zip$', filename) or re.search(r'\.tgz$', filename):
                    fileSpec = os.path.join(root, filename)
                    extract_nested_archives(fileSpec, root)

def recursive_walk(folder, delimeter, requiredFilesList, dest):
    for folderName, subfolders, filenames in os.walk(folder):
        for file in filenames:
            if file in requiredFilesList:
                copyfile(os.path.join(folderName, file), os.path.join(dest, file))
        if subfolders:
            for subfolder in subfolders:
                recursive_walk(subfolder, delimeter, requiredFilesList, dest)

def delete_unnecessary_files(path):
    for folderName, subfolders, filenames in os.walk(path):
        for subfolder in subfolders:
            rmtree(os.path.join(path, subfolder))


if __name__ == "__main__":

    # get the current path
    path = '.'

    # define the delimeter based on the operating system
    delimeter = '/'
    if platform.system() == 'Windows':
        delimeter = '\\'

    # define the links to download datasets
    urlDict = {
        'Blue Birds': ['https://raw.githubusercontent.com/welinder/cubam/public/demo/bluebirds/gt.yaml', 'https://raw.githubusercontent.com/welinder/cubam/public/demo/bluebirds/labels.yaml'],
        'HITspam-UsingCrowdflower': ['https://raw.githubusercontent.com/ipeirotis/Get-Another-Label/master/data/HITspam-UsingCrowdflower/gold.txt', 'https://raw.githubusercontent.com/ipeirotis/Get-Another-Label/master/data/HITspam-UsingCrowdflower/labels.txt'],
        'HITspam-UsingMTurk': ['https://raw.githubusercontent.com/ipeirotis/Get-Another-Label/master/data/HITspam-UsingMTurk/gold.txt', 'https://raw.githubusercontent.com/ipeirotis/Get-Another-Label/master/data/HITspam-UsingMTurk/labels.txt'],
        'NLP Annotations': ['https://sites.google.com/site/nlpannotations/snow2008_mturk_data_with_orig_files_assembled_201904.zip'],
        'Sentiment popularity - AMT': ['https://eprints.soton.ac.uk/376544/1/SP_amt.csv'],
        '2010 Crowdsourced Web Relevance Judgments': ['https://www.ischool.utexas.edu/~ml/data/trec-rf10-crowd.tgz'],
        'AdultContent2': ['https://raw.githubusercontent.com/ipeirotis/Get-Another-Label/master/data/AdultContent2/gold.txt', 'https://raw.githubusercontent.com/ipeirotis/Get-Another-Label/master/data/AdultContent2/labels.txt'],
        'AdultContent3': ['https://raw.githubusercontent.com/ipeirotis/Get-Another-Label/master/data/AdultContent3-HCOMP2010/labels.txt'],
        'Weather Sentiment - AMT': ['https://eprints.soton.ac.uk/376543/1/WeatherSentiment_amt.csv'],
        'Toloka Aggregation Relevance 2': ['https://tlk.s3.yandex.net/dataset/TlkAgg2.zip'],
        'Toloka Aggregation Relevance 5': ['https://tlk.s3.yandex.net/dataset/TlkAgg5.zip']
    }

    # download the datasets
    download_folders(path, urlDict)

    # define the directories which stores the downloaded dataset as an archive
    pathDict = {
        'Recognizing Textual Entailment' : ['.' + delimeter + 'classification' + delimeter + 'binary-classification' + delimeter + 'Recognizing Textual Entailment' + delimeter + 'data-raw', 'snow2008_mturk_data_with_orig_files_assembled_201904.zip', ['rte.standardized.tsv','rte1.tsv']],
        'Temporal Ordering' : ['.' + delimeter + 'classification' + delimeter + 'binary-classification' + delimeter + 'Temporal Ordering' + delimeter + 'data-raw', 'snow2008_mturk_data_with_orig_files_assembled_201904.zip', ['all.tsv', 'temp.standardized.tsv']],
        '2010 Crowdsourced Web Relevance Judgments' : ['.' + delimeter + 'classification' + delimeter + 'multi-label-classification' + delimeter + '2010 Crowdsourced Web Relevance Judgments' + delimeter + 'data-raw', 'trec-rf10-crowd.tgz', ['trec-rf10-data.txt']],
        'Emotion' : ['.' + delimeter + 'rating' + delimeter + 'Emotion' + delimeter + 'data-raw', 'snow2008_mturk_data_with_orig_files_assembled_201904.zip', ['affect.tsv', 'anger.standardized.tsv', 'disgust.standardized.tsv', 'fear.standardized.tsv', 'joy.standardized.tsv', 'sadness.standardized.tsv', 'surprise.standardized.tsv', 'valence.standardized.tsv']],
        'Toloka Aggregation Relevance 2' : ['.' + delimeter + 'rating' + delimeter + 'Toloka Aggregation Relevance 2' + delimeter + 'data-raw', 'TlkAgg2.zip', ['crowd_labels.tsv', 'golden_labels.tsv']],
        'Toloka Aggregation Relevance 5': ['.' + delimeter + 'rating' + delimeter + 'Toloka Aggregation Relevance 5' + delimeter + 'data-raw', 'TlkAgg5.zip', ['crowd_labels.tsv', 'golden_labels.tsv']],
        'Word Pair Similarity': ['.' + delimeter + 'rating' + delimeter + 'Word Pair Similarity' + delimeter + 'data-raw', 'snow2008_mturk_data_with_orig_files_assembled_201904.zip', ['wordsim.standardized.tsv']]
    }

    # extract the required files from the archived datasets
    for key, value in pathDict.items():
        extract_nested_archives(value[0] + delimeter + value[1], value[0])
        recursive_walk(value[0], delimeter, value[2], value[0])
        delete_unnecessary_files(value[0])



