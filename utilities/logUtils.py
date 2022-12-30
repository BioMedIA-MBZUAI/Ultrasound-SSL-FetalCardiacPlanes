import json
import csv

def LOG2CSV(datalist, csv_file, flag = 'a'):
    '''
    datalist: List of elements to be written
    '''
    with open(csv_file, flag) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(datalist)
    csvFile.close()

def LOG2TXT(text, file_path, flag = 'a', console= True):
    '''
    stats: dictionary object with stats to be logged
    '''
    with open(file_path, 'a', buffering=1) as txt_file:
        if console: print(text)
        print(text, file=txt_file)


def LOG2DICTXT(dic, file_path, flag = 'a', console= True):
    '''
    stats: dictionary object with stats to be logged
    '''
    with open(file_path, 'a', buffering=1) as txt_file:
        if console: print(json.dumps(dic))
        print(json.dumps(dic), file=txt_file)


