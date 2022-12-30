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


def LOG2DICTXT(stats, file_path, flag = 'a'):
    '''
    stats: dictionary object with stats to be logged
    '''
    with open(file_path, 'a', buffering=1) as stats_file:
        print(json.dumps(stats))
        print(json.dumps(stats), file=stats_file)


