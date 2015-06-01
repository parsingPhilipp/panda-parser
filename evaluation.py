__author__ = 'kilian'

import sys
import experiment_database
import texttable as tt
import re
import time
import os
from collections import OrderedDict

from sqlite3 import DatabaseError


def list_experiments(exp_db):
    try:
        connection = experiment_database.initialize_database(exp_db)
        rows = experiment_database.list_experiments(connection)
        tab = tt.Texttable()
        header = ['Id', 'Terminals', 'Nonterminals', 'Rec. Part.', 'Punct.', 'Corpus train', 'Corpus test',
                  'Date (start)']
        tab.header(header)
        tab.set_cols_width([4, 9, 15, 10, 6, 26, 26, 20])
        for row in rows:
            t_row = []
            t_row.append(row[0])

            if row[1] == 'term_pos':
                t_row.append('POS')
            elif row[1] == 'term_words':
                t_row.append('Word')
            else:
                t_row.append(row[1])

            t_row.append(row[2].replace('_', ' '))
            t_row.append(row[3].replace('_', ' '))

            if row[4]:
                t_row.append('IGNORE')
            else:
                t_row.append('YES')

            # t_row.append(row[5])
            match = re.search(r'^.*/([^/]+)$', row[5])
            if match:
                t_row.append(match.group(1))
            else:
                t_row.append(row[5])

            # t_row.append(row[5])
            match = re.search(r'^.*/([^/]+)$', row[6])
            if match:
                t_row.append(match.group(1))
            else:
                t_row.append(row[6])

            time_string = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(row[7]))
            t_row.append(time_string)
            # print(t_row)

            tab.add_row(t_row)
        s = tab.draw()
        print s
        experiment_database.finalize_database(connection)
    except DatabaseError as e:
        print 'file at \'' + exp_db + '\' is not a valid experiment database:'
        print e.message


def plot_table(exp_db):
    experiments = []
    outfile = ''
    max_length = sys.maxint
    for arg in sys.argv:
        match = re.search(r'^--experiments=((?:[0-9]|,|-)+)$', arg)
        if match:
            string = match.group(1)
            while string:
                match = re.search(r'^(\d+)(?:,((?:\d|,|-)+))?$', string)
                if match:
                    experiments.append(int(match.group(1)))
                    string = match.group(2)
                    continue
                match = re.search(r'^(\d+)-(\d+)(?:,((?:\d|,|-)+))?$', string)
                if match:
                    low = int(match.group(1))
                    high = int(match.group(2))
                    if low < high:
                        experiments += range(low, high + 1)
                    else:
                        print "Error: Parse failure at " + string + ": " + str(low) + " is not smaller than " + str(
                            high)
                        exit(1)
                    string = match.group(3)
                    continue
                print "Error: Parse failure at " + string
                exit(1)

        match = re.search(r'^--outfile=(.+)$', arg)
        if match:
            path = match.group(1)
            print path
            if not os.path.isfile(exp_db):
                try:
                    open(path, 'w').close()
                    os.unlink(path)
                except IOError:
                    print 'invalid path: \'' + path + '\''
                    exit(1)

            outfile = path

        match = re.search(r'^--max-sentence-length=(\d+)$', arg)
        if match:
            max_length = int(match.group(1))

    # remove duplicates
    experiments = list(OrderedDict.fromkeys(experiments))

    # TODO: print information on what table was created
    print "exp: ", experiments
    print "out: ", outfile
    print "max_length: ", max_length
    if not outfile:
        print "No outfile specified!"
        exit(1)

    file = open(outfile, 'w')
    connection = experiment_database.initialize_database(exp_db)
    experiment_database.create_latex_table_from_database(connection, experiments, max_length, file)
    experiment_database.finalize_database(connection)
    file.close()


if __name__ == '__main__':
    # print "List of command line options:"
    command = ''
    exp_db = ''
    if len(sys.argv) > 1:
        exp_db = sys.argv[1]
    if len(sys.argv) > 2:
        command = sys.argv[2]

    if not os.path.isfile(exp_db):
        print 'file not found \'' + exp_db + '\''
    elif command == 'list':
        list_experiments(exp_db)
    elif command == 'plot':
        plot_table(exp_db)
