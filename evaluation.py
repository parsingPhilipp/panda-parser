__author__ = 'kilian'

import sys
import experiment_database
import texttable as tt
import re
import time

if __name__ == '__main__':
    # print "List of command line options:"
    command = ''
    exp_db = ''
    if len(sys.argv) > 1:
        exp_db = sys.argv[1]
    if len(sys.argv) > 2:
        command = sys.argv[2]

    if exp_db and command == 'list':
        connection = experiment_database.initalize_database(exp_db)
        rows = experiment_database.list_experiments(connection)

        tab = tt.Texttable()
        header = ['Id', 'Terminals', 'Nonterminals', 'Rec. Part.', 'Punct.', 'Corpus', 'Date (start)']
        tab.header(header)
        tab.set_cols_width([4,9,15,10,6,30,20])
        for row in rows:
            t_row = []
            t_row.append(row[0])

            if row[1] == 'term_pos':
                t_row.append('POS')
            elif row[1] == 'term_words':
                t_row.append('Word')
            else:
                t_row.append(row[1])

            t_row.append(row[2].replace('_',' '))
            t_row.append(row[3].replace('_',' '))

            if row[4]:
                t_row.append('IGNORE')
            else:
                t_row.append('YES')

            # t_row.append(row[5])
            match = re.search(r'^.*/([^/]+)$',row[5])
            if match:
                t_row.append(match.group(1))
            else:
                t_row.append(row[5])

            time_string = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(row[6]))
            t_row.append(time_string)
            # print(t_row)

            tab.add_row(t_row)

        s = tab.draw()

        print s