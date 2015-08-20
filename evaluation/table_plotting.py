import sys
from evaluation.eval_pl_scorer import eval_pl_scores
from evaluation.experiment_database import nontlabelling_strategies, recpac_stategies, punct, fanout, \
    scores_and_parse_time, recognised_sentences_lesseq_than, parse_time_trees_lesseq_than, percentify, \
    test_sentences_length_lesseq_than, all_recognised_sentences_lesseq_than, common_recognised_sentences, openDatabase, \
    sampledb, finalize_database

__author__ = 'kilian'


def compute_line(connection, ids, exp, max_length):
    line = {}

    cursor = connection.cursor()
    experiment = cursor.execute(
        'SELECT nont_label, rec_par, training_corpus, test_corpus, ignore_punctuation, term_label FROM experiments WHERE e_id = ?',
        (exp,)).fetchone()
    g_id, nont, rules = cursor.execute('SELECT g_id, nonterminals, rules FROM grammar WHERE experiment = ?',
                                       (exp,)).fetchone()
    fanouts = cursor.execute('SELECT fanout, nonterminals FROM fanouts WHERE g_id = ?', (g_id,)).fetchall()

    line['nont_labelling'] = nontlabelling_strategies(experiment[0])
    line['rec_par'] = recpac_stategies(experiment[1])
    line['training_corpus'] = experiment[2]
    test_corpus = experiment[3]
    line['punc'] = punct(experiment[4])
    line['n_nonterminals'] = nont
    line['n_rules'] = rules
    line['term_labelling'] = experiment[5].replace('_', '-')
    # line['fanout'] = 'fanout'
    line['f1'] = fanout(fanouts, 1)
    line['f2'] = fanout(fanouts, 2)
    line['f3'] = fanout(fanouts, 3)
    line['f4'] = fanout(fanouts, 4)
    line['f5'] = fanout(fanouts, 5)

    UAS_a, LAS_a, UAS_t, LAS_t, time_on_int = scores_and_parse_time(connection, ids, exp)

    recogn_ids = recognised_sentences_lesseq_than(connection, exp, max_length, test_corpus)
    UAS_c_a, LAS_c_a, UAS_c_t, LAS_c_t, _ = scores_and_parse_time(connection, recogn_ids, exp)

    total_parse_time = parse_time_trees_lesseq_than(connection, exp, max_length, test_corpus)

    precicion = 1

    percent = lambda x: percentify(x, precicion)
    line['LAS_e'], line['UAS_e'], line['LAc_e'] = tuple(
        map(percent, eval_pl_scores(connection, test_corpus, exp, recogn_ids)))
    line['LAS^t_e'], line['UAS^t_e'], line['LAc^t_e'] = tuple(
        map(percent, eval_pl_scores(connection, test_corpus, exp)))
    line['LAS^c_e'], line['UAS^c_e'], line['LAc^c_e'] = tuple(
        map(percent, eval_pl_scores(connection, test_corpus, exp, ids)))

    # UAS_c_a, LAS_c_a, UAS_c_t, LAS_c_t, LEN_c = 0, 0, 0, 0, 0
    # for id in recogn_ids:
    # c, l , uas_a = uas(connection, id, exp)
    # UAS_c_a = UAS_c_a + uas_a
    # LEN_c = LEN_c + l
    # UAS_c_t = UAS_c_t + c
    #
    # cl, _, las_a = las(connection, id, exp)
    # LAS_c_a += las_a
    # LAS_c_t += cl
    # UAS_c_a = UAS_c_a / len(recogn_ids)
    # LAS_c_a = LAS_c_a / len(recogn_ids)
    # UAS_c_t = 1.0 * UAS_c_t / LEN_c
    # LAS_c_t = 1.0 * LAS_c_t / LEN_c

    # line['test_total'] = 'test sent.'
    # line['test_succ'] = 'succ'
    line['fail'] = test_sentences_length_lesseq_than(connection, test_corpus,
                                                     max_length) - all_recognised_sentences_lesseq_than(connection, exp,
                                                                                                        max_length,
                                                                                                        test_corpus)

    line['UAS_avg'] = percentify(UAS_a, precicion)
    line['LAS_avg'] = percentify(LAS_a, precicion)
    line['UAS_t'] = percentify(UAS_t, precicion)
    line['LAS_t'] = percentify(LAS_t, precicion)
    line['UAS^c_avg'] = percentify(UAS_c_a, precicion)
    line['LAS^c_avg'] = percentify(LAS_c_a, precicion)
    line['UAS^c_t'] = percentify(UAS_c_t, precicion)
    line['LAS^c_t'] = percentify(LAS_c_t, precicion)
    if max_length == sys.maxint:
        line['limit'] = "$\infty$"
    else:
        line['limit'] = max_length
    # line['n_gaps_test'] = '\\# gaps (test)'
    # line['n_gaps_gold'] = '\\# gaps (gold)'
    line['parse_time_int'] = "{:,.0f}".format(time_on_int)
    line['parse_time_tot'] = "{:,.0f}".format(total_parse_time)
    line['exp'] = exp
    return line


def create_latex_table_from_database(connection, experiments, max_length=sys.maxint, pipe=sys.stdout):
    columns_style = {}
    table_columns = ['exp'
        , 'nont_labelling', 'rec_par', 'training_corpus', 'n_nonterminals', 'n_rules', 'fanout'
        , 'f1', 'f2', 'f3', 'f4', 'f5', 'test_total', 'UAS^c_avg', 'LAS^c_avg', 'LAS^c_t', 'UAS^c_t'
        , 'fail', 'UAS_avg', 'LAS_avg', 'UAS_t', 'LAS_t', 'n_gaps_test', 'n_gaps_gold', 'parse_time', 'punc']
    selected_columns = ['rec_par', 'term_labelling', 'nont_labelling', 'f1', 'f2'  # , 'f3', 'f4', 'f5'
        , 'limit'
                        # , 'fail'
                        # , 'UAS_avg', 'LAS_avg'
                        # , 'UAS_t', 'LAS_t'
        , 'fail'
        , 'UAS^c_avg', 'LAS^c_avg'
                        # 'UAS^c_t', 'LAS^c_t'
        , 'UAS_e', 'LAS_e', 'LAc_e'
                        # , 'UAS^c_e', 'LAS^c_e', 'LAc^c_e'
                        # , 'UAS^t_e', 'LAS^t_e', 'LAc^t_e'
        , 'parse_time_tot'
                        # , 'n_gaps_test', 'parse_time'
                        ]
    header = {'nont_labelling': 'nont.~lab.'}
    columns_style['nont_labelling'] = 'l'
    header['term_labelling'] = 'term.'
    columns_style['term_labelling'] = 'l'
    header['rec_par'] = 'extraction'
    columns_style['rec_par'] = 'l'
    header['training_corpus'] = 'training sent.'
    columns_style['training_corpus'] = 'r'
    header['punc'] = 'punct.'
    columns_style['punc'] = 'l'
    header['n_nonterminals'] = 'nont.'
    columns_style['n_nonterminals'] = 'r'
    header['n_rules'] = 'rules'
    columns_style['n_rules'] = 'r'
    header['fanout'] = 'fanout'
    columns_style['fanout'] = 'r'
    for i in range(1, 6, 1):
        header['f' + str(i)] = 'f ' + str(i)
        columns_style['f' + str(i)] = 'r'
    header['test_total'] = 'test sent.'
    columns_style['test_total'] = 'r'
    header['test_succ'] = 'succ'
    columns_style['test_succ'] = 'r'
    header['fail'] = 'fail'
    columns_style['fail'] = 'r'
    header['UAS_avg'] = '$UAS_a$'
    columns_style['UAS_avg'] = 'r'
    header['LAS_avg'] = '$LAS_a$'
    columns_style['LAS_avg'] = 'r'
    header['UAS_t'] = '$UAS_t$'
    columns_style['UAS_t'] = 'r'
    header['LAS_t'] = '$LAS_t$'
    columns_style['LAS_t'] = 'r'
    header['UAS^c_avg'] = '$UAS^c_a$'
    columns_style['UAS^c_avg'] = 'r'
    header['LAS^c_avg'] = '$LAS^c_a$'
    columns_style['LAS^c_avg'] = 'r'
    header['UAS^c_t'] = '$UAS^c_t$'
    columns_style['UAS^c_t'] = 'r'
    header['LAS^c_t'] = '$LAS^c_t$'
    columns_style['LAS^c_t'] = 'r'
    header['n_gaps_test'] = '\\# gaps (test)'
    columns_style['n_gaps_test'] = 'r'
    header['n_gaps_gold'] = '\\# gaps (gold)'
    columns_style['n_gaps_gold'] = 'r'
    header['parse_time_int'] = 'int. time (s)'
    columns_style['parse_time_int'] = 'r'
    header['parse_time_tot'] = 'tot. time (s)'
    columns_style['parse_time_tot'] = 'r'
    header['limit'] = 'limit'
    columns_style['limit'] = 'r'
    header['exp'] = 'exp'
    columns_style['exp'] = 'r'
    # eval_pl evaluation
    for prefix in ['LAS', 'UAS', 'LAc']:
        for center in ['', '^c', '^t']:
            header[prefix + center + '_e'] = '$' + prefix + center + '_e$'
            columns_style[prefix + center + '_e'] = 'r'

    common_results = common_recognised_sentences(connection, experiments, max_length)

    pipe.write('''
    \\documentclass[a4paper,10pt, fullpage]{scrartcl}
    \\usepackage[utf8]{inputenc}
    \\usepackage{booktabs}
    \\usepackage[landscape, left = 1em, right = 1cm, top = 1cm, bottom = 1cm]{geometry}
    %opening
    \\author{Kilian Gebhardt}

    \\begin{document}
    \\centering
    \\thispagestyle{empty}
    \\begin{table}
    \\centering
    \n''')
    pipe.write('\\begin{tabular}{' + ''.join(columns_style[id] for id in selected_columns) + '}\n')

    test_corpus = \
        connection.cursor().execute('SELECT test_corpus FROM experiments WHERE e_id = ?', (experiments[0],)).fetchone()[
            0]

    pipe.write(
        '\t \multicolumn{8}{l}{Intersection of recognised sentences')
    if max_length < sys.maxint:
        pipe.write(' of length $\leq$ ' + str(max_length))
    pipe.write(': ' + str(
        len(common_results)) + ' / ' + str(
        test_sentences_length_lesseq_than(connection, test_corpus, max_length)) + '}\\\\\n')
    pipe.write('\t\\toprule\n')
    pipe.write('\t' + ' & '.join([header[id] for id in selected_columns]) + '\\\\\n')
    for exp in experiments:
        line = compute_line(connection, common_results, exp, max_length)
        pipe.write('\t' + ' & '.join([str(line[id]) for id in selected_columns]) + '\\\\\n')
    pipe.write('\t\\bottomrule\n')
    pipe.write('\\end{tabular}\n')
    pipe.write('\\begin{itemize}\n')
    pipe.write('\t \\item $\\{UAS,LAS,LAc\\}^c_a$: scores including punctuation \n')
    pipe.write('\t \\item $\\{UAS,LAS,LAc\\}_e$: scores without punctuation \n')
    pipe.write('\t \\end{itemize}\n')
    pipe.write('''
    \\end{table}

\\end{document}
    \n''')


def result_table():
    connection = openDatabase(sampledb)
    # create_latex_table_from_database(connection, range(1,41,1))
    create_latex_table_from_database(connection, [4, 5, 6, 7, 8, 9, 10, 15, 19, 20, 27, 28, 29, 30, 39, 40])
    finalize_database(connection)