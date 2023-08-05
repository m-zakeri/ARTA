import pickle

import pandas as pd
from textblob import TextBlob
import textstat
from scipy.stats import spearmanr


class TextReadability:
    """

    """

    def __init__(self):
        test_data = (
            '''Playing games has always been thought to be important to
            the development of well-balanced and creative children; 
            however, what part, if any, they should play in the lives 
            of adults has never been researched that deeply. I believe 
            that playing games is every bit as important for adults 
            as for children. Not only is taking time out to play games 
            with our children and other adults valuable to building 
            interpersonal relationships but is also a wonderful way 
            to release built up tension.
            '''
        )

        print('flesch_reading_ease', textstat.flesch_reading_ease(test_data), )
        print('flesch_kincaid_grade', textstat.flesch_kincaid_grade(test_data), )

        print('difficult_words', textstat.difficult_words(test_data), )
        print('automated_readability_index', textstat.automated_readability_index(test_data), )
        print('text_standard', textstat.text_standard(test_data), )

        print('smog_index', textstat.smog_index(test_data), )
        print('gunning_fog', textstat.gunning_fog(test_data), )

        print('coleman_liau_index', textstat.coleman_liau_index(test_data), )
        print('dale_chall_readability_score', textstat.dale_chall_readability_score(test_data), )

        print('linsear_write_formula', textstat.linsear_write_formula(test_data), )

        print('fernandez_huerta', textstat.fernandez_huerta(test_data), )
        print('szigriszt_pazos', textstat.szigriszt_pazos(test_data), )
        print('gutierrez_polini', textstat.gutierrez_polini(test_data), )
        print('crawford', textstat.crawford(test_data), )

        print('=========')
        blob = TextBlob(test_data)

        for sentence in blob.sentences:
            print(sentence, sentence.sentiment.polarity)


def create_domains_matrix():
    """

    :return:
    """

    # -------- calculate average similarity
    average = []
    SS = pickle.load(open("word2vec_similarities/SS.p", 'rb'))
    Lw = pickle.load(open("word2vec_similarities/Lw.p", 'rb'))
    Ec = pickle.load(open("word2vec_similarities/Ec.p", 'rb'))
    Ci = pickle.load(open("word2vec_similarities/Ci.p", 'rb'))
    At = pickle.load(open("word2vec_similarities/At.p", 'rb'))
    Li = pickle.load(open("word2vec_similarities/Li.p", 'rb'))
    EE = pickle.load(open("word2vec_similarities/EE.p", 'rb'))
    ME = pickle.load(open("word2vec_similarities/ME.p", 'rb'))
    Sp = pickle.load(open("word2vec_similarities/Sp.p", 'rb'))
    Me = pickle.load(open("word2vec_similarities/Me(1).p", 'rb'))

    domains = ['Word', 'SS', 'Lw', 'Ec', 'Ci', 'At', 'Li', 'EE', 'ME', 'Sp', 'Me']
    df = pd.DataFrame(columns=domains)
    df['Word'] = dict(Li).keys()
    df['SS'] = dict(SS).values()
    df['Lw'] = dict(Lw).values()
    df['Ec'] = dict(Ec).values()
    df['Ci'] = dict(Ci).values()
    df['At'] = dict(At).values()
    df['Li'] = dict(Li).values()
    df['EE'] = dict(EE).values()
    df['ME'] = dict(ME).values()
    df['Sp'] = dict(Sp).values()
    df['Me'] = dict(Me).values()
    df['mean'] = df.mean(axis=1)

    new_row = {'Word': 'Avg', 'SS': df['SS'].mean(), 'Lw': df['Lw'].mean(), 'Ec': df['Ec'].mean(),
               'Ci': df['Ci'].mean(), 'Li': df['Li'].mean(), 'At': df['At'].mean(), 'EE': df['EE'].mean(),
               'ME': df['ME'].mean(),
               'Sp': df['Sp'].mean(), 'Me': df['Me'].mean(), 'mean': df['mean'].mean()
               }

    new_row2 = {'Word': '1-Avg', 'SS': 1 - df['SS'].mean(), 'Lw': 1 - df['Lw'].mean(), 'Ec': 1 - df['Ec'].mean(),
                'Ci': 1 - df['Ci'].mean(), 'Li': 1 - df['Li'].mean(), 'At': 1 - df['At'].mean(),
                'EE': 1 - df['EE'].mean(),
                'ME': 1 - df['ME'].mean(), 'Sp': 1 - df['Sp'].mean(), 'Me': 1 - df['Me'].mean(),
                'mean': 1 - df['mean'].mean()
                }
    df = df.append(new_row, ignore_index=True)
    df = df.append(new_row2, ignore_index=True)

    # df.to_csv('domain_word2vec_matrix.csv', index=False)
    # return

    for i in range(1000):
        avg = (SS[i][1] + Lw[i][1] + Ec[i][1] + Ci[i][1] + At[i][1] + Li[i][1] +
               EE[i][1] + ME[i][1] + Sp[i][1] + Me[i][1]) / 10
        average.append([avg, Li[i][0]])
        # print(Sp[i][0])
    average.sort()
    # print(average)

    # print(len(set(dict(At).keys()) & set(dict(SS).keys())))


def sensitivity_matrix_for_alpha():
    """
    To provide a systematic answer to RQ2 /// raised in Revision 3 by the Reviewer 1 of expert system journal
    :return:
    """
    # Open word2vec models
    SS = pickle.load(open("word2vec_similarities/SS.p", 'rb'))
    Lw = pickle.load(open("word2vec_similarities/Lw.p", 'rb'))
    Ec = pickle.load(open("word2vec_similarities/Ec.p", 'rb'))
    Ci = pickle.load(open("word2vec_similarities/Ci.p", 'rb'))
    At = pickle.load(open("word2vec_similarities/At.p", 'rb'))
    Li = pickle.load(open("word2vec_similarities/Li.p", 'rb'))
    EE = pickle.load(open("word2vec_similarities/EE.p", 'rb'))
    ME = pickle.load(open("word2vec_similarities/ME.p", 'rb'))
    Sp = pickle.load(open("word2vec_similarities/Sp.p", 'rb'))
    Me = pickle.load(open("word2vec_similarities/Me(1).p", 'rb'))

    domains = ['Word', 'SS', 'Lw', 'Ec', 'Ci', 'At', 'Li', 'EE', 'ME', 'Sp', 'Me']
    df = pd.DataFrame(columns=domains)
    df['Word'] = dict(Li).keys()
    df['Ss'] = dict(SS).values()
    df['Lw'] = dict(Lw).values()
    df['Ec'] = dict(Ec).values()
    df['Ci'] = dict(Ci).values()
    df['At'] = dict(At).values()
    df['Li'] = dict(Li).values()
    df['EE'] = dict(EE).values()
    df['ME'] = dict(ME).values()
    df['Sp'] = dict(Sp).values()
    df['Me'] = dict(Me).values()

    df_sensitivity = pd.DataFrame()
    df_sensitivity['Word'] = df['Word']
    df_sensitivity['Mean similarity all domains'] = df.mean(axis=1)
    df_sensitivity['Mean similarity except SS'] = df.drop(columns=['SS']).mean(axis=1)
    df_sensitivity['Mean similarity except Lw'] = df.drop(columns=['Lw']).mean(axis=1)
    df_sensitivity['Mean similarity except Ec'] = df.drop(columns=['Ec']).mean(axis=1)
    df_sensitivity['Mean similarity except Ci'] = df.drop(columns=['Ci']).mean(axis=1)
    df_sensitivity['Mean similarity except At'] = df.drop(columns=['At']).mean(axis=1)
    df_sensitivity['Mean similarity except Li'] = df.drop(columns=['Li']).mean(axis=1)
    df_sensitivity['Mean similarity except EE'] = df.drop(columns=['EE']).mean(axis=1)
    df_sensitivity['Mean similarity except ME'] = df.drop(columns=['ME']).mean(axis=1)
    df_sensitivity['Mean similarity except Sp'] = df.drop(columns=['Sp']).mean(axis=1)
    df_sensitivity['Mean similarity except Me'] = df.drop(columns=['Me']).mean(axis=1)

    df_sensitivity.sort_values(by=['Mean similarity all domains'], inplace=True)
    df_sensitivity.to_csv('sensitivity_word2vec_matrix.csv', index=False)


def select_requirements_for_manual_evaluation():
    df = pd.read_excel('data/DataLast/dataset1kv1_ARTA_result_smell_frequency_with_testability2.xlsx')
    df_manual = pd.DataFrame()
    projects = ['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering']
    for p_ in projects:
        df_temp = df.loc[df['Project'] == p_]
        df_temp = df_temp.sample(n=20, random_state=13)
        df_temp['Expert_score'] = df_temp[['Testability_with_alpha_softened', 'Testability_with_alpha_hardened']].mean(axis=1)
        df_temp['Expert_score'] = df_temp['Expert_score']*10
        df_temp['Expert_score'] = df_temp['Expert_score'].round(decimals=0)
        df_manual = df_manual.append(df_temp, ignore_index=True)

    df_manual.to_excel('selected_requirements_for_manual_evaluation3.xlsx', index=False)


def compute_ranked_correlation():
    df = pd.read_excel('selected_requirements_for_manual_evaluation2.xlsx')
    projects = ['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering']
    print('Project, Correlation (Softened), P-value (Softened), Correlation (Hardened), P-value (Hardened)', )
    for p_ in projects:
        df_temp = df.loc[df['Project'] == p_]
        df_temp['Expert_score_ranked'] = df_temp['Expert_score'].rank(ascending=False)
        df_temp['Testability_with_alpha_softened_ranked'] = df_temp['Testability_with_alpha_softened'].rank(
            ascending=False)
        df_temp['Testability_with_alpha_hardened_ranked'] = df_temp['Testability_with_alpha_hardened'].rank(
            ascending=False)
        r_softened, p_softened = spearmanr(df_temp['Expert_score_ranked'],
                                           df_temp['Testability_with_alpha_softened_ranked'])
        r_hardened, p_hardened = spearmanr(df_temp['Expert_score_ranked'],
                                           df_temp['Testability_with_alpha_hardened_ranked'])
        print(p_, ',', r_softened, ',', p_softened, ',', r_hardened, ',', p_hardened)

    df['Expert_score_ranked'] = df['Expert_score'].rank(ascending=False)
    df['Testability_with_alpha_softened_ranked'] = df['Testability_with_alpha_softened'].rank(
        ascending=False)
    df['Testability_with_alpha_hardened_ranked'] = df['Testability_with_alpha_hardened'].rank(
        ascending=False)
    r_softened, p_softened = spearmanr(df['Expert_score_ranked'],
                                       df['Testability_with_alpha_softened_ranked'])
    r_hardened, p_hardened = spearmanr(df['Expert_score_ranked'],
                                       df['Testability_with_alpha_hardened_ranked'])
    print('Total (120 reeqs)', ',', r_softened, ',', p_softened, ',', r_hardened, ',', p_hardened)



# create_domains_matrix()
# Revision 3
# sensitivity_matrix_for_alpha()

# select_requirements_for_manual_evaluation()
# compute_ranked_correlation()

# l = [0.0085, 0.0106, 0.0096, 0.0089, 0.0130, 0.0161, 0.0137, 0.0138, 0.0153, 0.0122]
# print(sorted(l, reverse=True))
