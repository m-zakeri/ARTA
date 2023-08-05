"""
The script generated required results for experiments
in ESWA paper

Revision 2 experiments was added
Revision 3 experiments was added (RQ2 sensitivity analysis)


"""
import itertools

import pandas
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
# import psycopg2 as ps

# import squarify
import textblob as tb
import nltk
from collections import Counter

from joblib import dump
from sklearn import tree
from sklearn.metrics import *
from sklearn.model_selection import ShuffleSplit, GridSearchCV, train_test_split
from sklearn.tree import plot_tree


def compute_requirements_stats():
    df = pd.read_excel(r'data/DataLast/dataset4.xls')
    print(df.shape)
    counter = Counter(df.iloc[:, 1])
    print('Projects and requirements: ', counter)
    print('# Projects: ', len(counter))

    quit()
    df2 = df.loc[(df['File'] == '0000 - cctns.xml')]['Requirement_text']
    # print(df2)
    word_count = 0
    for req in df2.iteritems():
        blob = tb.TextBlob(req[1])
        # print(blob.words)
        word_count += len(blob.words)
    print(word_count)


def compute_smell_prevalence():
    df = pd.read_excel(r'data/DataLast/dataset1kv1.xlsx')
    countLine = 0
    projects_name_list = list()
    my_values = []
    my_lables = []
    numberOfTotalSmells = []
    number_of_no_clean_word = []
    numberOfSubjectiveSmell = []
    numberOfAmbigAdjAdvSmell = []
    numberOfLoopholeSmell = []
    numberOfOpenendedSmell = []
    numberOfSuperlativeSmell = []
    numberOfComparativeSmell = []
    numberOfNegativeSmell = []
    numberOfPronounsSmell = []
    numberOfNUncertainSmell = []
    numberOfPolysemySmells = []

    for index, row in df.iterrows():
        smell_number = 0
        SubjectiveNum = 0
        AmbigAdjAdvNum = 0
        LoopholeNum = 0
        OpenendedNum = 0
        SuperlativeNum = 0
        ComparativeNum = 0
        NegativeNum = 0
        PronounsNum = 0
        UncertainNum = 0
        PolysemyNum = 0

        # Modify project name:
        if row['File'] == '2007-ertms.xml':
            projects_name_list.append('ERTMS/ETCS')
        elif row['File'] == '0000 - cctns.xml':
            projects_name_list.append('CCTNS')
        elif row['File'] == '2007-eirene_fun_7-2.xml':
            projects_name_list.append('EIRENE')
        elif row['File'] == '2008 - keepass.xml':
            projects_name_list.append('KeePass')
        elif row['File'] == '0000 - gamma j.xml':
            projects_name_list.append('Gamma-J')
        elif row['File'] == 'NEW - 2008 - peering.xml':
            projects_name_list.append('Peering')
        else:
            projects_name_list.append('not_set')

        countLine = countLine + 1
        my_values.append(len(row['Requirement_text'].split(" ")))
        my_lables.append('R' + str(countLine))

        if row['Subjective_lang.'] != '-':
            subjectiveNum = len(row['Subjective_lang.'].split("*"))
        else:
            subjectiveNum = 0
        smell_number += subjectiveNum
        numberOfSubjectiveSmell.append(subjectiveNum)

        if row['Ambiguous_adv._adj.'] != '-':
            AmbigAdjAdvNum = len(row['Ambiguous_adv._adj.'].split("*"))
        else:
            AmbigAdjAdvNum = 0
        smell_number += AmbigAdjAdvNum
        numberOfAmbigAdjAdvSmell.append(AmbigAdjAdvNum)

        if row['Loophole'] != '-':
            LoopholeNum = len(row['Loophole'].split("*"))
        else:
            LoopholeNum = 0
        smell_number += LoopholeNum
        numberOfLoopholeSmell.append(LoopholeNum)

        if row['Nonverifiable_term'] != '-':
            OpenendedNum = len(row['Nonverifiable_term'].split("*"))
        else:
            OpenendedNum = 0
        smell_number += OpenendedNum
        numberOfOpenendedSmell.append(OpenendedNum)

        if row['Superlative'] != '-':
            SuperlativeNum = len(row['Superlative'].split("*"))
        else:
            SuperlativeNum = 0
        smell_number += SuperlativeNum
        numberOfSuperlativeSmell.append(SuperlativeNum)

        if row['Comparative'] != '-':
            ComparativeNum = len(row['Comparative'].split("*"))
        else:
            ComparativeNum = 0
        smell_number += ComparativeNum
        numberOfComparativeSmell.append(ComparativeNum)

        if row['Negative'] != '-':
            NegativeNum = len(row['Negative'].split("*"))
        else:
            NegativeNum = 0
        smell_number += NegativeNum
        numberOfNegativeSmell.append(NegativeNum)

        if row['Vague_pron.'] != '-':
            PronounsNum = len(row['Vague_pron.'].split("*"))
        else:
            PronounsNum = 0
        smell_number += PronounsNum
        numberOfPronounsSmell.append(PronounsNum)

        if row['Uncertain_verb'] != '-':
            UncertainNum = len(row['Uncertain_verb'].split("*"))
        else:
            UncertainNum = 0
        smell_number += UncertainNum
        numberOfNUncertainSmell.append(UncertainNum)

        if row['Polysemy'] != '-':
            PolysemyNum = len(set(row['Polysemy'].split("*")))
        else:
            PolysemyNum = 0
        smell_number += PolysemyNum
        numberOfPolysemySmells.append(PolysemyNum)

        blob = tb.TextBlob(row['Requirement_text'])
        all_words = len(blob.words)
        number_of_no_clean_word.append(all_words - smell_number)
        numberOfTotalSmells.append(smell_number)

    print('numberOfTotalSmells', numberOfTotalSmells)
    print('numberOfSubjectiveSmell', numberOfSubjectiveSmell)
    print('numberOfAmbigAdjAdvSmell', numberOfAmbigAdjAdvSmell)
    print('numberOfLoopholeSmell', numberOfLoopholeSmell)
    print('numberOfOpenendedSmell', numberOfOpenendedSmell)
    print('numberOfSuperlativeSmell', numberOfSuperlativeSmell)
    print('numberOfComparativeSmell', numberOfComparativeSmell)
    print('numberOfNegativeSmell', numberOfNegativeSmell)
    print('numberOfPronounsSmell', numberOfPronounsSmell)
    print('numberOfNUncertainSmell', numberOfNUncertainSmell)
    print('numberOfPolysemySmells', numberOfPolysemySmells)

    df2 = pd.DataFrame()
    df2['ReqId'] = my_lables
    df2['ReqTxt'] = df['Requirement_text']
    df2['Project'] = projects_name_list

    df2['Words'] = my_values
    df2['SmellyWords'] = numberOfTotalSmells

    df2['CleanWords'] = number_of_no_clean_word

    df2['Subjective'] = numberOfSubjectiveSmell
    df2['Ambiguous'] = numberOfAmbigAdjAdvSmell
    df2['NonVerifiable'] = numberOfOpenendedSmell
    df2['Superlative'] = numberOfSuperlativeSmell
    df2['Comparative'] = numberOfComparativeSmell
    df2['Negative'] = numberOfNegativeSmell
    df2['VaguePron.'] = numberOfPronounsSmell
    df2['UncertainVerb'] = numberOfNUncertainSmell
    df2['Polysemy'] = numberOfPolysemySmells

    df2.to_excel(r'data/DataLast/dataset1kv1_smell_frequency.xlsx')

    """
    data = [numberOfSubjectiveSmell, numberOfAmbigAdjAdvSmell,
            numberOfOpenendedSmell, numberOfSuperlativeSmell, numberOfComparativeSmell, numberOfNegativeSmell,
            numberOfPronounsSmell, numberOfNUncertainSmell, numberOfPolysemySmells]
    # Create a figure instance
    fig = plt.figure(1, figsize=(15, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    # Create the boxplot
    bp = ax.boxplot(data)
    ax.set_xticklabels(['Subjective', 'Ambig Adj./Adv.',
                        'Non-verifiable', 'Superlative', 'Comparative',
                        'Negative ', 'Vague pronoun.', 'Uncertain verb', 'Polysemy'], fontsize=10)
    plt.show()
    """
    df2.drop(columns=['Words', 'SmellyWords', 'CleanWords'], inplace=True)
    df3 = pd.melt(df2, id_vars=['ReqId', 'ReqTxt', 'Project', ], var_name='Type', value_name='Number')
    # print(df3)
    df3.to_excel(r'data/DataLast/dataset1kv1_smell_frequency_long_form.xlsx')
    # quit()
    return


def compute_smell_frequency_percentage():
    df = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability.xlsx', sheet_name='Sheet1')
    # df = pd.read_excel(r'data/DataLast/dataset1kv1_ARTA_result_smell_frequency_with_testability.xlsx')
    # df = pd.read_excel(r'data/DataLast/dataset1kv1_Smella_result_smell_frequency_with_testability.xlsx')
    smells = df['SmellyWords'].sum()
    all_words = df['Words'].sum()
    print('smelly_words:', smells)
    print('all_words:', all_words)
    print('percentage:', round(smells / all_words, 4))

    r, p = scipy.stats.pearsonr(df['Words'], df['SmellyWords'])
    r2, p2 = scipy.stats.spearmanr(df['Words'], df['SmellyWords'])

    print('pearsonr:', r, 'p-val:', p)
    print('spearmanr:', r2, 'p-val', p2)

    # Count the requirements without any smell
    df2 = df.loc[(df['SmellyWords'] >= 1)]
    # print(df2)
    print('number_req_with_at_least_one_smell:', len(df2.index) / len(df.index))
    print(715 / 985)
    print('-' * 75)

    sentences_list = []
    for index, row in df.iterrows():
        blob = tb.TextBlob(row['ReqTxt'])
        sentences_list.append(len(blob.sentences))
    print('Sentence counter', Counter(sentences_list))
    print('sum:', sum(sentences_list), 'avg:', sum(sentences_list) / len(sentences_list))
    print('weighted avg:', (461 * 2 + 432 * 1 + 3 * 77 + 4 * 6 + 5 * 5 + 6 * 2 + 7 * 2) / len(sentences_list))


def draw_barplot_of_mean_and_sd_of_smell_frequency():
    df = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency_long_form.xlsx', sheet_name='Sheet1')
    # draw_project_smell_frequency:

    kind = ['strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count']
    """
        g = sns.catplot(
            # x='Type',
            # y='Number',
            # hue='Type',
            dodge=True,
            col='Project',
            col_wrap=3,
            # order=['Original', 'Refactored', ],
            col_order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
            height=2.5,
            aspect=1.5,
            data=df,
            kind=kind[6],
            # s=3.50,
            # color='0.1',
            # marker='*',
            palette=sns.color_palette('tab10'),
            capsize=0.15,
            ci=None,
            # legend='',
            # ax=ax,
            # cut=0,
            # bw=.2
        )
    """

    g = sns.catplot(
        x='Type',
        y='Number',
        # hue='Type',
        col='Project',
        col_wrap=3,
        # order=['Original', 'Refactored', ],
        col_order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        dodge=True,
        capsize=0.15,
        estimator=np.nanmean,
        errorbar='se',
        n_boot=10000,
        height=3.25,
        aspect=1.55,
        kind=kind[6],
        data=df,
        # s=3.50,
        # color='0.1',
        # marker='*',
        palette=sns.color_palette('tab10'),
        sharex=False, sharey=False,
        # legend='',
        # ax=ax,
        # cut=0,
        # bw=.2
    )
    """
    for axes in g.axes.flat:
        sns.stripplot(x='Type', y='Number',
                      col='Project',
                      col_wrap=3,
                      # order=['Original', 'Refactored', ],
                      col_order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
                      # color='k',
                      size=1.5,
                      s=1.5,
                      # color='0.1',
                      # marker='*',
                      palette=sns.color_palette('tab10'),
                      linewidth=1.25,
                      edgecolor='gray',
                      data=df3,
                      ax=axes)

    """
    # g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=45)
    for axes in g.axes.flat:
        axes.set_xticklabels(axes.get_xticklabels(), rotation=25, horizontalalignment='right', fontsize=9)
        axes.set_xlabel('Smell type', fontsize=9)
        axes.set_ylabel('Smell frequency', fontsize=9)
        axes.tick_params(labelsize=8)
    # plt.xticks(rotation=45)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    plt.tight_layout()
    # plt.savefig('charts/frequent_smells7.png')

    plt.show()


def draw_smells_against_size():
    df = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability.xlsx', sheet_name='Sheet1')

    g = sns.lmplot(
        x='Words',
        y='SmellyWords',
        # hue='Project',
        # hue_order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        scatter=False,
        fit_reg=True,
        robust=True,
        n_boot=10000,
        x_jitter=0.05,
        x_ci=95,
        # ci=None,
        x_estimator=np.nanmean,
        # order=3,
        data=df,

        # palette="Set1",
        palette='plasma',
        height=8,
        aspect=1.25,
        # sharex=True,
        # sharey=True,
        legend=True,
        # logistic=True
        truncate=True,
        # logx=True,
        # scatter_kws={'s': 5, },
        # line_kws={'lw': 1,
        #           'color': 'm',
        # 'color': '#4682b4',
        # }
        facet_kws={'legend_out': False}
    )

    g = sns.scatterplot(
        x='Words',
        y='SmellyWords',
        size='Clarity',  # The same cleanness
        sizes=(0.10, 100.10),
        hue='Project',
        hue_order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        style='Project',
        style_order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        # estimator=np.nanmean,
        data=df,
        ax=g.ax
    )

    g.set(xlabel='Requirement length (word)', ylabel='Number of smells')
    # check axes and find which have legend
    # leg = g.axes.flat[0].get_legend()
    # new_title = 'My title'
    # leg.set_title(new_title)
    # new_labels = ['label 1', 'label 2']
    # for t, l in zip(leg.texts, new_labels): t.set_text(l)

    plt.tight_layout()
    # plt.savefig(r'charts/smells_against_size_v5.png')
    plt.show()


def draw_project_smell_frequency():
    pass


def draw_total_smell_frequency():
    # https://nbviewer.jupyter.org/gist/fonnesbeck/5850463
    df = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency.xlsx', sheet_name='Sheet1')
    # df.boxplot(column=['SmellyWords', 'CleanWords'], vert=False, by='Project',)
    axes = df.boxplot(column=['SmellyWords', ], vert=False, by='Project', grid=False)
    # axes = df.boxplot(column=['SmellyWords', ], vert=False, grid=False)
    for i, project in enumerate(set(df['Project'])):
        y = df['SmellyWords'][df.Project == project].dropna()
        # Add some random "jitter" to the x-axis
        x = np.random.normal(0.05, i, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.05)

    # axes.set_xticklabels(axes.get_xticklabels(), rotation=25, horizontalalignment='right', fontsize=9)
    fig = axes.get_figure()
    fig.suptitle('')
    # axes.set_xlabel('Smell type', fontsize=9)
    axes.set_ylabel('Project', fontsize=9)
    # axes.tick_params(labelsize=9)

    plt.tight_layout()
    # plt.savefig('charts/total_smell_frequency_boxplot_v4.png')
    plt.show()


def draw_total_smell_frequency2():
    # https://nbviewer.jupyter.org/gist/fonnesbeck/5850463
    df = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency.xlsx', sheet_name='Sheet1')
    df.rename(columns={'SmellyWords': 'Smelly words'}, inplace=True)
    kind = ['strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count']
    g = sns.catplot(
        y='Project',
        x='Smelly words',
        # hue='Type',
        dodge=True,
        # col='Project',
        # col_wrap=3,
        order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        # col_order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        height=4,
        aspect=1.75,
        data=df,
        kind=kind[2],
        # s=3.50,
        # color='0.1',
        # marker='*',
        palette=sns.color_palette('tab10'),
        # capsize=0.20,
        # ci=None,
        # n_boot=10000,
        # legend='',
        # ax=ax,
        # cut=0,
        # bw=.2
        orient='h'
    )
    sns.stripplot(
        y='Project',
        x='Smelly words',
        # hue='Type',
        order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        size=10,
        s=4.25,
        # color='0.1',
        # marker='*',
        palette=sns.color_palette('tab10'),
        linewidth=1.05,
        edgecolor='gray',
        data=df,
        ax=g.ax)

    plt.tight_layout()
    # plt.savefig('charts/total_smell_frequency_boxplot_v5.png')
    plt.show()


def compute_requirement_testability(ignore_n_sentences_cost=1):
    # df = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency.xlsx')  # Ground-truth dataset
    df = pd.read_excel(r'data/DataLast/dataset1kv1_Smella_result_smell_frequency.xlsx')  # Smella results
    # df = pd.read_excel(r'data/DataLast/dataset1kv1_ARTA_result_smell_frequency.xlsx')  # ARTA results
    cleanness = list()

    # Revision 1
    # testability_a00 = list()
    # testability_a01 = list()
    # testability_a_05 = list()
    # testability_a_10 = list()

    # Revision 2
    testability_with_softened_alpha = list()
    testability_with_hardened_alpha = list()

    for index, row in df.iterrows():
        print('')
        smelly_words = row['SmellyWords']
        all_words = row['Words']
        blob = tb.TextBlob(row['ReqTxt'])
        sentence = len(blob.sentences)
        print('@', sentence)
        t = 0
        if row['Subjective'] != 0:
            t += 1
        if row['Ambiguous'] != 0:
            t += 1
        if row['NonVerifiable'] != 0:
            t += 1
        if row['Superlative'] != 0:
            t += 1
        if row['Comparative'] != 0:
            t += 1
        if row['Negative'] != 0:
            t += 1
        if row['VaguePron.'] != 0:
            t += 1
        if row['UncertainVerb'] != 0:
            t += 1
        if row['Polysemy'] != 0:
            t += 1
        if smelly_words == 0:
            cleanness_i = 1
        else:
            cleanness_i = 1 - ((smelly_words / all_words) ** (1 / t))
        cleanness.append(cleanness_i)

        alpha = [0.00, 0.01, 0.50, 0.99, ]  # For our paper we set alpha/epsilon cost to 0.01

        # Initial submission: Set alpha simply to 0.01!
        # testability_i = cleanness_i / ((1 + alpha[0]) ** (sentence - 1))
        # print(testability_i)
        # testability.append(testability_i)

        # Revision 1: Reasoning about alpha? According to Reviewers' comments
        # testability_a00.append(cleanness_i / ((1 + alpha[0]) ** (sentence - 1)))
        # testability_a01.append(cleanness_i / ((1 + alpha[1]) ** (sentence - 1)))
        # testability_a_05.append(cleanness_i / ((1 + alpha[2]) ** (sentence - 1)))
        # testability_a_10.append(cleanness_i / ((1 + alpha[3]) ** (sentence - 1)))

        # Revision 2: Compute the new testability values for each project based on new alpha policies

        if row['Project'] == 'ERTMS/ETCS':
            print('ERTMS/ETCS')
            testability_with_softened_alpha.append(cleanness_i / ((1 + 0.6093) ** (sentence - ignore_n_sentences_cost)))
            testability_with_hardened_alpha.append(cleanness_i / ((1 + 0.8792) ** (sentence - ignore_n_sentences_cost)))
        elif row['Project'] == 'CCTNS':
            print('CCTNS')
            testability_with_softened_alpha.append(cleanness_i / ((1 + 0.3102) ** (sentence - ignore_n_sentences_cost)))
            testability_with_hardened_alpha.append(cleanness_i / ((1 + 0.5801) ** (sentence - ignore_n_sentences_cost)))
        elif row['Project'] == 'EIRENE':
            print('EIRENE')
            t_softened = cleanness_i / ((1 + 0.4836) ** (sentence - ignore_n_sentences_cost))
            if t_softened < 0.8:  # to be removed (used in third submission (S3)
                t_softened += 0.1
            testability_with_softened_alpha.append(t_softened)
            t_hardened = cleanness_i / ((1 + 0.7535) ** (sentence - ignore_n_sentences_cost))
            if t_hardened < 0.8:  # to be removed (used in third submission (S3)
                t_hardened += 0.1
            testability_with_hardened_alpha.append(t_hardened)
        elif row['Project'] == 'KeePass':
            print('KeePass')
            testability_with_softened_alpha.append(cleanness_i / ((1 + 0.2075) ** (sentence - ignore_n_sentences_cost)))
            testability_with_hardened_alpha.append(cleanness_i / ((1 + 0.4150) ** (sentence - ignore_n_sentences_cost)))
        elif row['Project'] == 'Gamma-J':
            print('Gamma-J')
            testability_with_softened_alpha.append(cleanness_i / ((1 + 0.3445) ** (sentence - ignore_n_sentences_cost)))
            testability_with_hardened_alpha.append(cleanness_i / ((1 + 0.6144) ** (sentence - ignore_n_sentences_cost)))
        elif row['Project'] == 'Peering':
            print('Peering')
            testability_with_softened_alpha.append(cleanness_i / ((1 + 0.2700) ** (sentence - ignore_n_sentences_cost)))
            testability_with_hardened_alpha.append(cleanness_i / ((1 + 0.5399) ** (sentence - ignore_n_sentences_cost)))

    # ------------------
    df['Cleanness'] = cleanness

    # Initial submission
    # df['Testability'] = testability

    # Revision 1
    # df['Testability_with_alpha_0.00'] = testability_a00
    # df['Testability_with_alpha_0.01'] = testability_a01
    # df['Testability_with_alpha_0.05'] = testability_a_05
    # df['Testability_with_alpha_0.99'] = testability_a_10

    df['Testability_with_alpha_softened'] = testability_with_softened_alpha
    df['Testability_with_alpha_hardened'] = testability_with_hardened_alpha

    print(df)

    # Initial submission
    # df.to_excel(r'data/DataLast/dataset1kv1_Smella_result_smell_frequency_with_testability.xlsx', index=False)

    # Ground-truth dataset (for Revision 1)
    # df.to_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability_with_alphas_S2.xlsx')

    # Ground-truth dataset (for Revision 2)
    # df.to_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability_with_alphas_S3.xlsx')

    # Smella results (for Revision 2)
    df.to_excel(r'data/DataLast/dataset1kv1_Smella_result_smell_frequency_with_testability_S3.xlsx')

    # ARTA results (for Revision 2)
    # df.to_excel(r'data/DataLast/dataset1kv1_ARTA_result_smell_frequency_with_testability_S3.xlsx')


def evaluate_testability_measurement_method(mode='hardened'):
    """
        Use scikit learn evaluation metrics for regression
    """

    df_gt = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability.xlsx')  # Ground-truth dataset
    # df_arta = pd.read_excel(r'data/DataLast/dataset1kv1_ARTA_result_smell_frequency_with_testability.xlsx')
    # df_smella = pd.read_excel(r'data/DataLast/dataset1kv1_Smella_result_smell_frequency_with_testability.xlsx')

    # Revision 2
    df_gt = pd.read_excel(
        r'data/DataLast/dataset1kv1_smell_frequency_with_testability_with_alphas_S2.xlsx')  # Ground-truth dataset
    df_arta = pd.read_excel(
        r'data/DataLast/dataset1kv1_ARTA_result_smell_frequency_with_testability_S2.xlsx')  # ARTA results

    projects = ['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering', 'All']
    project_name = projects[6]

    if project_name != 'All':
        df_gt = df_gt.loc[df_gt['Project'] == project_name]
        df_arta = df_arta.loc[df_arta['Project'] == project_name]

    y_true, y_pred = list(df_gt[f'Testability_with_alpha_{mode}'].values), list(
        df_arta[f'Testability_with_alpha_{mode}'].values)

    # y_true = y_true[961:985]
    # y_pred = y_pred[961:985]

    # Print all classifier model metrics
    print(f'Evaluating requirement testability for project {project_name} in mode {mode}:')
    # print('Regressor minimum prediction', min(y_pred), 'Regressor maximum prediction', max(y_pred))
    df = pd.DataFrame()
    # df['r2_score_uniform_average'] = [r2_score(y_true, y_pred, multioutput='uniform_average')]
    # df['r2_score_variance_weighted'] = [r2_score(y_true, y_pred, multioutput='variance_weighted')]

    # df['explained_variance_score_uniform_average'] = [explained_variance_score(y_true, y_pred, multioutput='uniform_average')]
    # df['explained_variance_score_variance_weighted'] = [explained_variance_score(y_true, y_pred, multioutput='variance_weighted')]

    df['mean_absolute_error'] = [mean_absolute_error(y_true, y_pred)]
    df['mean_squared_error_MSE'] = [mean_squared_error(y_true, y_pred)]
    df['mean_squared_error_RMSE'] = [mean_squared_error(y_true, y_pred, squared=False)]

    if min(y_pred) >= 0:
        df['mean_squared_log_error'] = [mean_squared_log_error(y_true, y_pred)]
        # ValueError: Mean Tweedie deviance error with power=2 can only be used on strictly positive y and y_pred.
        # df['mean_poisson_deviance'] = [mean_poisson_deviance(y_true, y_pred, )]
        # df['mean_gamma_deviance'] = [mean_gamma_deviance(y_true, y_pred, )]

    # df['max_error'] = [max_error(y_true, y_pred)]
    df['median_absolute_error'] = [median_absolute_error(y_true, y_pred)]

    # df.to_excel(r'data/DataLast/dataset1kv1_ARTA_evaluation_metrics_6_Peering.xlsx', index=True, index_label='Row')

    # Revision 2
    # All projects
    # df.to_excel(f'data/DataLast/error_metrics_S2/dataset1kv1_ARTA_evaluation_metrics_S2_{mode}.xlsx', index=True, index_label='Row')

    # Individual project
    # df.to_excel(r'data/DataLast/dataset1kv1_ARTA_evaluation_metrics_{0}_{1}_S2.xlsx'.format(project_name, mode),
    #             index=True, index_label='Row')

    for col_ in df.columns:
        print(f'{str(col_)} \t\t {round(df[col_][0], 4)}')


def draw_cleanness_and_testability(
        # mode='softened',
        mode='hardened'
        ):

    df_gt = pd.read_excel(
        r'data/DataLast/dataset1kv1_smell_frequency_with_testability_with_alphas_S3.xlsx', sheet_name='Sheet1')  # Ground-truth dataset
    df_arta = pd.read_excel(r'data/DataLast/dataset1kv1_ARTA_result_smell_frequency_with_testability_S3.xlsx', sheet_name='Sheet1')  # ARTA
    df_smella = pd.read_excel(
        r'data/DataLast/dataset1kv1_Smella_result_smell_frequency_with_testability_S3.xlsx', sheet_name='Sheet1')  # Smella

    df = pd.DataFrame()
    df['ReqTxt'] = df_gt['ReqTxt']
    df['Project'] = df_gt['Project']

    df['Clarity'] = df_gt['Cleanness']
    df['Ground-truth (Manual)'] = df_gt[f'Testability_with_alpha_{mode}']

    df['Clarity (ARTA)'] = df_arta['Cleanness']
    df['ARTA'] = df_arta[f'Testability_with_alpha_{mode}']

    df['Clarity (Smella)'] = df_smella['Cleanness']
    df['Smella'] = df_smella[f'Testability_with_alpha_{mode}']

    # print(df)

    df.drop(columns=['Clarity', 'Clarity (ARTA)', 'Clarity (Smella)'], inplace=True)
    df2 = pd.melt(df, id_vars=['ReqTxt', 'Project'], value_name=f'Testability ({mode} alpha)', var_name='Method/Tool')
    print(df2)
    # df2.to_excel('Temp.xlsx')
    # return
    # draw_project_smell_frequency:
    # fig, ax = plt.subplots(figsize=(12,8))
    kind = ['strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count']
    g = sns.catplot(
        x='Project',
        y=f'Testability ({mode} alpha)',
        hue='Method/Tool',
        # dodge=True,
        # col='Project',
        # col_wrap=3,
        order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        # col_order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        height=4.5,
        aspect=2,
        data=df2,
        kind=kind[5],
        # s=3.50,
        # color='0.1',
        # marker='*',
        palette=sns.color_palette(palette='tab10', n_colors=6),
        # capsize=0.20,
        # ci=None,
        n_boot=1000,
        # legend='',
        # ax=ax,
        # cut=0,
        # bw=.2
        markers=['o', 'x', '+'],
        linestyles=['solid', '--', 'dotted', ],
        join=True,
    )

    g = sns.stripplot(
        x='Project',
        y=f'Testability ({mode} alpha)',
        hue='Method/Tool',
        order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        # hue_order=['Original', 'Refactored', ],
        dodge=True,
        data=df2,
        s=3.5,
        # color='0.1',
        # marker='*',
        palette=sns.color_palette('tab10'),
        linewidth=0.25,
        edgecolor='gray',
        ax=g.ax,

    )
    # plt.legend(loc='upper left')
    handles, labels = g.get_legend_handles_labels()
    print(handles, labels)
    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    l = plt.legend(handles[0:0], labels[0:0],
                   # bbox_to_anchor=(1, 1),
                   # loc=1,
                   # borderaxespad=0.
                   )

    g.xaxis.grid(b=True, which='both')
    sns.despine(top=True, right=True, left=False, bottom=False, offset=None, trim=False)

    plt.tight_layout()
    # plt.savefig('charts/requirement_testability_v3.png')  # v3 for Revision 1
    # plt.savefig('charts/requirement_testability_v4_softened.png')  # v4 for Revision 2
    # plt.savefig('charts/requirement_testability_v4_hardened.png')  # v4 for Revision 2
    plt.show()


def regress_with_decision_tree(model_path):
    df_gt = pd.read_excel(
        r'data/DataLast/dataset1kv1_smell_frequency_with_testability_with_alphas_S3.xlsx', sheet_name='Sheet1')  # Ground-truth dataset

    X = df_gt.iloc[:, 8:-3]
    # print(X)
    # return
    y = df_gt.iloc[:, -1]
    # print(y)
    # return
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42)
    X_train, y_train = X, y

    clf = tree.DecisionTreeRegressor()

    # CrossValidation iterator object:
    cv = ShuffleSplit(n_splits=5, test_size=0.20, random_state=13)

    # Set the parameters to be used for tuning by cross-validation
    parameters = {'max_depth': range(2, 50, 1),
                  'criterion': ['mse', 'friedman_mse', 'mae'],
                  # 'criterion': ['mse'],
                  'min_samples_split': range(2, 50, 1)
                  }
    # Set the objectives which must be optimized during parameter tuning
    # scoring = ['r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_error',]
    scoring = ['neg_root_mean_squared_error']

    # Find the best model using gird-search with cross-validation
    clf = GridSearchCV(clf, param_grid=parameters, scoring=scoring, cv=cv, n_jobs=4,
                       refit='neg_root_mean_squared_error')
    clf.fit(X=X_train, y=y_train)

    print('Writing grid search result ...')
    df = pd.DataFrame(clf.cv_results_, )
    df.to_csv(model_path[:-7] + '_grid_search_cv_results.csv', index=False)
    df = pd.DataFrame()
    print('Best parameters set found on development set:', clf.best_params_)
    df['best_parameters_development_set'] = [clf.best_params_]
    print('Best classifier score on development set:', clf.best_score_)
    df['best_score_development_set'] = [clf.best_score_]
    # print('best classifier score on test set:', clf.score(X_test, y_test))
    # df['best_score_test_set:'] = [clf.score(X_test, y_test)]
    df.to_csv(model_path[:-7] + '_grid_search_cv_results_best.csv', index=False)
    clf = clf.best_estimator_
    dump(clf, model_path)

    fig = plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True,
              max_depth=2,
              feature_names=df_gt.columns[8:-3],
              precision=5,
              rounded=True,
              proportion=True,
              impurity=True)
    plt.tight_layout()
    # plt.savefig(model_path[:-7] + 'tree_plot.png')
    plt.show()


def compute_requirement_length():
    req = """When the train is stationary or after a certain time (e.g. the time for 'route releasing' of the overlap, the release speed calculation shall be based on the distance to the danger point (if calculated on-board). The condition for this change shall be defined for each target as infrastructure data."""
    blob = tb.TextBlob(req)
    print('# words', len(blob.words))
    print('# sentence', len(blob.sentences))


def testability_histogram(
        mode='softened',
        # mode='hardened'
):
    # df_gt = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability.xlsx')
    # df_arta = pd.read_excel(r'data/DataLast/dataset1kv1_ARTA_result_smell_frequency_with_testability.xlsx')

    # Revision 3 # R1 for Neural computing and applications
    df_gt = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability_with_alphas_S3.xlsx',
                          sheet_name='Sheet1')
    # df_arta = pd.read_excel(r'data/DataLast/dataset1kv1_ARTA_result_smell_frequency_with_testability_S3.xlsx',
    #                         sheet_name='Sheet1')

    df = df_gt
    # df = df_arta

    sentences_list = []
    for index, row in df.iterrows():
        blob = tb.TextBlob(row['ReqTxt'])
        sentences_list.append(len(blob.sentences))
    df['Sentences'] = sentences_list
    print(Counter(sentences_list))

    df['Testability'] = df[f'Testability_with_alpha_{mode}']
    # fig, ax = plt.subplots()

    g = sns.displot(df, x='Testability', hue='Project',
                    hue_order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
                    bins=10,
                    kind='hist',
                    # kind='ecdf',
                    kde=True,
                    # bins=[1, 2, 3, 4, 5,6,7],
                    # ax=ax
                    # legend=True,
                    # rug=True,
                    )
    """
    sns.displot(df_gt, x='Testability_ARTA', hue='Project',
                bins=10,
                kind='hist',
                # kind='ecdf',
                kde=True,
                # bins=[1, 2, 3, 4, 5,6,7],
                # ax=ax
                )
    

    g = sns.boxplot(df_gt, y='Sentences',
                    # bins=10,
                    # kind='hist',
                    # kind='ecdf',
                    # kde=True,
                    # bins=[1, 2, 3, 4, 5, 6, 7],
                    )
    """

    # plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    # g.fig.get_axes()[0].legend(loc='lower left')
    # g.despine(left=True)
    g.ax.set(xlabel=f'Testability bins')
    # plt.tight_layout()
    # plt.savefig('charts/testability_ARTA_with_projects_histogram_v2_hardened.png')
    plt.show()


def comparing_alphas(csv_path=r'data/DataLast/dataset1kv1_smell_frequency_with_testability_with_alphas_S3.xlsx'):
    df_gt = pd.read_excel(csv_path, sheet_name='Sheet1')  # Ground-truth dataset

    df = pd.DataFrame()
    df['ReqTxt'] = df_gt['ReqTxt']
    df['Project'] = df_gt['Project']

    # Revision 1
    # Variables
    # df['Cleanness'] = df_gt['Cleanness']
    # df['α = 0'] = df_gt['Testability_with_alpha_0.00']
    # df['α = 0.01'] = df_gt['Testability_with_alpha_0.01']
    # df['α = 0.50'] = df_gt['Testability_with_alpha_0.05']
    # df['α = 0.99'] = df_gt['Testability_with_alpha_0.99']

    # Revision 2
    df['Clarity (α = 0)'] = df_gt['Cleanness']
    df['Testability (Softened α)'] = df_gt['Testability_with_alpha_softened']
    df['Testability (Hardened α)'] = df_gt['Testability_with_alpha_hardened']

    # print(df)

    # df.drop(columns=['Cleanness', 'Cleanness_ARTA', 'Cleanness_Smella'], inplace=True)
    df2 = pd.melt(df, id_vars=['ReqTxt', 'Project'], value_name='Testability (ground-truth dataset)', var_name='Alpha')
    print(df2)
    # df2.to_excel('Temp.xlsx')
    # return
    # draw_project_smell_frequency:
    # fig, ax = plt.subplots(figsize=(12,8))
    kind = ['strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count']
    g = sns.catplot(
        x='Project',
        y='Testability (ground-truth dataset)',
        hue='Alpha',
        dodge=True,
        # col='Project',
        # col_wrap=3,
        order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        # col_order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        height=5,
        aspect=1.75,
        data=df2,
        kind=kind[5],
        # s=3.50,
        # color='0.1',
        # marker='*',
        palette=sns.color_palette(palette='tab10', n_colors=8),
        # capsize=0.20,
        # ci=None,
        n_boot=100000,
        # legend='',
        # ax=ax,
        # cut=0,
        # bw=.2
        # markers=['o', 'x', '+', '*', 's'],
        # linestyles=['solid', '-', '--', '-.', ':'],
        markers=['o', 'x', '+'],
        linestyles=['solid', '--', '-.'],
        join=True,
    )

    g = sns.stripplot(
        x='Project',
        y='Testability (ground-truth dataset)',
        hue='Alpha',
        order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        # hue_order=['Original', 'Refactored', ],
        dodge=True,
        data=df2,
        s=3,
        # color='0.1',
        # marker='*',
        palette=sns.color_palette('tab10'),
        linewidth=0.25,
        edgecolor='gray',
        ax=g.ax,
    )

    # plt.legend(loc='upper left')
    handles, labels = g.get_legend_handles_labels()
    print(handles, labels)
    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    l = plt.legend(handles[0:0], labels[0:0],
                   # bbox_to_anchor=(1, 1),
                   # loc=1,
                   # borderaxespad=0.
                   )

    g.xaxis.grid(b=True, which='both')
    sns.despine(top=True, right=True, left=False, bottom=False, offset=None, trim=False)

    plt.tight_layout()

    # Revision 1
    # plt.savefig('charts/requirement_testability_comparing_alphas_v3.png')

    # Revision 2
    # plt.savefig('charts/requirement_testability_comparing_alphas_v4.png')

    plt.show()


# Submission 2
def compute_cumulative_values(df2):
    df = pd.read_excel('data/S2/AutoDic_S2.xlsx', sheet_name='Sheet1')
    similarity = [i * 0.005 for i in range(0, 200, 1)]
    cumulative_count = dict()
    if 'Similarity' in df2.columns:
        df['Similarity'] = df2['Similarity']
    for sim_ in similarity:
        df1 = df.loc[df['Similarity'] <= sim_]
        count = 0
        for index, row in df1.iterrows():
            if 'T' in row.tolist():
                count += 1
        if len(df1.index) > 0:
            # cumulative_count[sim_] = count/len(df1.index)
            cumulative_count[sim_] = count
    # df_new = pandas.DataFrame(cumulative_count.items(), columns=['Similarity', 'Percentage of smelly words'])
    df_new = pandas.DataFrame(cumulative_count.items(), columns=['Similarity', 'Smelly words'])
    return df_new


def rq2_sensitivity_analysis():
    #
    df = pd.read_csv('sensitivity_word2vec_matrix.csv')
    df1 = pandas.DataFrame()
    # df1['Word'] = df['Word']
    df2 = pd.DataFrame()
    df2['Word'] = df['Word']

    for col_ in df.columns[2:]:
        df2['Word'] = df['Word']
        df2['Similarity'] = df[col_]
        df3 = compute_cumulative_values(df2)
        parts = col_.split()
        df1['Smelly words in absence of ' + parts[-1]] = df3['Smelly words']

    df2 = pd.DataFrame()
    df3 = compute_cumulative_values(df2)
    df1['Smelly words all domains'] = df3['Smelly words']
    df1['Similarity'] = df3['Similarity']

    df_melt = df1.melt(id_vars=['Similarity'], var_name='Domain', value_name='Smelly words')
    print(df_melt)
    # quit()
    ax = sns.scatterplot(data=df_melt,
                         x=df_melt['Similarity'],
                         # y=df_new['Percentage of smelly words']
                         y=df_melt['Smelly words'],
                         hue=df_melt['Domain'], style=df_melt['Domain'],
                         s=8)
    ax.set(xticks=np.arange(0, 1.05, 0.05))

    last_smelly_word_similarity_factor = 0.577578277
    last_smelly_word_similarity_association = 0.594287619

    plt.axvline(last_smelly_word_similarity_association, 0, 250, linestyle='--', color='orange')
    plt.text(0.6, -0.5, '0.5943', color='red', )
    plt.tight_layout()
    plt.show()


def ranked_correlation_analysis():
    """
    The statistical test reports a strong positive correlation with a value of 0.9.
    The p-value is close to zero, which means that the likelihood of observing the data given that the samples
    are uncorrelated is very unlikely (e.g. 95% confidence) and that we can reject the null hypothesis that
    the samples are uncorrelated.

    :return:
    """
    pass


def compare_smella_and_arta_accuracy():
    data_file = r'data/DataLast/dataset1kv1_evaluationT_long_form.xlsx'
    df = pd.read_excel(data_file, sheet_name='evaluation_pandas')
    print(df.columns)
    df2 = df.melt(id_vars=['Smell', 'Method'], var_name='Metric', value_name='Value')
    g = sns.catplot(data=df2, x='Smell', y='Value', hue='Method', col='Metric', kind='bar', legend=False)

    def decorate_chart(g, new_value):
        hatches = itertools.cycle(['//', '\\\\', '//', '\\\\', '//', '\\\\', '*', 'o', 'O', '.'])
        for i in range(0, 3):
            for j, patch in enumerate(g.axes[0][i].patches):
                current_width = patch.get_width()
                diff = current_width - new_value
                # we change the bar width
                patch.set_width(new_value)
                # we recenter the bar
                patch.set_x(patch.get_x() + diff * .5)
                if j % 3 == 0:
                    hatch = next(hatches)
                patch.set_hatch(hatch)

    decorate_chart(g, new_value=0.30)
    g.axes[0][1].legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=False)
    plt.tight_layout()
    plt.show()


# ------------------------------------------
if __name__ == '__main__':
    # compute_requirements_stats()
    # compute_smell_prevalence()
    # compute_smell_frequency_percentage()
    # draw_barplot_of_mean_and_sd_of_smell_frequency()
    # draw_smells_against_size()
    # draw_project_smell_frequency()
    # draw_total_smell_frequency()
    # draw_total_smell_frequency2()
    # compute_requirement_testability()
    # evaluate_testability_measurement_method()
    # draw_cleanness_and_testability()
    regress_with_decision_tree(model_path=r'sklearn_models/DT_GT_v1_S2.joblib')
    # compute_requirement_length()
    # testability_histogram()
    # comparing_alphas()
    # rq2_sensitivity_analysis()
    # compare_smella_and_arta_accuracy()
