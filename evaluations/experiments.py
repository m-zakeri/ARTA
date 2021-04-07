"""
The script generated required result for experiments
"""

import pandas as pd
import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2 as ps

import squarify
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
    df = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability.xlsx')
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
    print('-'*75)

    sentences_list = []
    for index, row in df.iterrows():
        blob = tb.TextBlob(row['ReqTxt'])
        sentences_list.append(len(blob.sentences))
    print('Sentence counter',Counter(sentences_list))
    print('sum:', sum(sentences_list), 'avg:', sum(sentences_list)/len(sentences_list))
    print('weighted avg:', (461*2+432*1+3*77+4*6+5*5+6*2+7*2)/len(sentences_list))


def draw_barplot_of_mean_and_sd_of_smell_frequency():
    df = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency_long_form.xlsx')
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
        capsize=0.25,
        estimator=np.mean,
        ci='sd',
        n_boot=10000,
        height=3,
        aspect=1.5,
        kind=kind[6],
        data=df,
        # s=3.50,
        # color='0.1',
        # marker='*',
        palette=sns.color_palette('tab10'),

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
        axes.set_ylabel('Smell Frequency', fontsize=9)
        axes.tick_params(labelsize=8)
    # plt.xticks(rotation=45)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    plt.tight_layout()
    plt.savefig('charts/frequent_smells7.png')

    plt.show()


def draw_smells_against_size():
    df = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability.xlsx')

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
        # x_estimator=np.mean,
        # order=3,
        data=df,

        # palette="Set1",
        palette='plasma',
        aspect=1.2,
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
        legend_out=False
    )

    g = sns.scatterplot(
        x='Words',
        y='SmellyWords',
        size='Clarity', # The same cleanness
        sizes=(0.1, 100.1),
        hue='Project',
        hue_order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        style='Project',
        style_order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        estimator=np.mean,
        data=df,
        ax=g.ax
    )

    g.set(xlabel='Requirement length (word)', ylabel='Number of smells')
    # check axes and find which is have legend
    # leg = g.axes.flat[0].get_legend()
    # new_title = 'My title'
    # leg.set_title(new_title)
    # new_labels = ['label 1', 'label 2']
    # for t, l in zip(leg.texts, new_labels): t.set_text(l)

    plt.tight_layout()
    plt.savefig(r'charts/smells_against_size_v5.png')
    plt.show()


def draw_project_smell_frequency():
    pass


def draw_total_smell_frequency():
    # https://nbviewer.jupyter.org/gist/fonnesbeck/5850463
    df = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency.xlsx')
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
    plt.savefig('charts/total_smell_frequency_boxplot_v4.png')
    plt.show()


def draw_total_smell_frequency2():
    # https://nbviewer.jupyter.org/gist/fonnesbeck/5850463
    df = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency.xlsx')
    kind = ['strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count']
    g = sns.catplot(
        y='Project',
        x='SmellyWords',
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
        x='SmellyWords',
        # hue='Type',
        order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        size=2.5,
        s=2.5,
        # color='0.1',
        # marker='*',
        palette=sns.color_palette('tab10'),
        linewidth=1.5,
        edgecolor='gray',
        data=df,
        ax=g.ax)

    plt.tight_layout()
    plt.savefig('charts/total_smell_frequency_boxplot_v5.png')
    plt.show()


def compute_requirement_testability():
    df = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency.xlsx')  # Ground-truth dataset
    testability_a00 = list()
    testability_a01 = list()
    testability_a_05 = list()
    testability_a_10 = list()
    cleanness = list()

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
            cleanness_i = 1 - (smelly_words / all_words) ** (1 / t)
        alpha = [0.00, 0.01, 0.50, 0.99, ]  # For our paper we set alpha/epsilon cost to 0.01

        # Reasoning about alpha? According to Reviewers' comments
        #
        # testability_i = cleanness_i / ((1 + alpha[0]) ** (sentence - 1))
        # print(testability_i)
        testability_a00.append(cleanness_i / ((1 + alpha[0]) ** (sentence - 1)))
        testability_a01.append(cleanness_i / ((1 + alpha[1]) ** (sentence - 1)))
        testability_a_05.append(cleanness_i / ((1 + alpha[2]) ** (sentence - 1)))
        testability_a_10.append(cleanness_i / ((1 + alpha[3]) ** (sentence - 1)))

        cleanness.append(cleanness_i)
        # testability.append(testability_i)


    df['Cleanness'] = cleanness
    # df['Testability'] = testability
    df['Testability_with_alpha_0.00'] = testability_a00
    df['Testability_with_alpha_0.01'] = testability_a01
    df['Testability_with_alpha_0.05'] = testability_a_05
    df['Testability_with_alpha_0.99'] = testability_a_10

    print(df)
    df.to_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability_with_alphas.xlsx')  # Ground-truth dataset
    # df.to_excel(r'data/DataLast/dataset1kv1_Smella_result_smell_frequency_with_testability.xlsx', index=False)


def evaluate_testability_measurement_method():
    # Use scikit learn evaluation metrics for regression

    df_gt = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability.xlsx')  # Ground-truth dataset
    df_arta = pd.read_excel(r'data/DataLast/dataset1kv1_ARTA_result_smell_frequency_with_testability.xlsx')
    df_smella = pd.read_excel(r'data/DataLast/dataset1kv1_Smella_result_smell_frequency_with_testability.xlsx')

    y_true, y_pred = list(df_gt['Testability'].values), list(df_arta['Testability'].values)

    y_true = y_true[961:985]
    y_pred = y_pred[961:985]

    # Print all classifier model metrics
    print('Evaluating requirement testability ...')
    print('Regressor minimum prediction', min(y_pred), 'Regressor maximum prediction', max(y_pred))
    df = pd.DataFrame()
    df['r2_score_uniform_average'] = [r2_score(y_true, y_pred, multioutput='uniform_average')]
    df['r2_score_variance_weighted'] = [r2_score(y_true, y_pred, multioutput='variance_weighted')]

    df['explained_variance_score_uniform_average'] = [
        explained_variance_score(y_true, y_pred, multioutput='uniform_average')]
    df['explained_variance_score_variance_weighted'] = [
        explained_variance_score(y_true, y_pred, multioutput='variance_weighted')]

    df['mean_absolute_error'] = [mean_absolute_error(y_true, y_pred)]
    df['mean_squared_error_MSE'] = [mean_squared_error(y_true, y_pred)]
    df['mean_squared_error_RMSE'] = [mean_squared_error(y_true, y_pred, squared=False)]
    df['median_absolute_error'] = [median_absolute_error(y_true, y_pred)]

    if min(y_pred) >= 0:
        df['mean_squared_log_error'] = [mean_squared_log_error(y_true, y_pred)]
        # ValueError: Mean Tweedie deviance error with power=2 can only be used on strictly positive y and y_pred.
        df['mean_poisson_deviance'] = [mean_poisson_deviance(y_true, y_pred, )]
        df['mean_gamma_deviance'] = [mean_gamma_deviance(y_true, y_pred, )]
    df['max_error'] = [max_error(y_true, y_pred)]

    df.to_excel(r'data/DataLast/dataset1kv1_ARTA_evaluation_metrics_6_Peering.xlsx', index=True, index_label='Row')


def draw_cleanness_and_testability():
    df_gt = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability.xlsx')  # Ground-truth dataset
    df_arta = pd.read_excel(r'data/DataLast/dataset1kv1_ARTA_result_smell_frequency_with_testability.xlsx')
    df_smella = pd.read_excel(r'data/DataLast/dataset1kv1_Smella_result_smell_frequency_with_testability.xlsx')

    df = pd.DataFrame()
    df['ReqTxt'] = df_gt['ReqTxt']
    df['Project'] = df_gt['Project']

    df['Clarity'] = df_gt['Clarity']
    df['Ground-truth (Manual)'] = df_gt['Testability']

    df['Clarity (ARTA)'] = df_arta['Clarity']
    df['ARTA'] = df_arta['Testability']

    df['Clarity (Smella)'] = df_smella['Clarity']
    df['Smella'] = df_smella['Testability']

    # print(df)

    df.drop(columns=['Clarity', 'Clarity (ARTA)', 'Clarity (Smella)'], inplace=True)
    df2 = pd.melt(df, id_vars=['ReqTxt', 'Project'], value_name='Testability (α = 0.01)', var_name='Method/Tool')
    print(df2)
    # df2.to_excel('Temp.xlsx')
    # return
    # draw_project_smell_frequency:
    # fig, ax = plt.subplots(figsize=(12,8))
    kind = ['strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count']
    g = sns.catplot(
        x='Project',
        y='Testability (α = 0.01)',
        hue='Method/Tool',
        # dodge=True,
        # col='Project',
        # col_wrap=3,
        order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        # col_order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        height=5,
        aspect=2,
        data=df2,
        kind=kind[5],
        # s=3.50,
        # color='0.1',
        # marker='*',
        palette=sns.color_palette(palette='tab10', n_colors=6),
        # capsize=0.20,
        # ci=None,
        n_boot=100000,
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
        y='Testability (α = 0.01)',
        hue='Method/Tool',
        order=['EIRENE', 'ERTMS/ETCS', 'CCTNS', 'Gamma-J', 'KeePass', 'Peering'],
        # hue_order=['Original', 'Refactored', ],
        dodge=True,
        data=df2,
        s=2.25,
        # color='0.1',
        # marker='*',
        palette=sns.color_palette('tab10'),
        linewidth=0.15,
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
    plt.savefig('charts/requirement_testability_v3.png')

    plt.show()


def regress_with_decision_tree(model_path):
    df_gt = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability.xlsx')  # Ground-truth dataset
    X = df_gt.iloc[:, 8:-2]
    # print(X)
    # return
    y = df_gt.iloc[:, -2]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42)
    X_train, y_train = X, y

    clf = tree.DecisionTreeRegressor()

    # CrossValidation iterator object:
    cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=42)

    # Set the parameters to be used for tuning by cross-validation
    parameters = {'max_depth': range(1, 100, 1),
                  'criterion': ['mse', 'friedman_mse', 'mae'],
                  # 'criterion': ['mse'],
                  'min_samples_split': range(2, 100, 1)
                  }
    # Set the objectives which must be optimized during parameter tuning
    # scoring = ['r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_error',]
    scoring = ['neg_root_mean_squared_error', ]

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
    plot_tree(clf, filled=True, max_depth=2, feature_names=df_gt.columns[8:-2])
    plt.tight_layout()
    plt.savefig(model_path[:-7] + 'tree_plot.png')
    plt.show()


def compute_requirement_length():
    req = """When the train is stationary or after a certain time (e.g. the time for 'route releasing' of the overlap, the release speed calculation shall be based on the distance to the danger point (if calculated on-board). The condition for this change shall be defined for each target as infrastructure data."""
    blob = tb.TextBlob(req)
    print('# words', len(blob.words))
    print('# sentence', len(blob.sentences))

def testability_histogram():
    df_gt = pd.read_excel(r'data/DataLast/dataset1kv1_smell_frequency_with_testability.xlsx')
    df_arta = pd.read_excel(r'data/DataLast/dataset1kv1_ARTA_result_smell_frequency_with_testability.xlsx')

    sentences_list = []
    for index, row in df_gt.iterrows():
        blob = tb.TextBlob(row['ReqTxt'])
        sentences_list.append(len(blob.sentences))
    df_gt['Sentences'] = sentences_list
    print(Counter(sentences_list))

    df_gt['Testability_ARTA']= df_arta['Testability']
    # fig, ax = plt.subplots()
    g = sns.displot(df_gt, x='Testability_ARTA', hue='Project',
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
    g.ax.set(xlabel='Testability')
    # plt.tight_layout()
    plt.savefig('charts/testability_arta_with_projects_histogram_v1.png')
    plt.show()


def comparing_alphas(csv_path=r'data/DataLast/dataset1kv1_smell_frequency_with_testability_with_alphas.xlsx'):
    df_gt = pd.read_excel(csv_path)  # Ground-truth dataset

    df = pd.DataFrame()
    df['ReqTxt'] = df_gt['ReqTxt']
    df['Project'] = df_gt['Project']

    # Variables
    # df['Cleanness'] = df_gt['Cleanness']
    df['α = 0'] = df_gt['Testability_with_alpha_0.00']
    df['α = 0.01'] = df_gt['Testability_with_alpha_0.01']
    df['α = 0.50'] = df_gt['Testability_with_alpha_0.05']
    df['α = 0.99'] = df_gt['Testability_with_alpha_0.99']

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
        markers=['o', 'x', '+', '*',],
        linestyles=['solid', '-', '--', ':'],
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
        s=2.25,
        # color='0.1',
        # marker='*',
        palette=sns.color_palette('tab10'),
        linewidth=0.15,
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
    plt.savefig('charts/requirement_testability_comparing_alphas_v3.png')

    plt.show()


# ------------------------------------------
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
# regress_with_decision_tree(model_path=r'sklearn_models/DT_GT_v2.joblib')
# compute_requirement_length()
#testability_histogram()
comparing_alphas()