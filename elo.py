import math
import os

import numpy as np
import pandas as pd
import requests


def main():
    pd.options.mode.chained_assignment = None
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)
    # get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    base_url = 'https://jellebuitenhuis.nl/backend/survivalrun/results/individual/paginated?'
    page_number = 0
    page_size = 10000
    response = requests.get(base_url + 'page=' + str(page_number) + '&pageSize=' + str(page_size))
    response_json = response.json()
    df_results = pd.DataFrame(response_json['content'])
    while not response_json['last']:
        response = requests.get(base_url + 'page=' + str(page_number + 1) + '&pageSize=' + str(page_size))
        response_json = response.json()
        page_number += 1
        df_results = df_results.append(pd.DataFrame(response_json['content']))

    df_results = convert_categories(df_results)

    # drop all rows where category is not 'LSR', 'MSR', 'KSR'
    df_results = df_results[df_results['category'].isin(['LSR', 'MSR', 'KSR'])]

    df_results = df_results[df_results['points'] != 0]
    # remove rows with empty values in column 'firstName' or 'lastName'
    df_results = df_results[df_results['firstName'].notnull()]
    df_results = df_results[df_results['lastName'].notnull()]

    # strip whitespace from column 'firstName' and 'lastName'
    df_results['firstName'] = df_results['firstName'].str.strip()
    df_results['lastName'] = df_results['lastName'].str.strip()

    # merge first and last name then drop first and last name
    df_results['name'] = df_results['firstName'] + ' ' + df_results['lastName']
    df_results.drop(columns=['firstName', 'lastName'], inplace=True)

    # convert 'name' to lowercase
    df_results['name'] = df_results['name'].str.lower()

    # sort by 'survivalrunDate' and 'category'
    df_results.sort_values(by=['survivalrunDate', 'category'], inplace=True)

    # get every unique combination of 'survivalrunDate' and 'category'
    unique_dates_categories = df_results[['survivalrunDate', 'category']].drop_duplicates()

    # Get all unique names
    names = df_results['name'].unique()
    df_elo = pd.DataFrame(columns=['name', 'elo', 'history', 'amount_of_runs'])
    df_elo['name'] = names
    # amount of runs is 0 for every name
    df_elo['amount_of_runs'] = [0] * len(names)
    # for every df_elo assign an empty dataframe to the column 'history'
    df_elo['history'] = [{} for _ in range(len(df_elo))]

    i = 0
    # for every date
    for row in unique_dates_categories.itertuples():
        date = row[1]
        category = row[2]
        calculate_new_elo(df_results, category, date, df_elo)
        i += 1
    df_elo.sort_values(by=['elo'], inplace=True)
    df_elo.to_csv('elo.csv', index=False)
    # return current directory for backend
    print(current_dir)


def calculate_new_elo(df_results, category, date, df_elo):
    # get the results for the first category and date
    df_results_category_date = df_results[
        (df_results['survivalrunDate'] == date) & (df_results['category'] == category)]

    # remove duplicate names
    df_results_category_date = df_results_category_date.drop_duplicates(subset=['name'])

    df_new_elo = calculate_scores(df_results_category_date, df_elo)

    if df_new_elo is None:
        return

    # for every row in df_new_elo
    for index, row in df_new_elo.iterrows():
        name = row['name']
        elo = row['elo']
        df_elo.loc[df_elo['name'] == name, 'elo'] = elo
        df_elo[df_elo['name'] == name]['history'].values[0][date] = elo
        df_elo.loc[df_elo['name'] == name, 'amount_of_runs'] += 1


def calculate_scores(df_results_category_date, df_elo):
    # sort df_results_category_date by 'position'
    df_results_category_date.sort_values(by=['name'], inplace=True)

    # print(df_results_category_date.head())

    participants = len(df_results_category_date)
    # calculate the amount of distinct matchups
    matchups = participants * (participants - 1) / 2

    if matchups == 0:
        return

    df_new_elo = pd.DataFrame(columns=['name', 'elo'])

    # get the elo of every participant from df_elo in a numpy array
    initial_ratings = df_elo[df_elo['name'].isin(df_results_category_date['name'])]
    # for every elo that is NaN in initial_ratings, assign the average elo
    initial_ratings['elo'].fillna(1000, inplace=True)
    # sort initial_ratings by 'name' using the same order as 'name' in df_results_category_date
    initial_ratings = initial_ratings.sort_values(by=['name'])
    initial_ratings = initial_ratings['elo'].values
    initial_ratings = np.array(initial_ratings)

    if not isinstance(initial_ratings, np.ndarray):
        initial_ratings = np.array(initial_ratings)
    if initial_ratings.ndim > 1:
        raise ValueError(f"initial_ratings should be 1-dimensional array (received {initial_ratings.ndim})")

    n = len(initial_ratings)

    expected_scores = get_expected_scores(initial_ratings, n)

    # get the position of every participant from df_results_category_date in an array sorted by 'position'
    positions = df_results_category_date['position'].values
    # positions to list
    positions = positions.tolist()

    actual_scores = get_actual_scores(positions, n)

    # assign the new elo to the df_new_elo
    for i in range(len(initial_ratings)):
        initial_rating = initial_ratings[i]
        new_rating = actual_scores[i] - expected_scores[i]
        amount_of_runs = df_elo[df_elo['name'] == df_results_category_date['name'].values[i]]['amount_of_runs'].values[0]
        k_runs = -1*amount_of_runs**2.8+256
        k = max(64, k_runs)
        scale_factor = k * (n - 1)
        new_rating = initial_rating + scale_factor * new_rating
        df_new_elo.loc[i] = [df_results_category_date['name'].values[i], new_rating]

    return df_new_elo


def get_actual_scores(positions, n):
    positions = positions or list(range(n))
    # sort the positions in ascending order
    positions = np.argsort(np.argsort(positions))
    # (participants - position) / matchups
    actual_scores = np.array([(n - p) / (n * (n - 1) / 2) for p in range(1, n + 1)])
    # sort the actual_scores by the positions
    actual_scores = actual_scores[positions]

    # if there are ties, average the actual_scores of all tied players
    distinct_results = set(positions)
    if len(distinct_results) != n:
        for place in distinct_results:
            idx = [i for i, x in enumerate(positions) if x == place]
            actual_scores[idx] = actual_scores[idx].mean()

    if not np.allclose(1, sum(actual_scores)):
        raise ValueError("scoring function does not return actual_scores summing to 1")
    if min(actual_scores) != 0:
        # tie for last place means minimum score doesn't have to be zero,
        # so only raise error if there isn't a tie for last place
        last_place = max(positions)
        if positions.count(last_place) == 1:
            raise ValueError("scoring function does not return minimum value of 0")
    if not np.all(np.diff(actual_scores[np.argsort(positions)]) <= 0):
        raise ValueError("scoring function does not return monotonically decreasing values")
    return actual_scores


def get_expected_scores(initial_ratings, n):
    d = 400
    rating_differences = initial_ratings - initial_ratings[:, np.newaxis]

    expected_scores_function = 1 / (1 + 10 ** (rating_differences / d))

    np.fill_diagonal(expected_scores_function, 0)
    expected_scores = expected_scores_function.sum(axis=1)
    expected_scores = expected_scores / (n * (n - 1) / 2)

    if not np.allclose(1, sum(expected_scores)):
        raise ValueError("expected actual_scores do not sum to 1")
    return expected_scores


def convert_categories(df_results):
    df_results.loc[df_results['category'].str.contains('BSR'), 'category'] = 'BSR'

    df_results.loc[df_results['category'].str.contains('KSR'), 'category'] = 'KSR'
    df_results.loc[df_results['category'].str.contains('BSC'), 'category'] = 'KSR'
    df_results.loc[df_results['category'].str.contains('Korte Survival Run'), 'category'] = 'KSR'
    df_results.loc[df_results['category'].str.contains('Basis Survival Circuit'), 'category'] = 'KSR'

    df_results.loc[df_results['category'].str.contains('MSR'), 'category'] = 'MSR'
    df_results.loc[df_results['category'].str.contains('RUC'), 'category'] = 'MSR'
    df_results.loc[df_results['category'].str.contains('RUC'), 'category'] = 'MSR'
    df_results.loc[df_results['category'].str.contains('Runner Up Circuit'), 'category'] = 'MSR'
    df_results.loc[df_results['category'].str.contains('Runner-up Circuit'), 'category'] = 'MSR'
    df_results.loc[df_results['category'].str.contains('Run Up Circuit'), 'category'] = 'MSR'
    df_results.loc[df_results['category'].str.contains('Middellange Suvivalrun'), 'category'] = 'MSR'

    df_results.loc[df_results['category'].str.contains('LSR'), 'category'] = 'LSR'
    df_results.loc[df_results['category'].str.contains('Lange Survivalrun'), 'category'] = 'LSR'
    df_results.loc[df_results['category'].str.contains('TSC'), 'category'] = 'LSR'
    df_results.loc[df_results['category'].str.contains('Top Survival Circuit'), 'category'] = 'LSR'

    df_results.loc[df_results['category'].str.contains('JSC'), 'category'] = 'Jeugd'
    df_results.loc[df_results['category'].str.contains('MRC'), 'category'] = 'Jeugd'
    df_results.loc[df_results['category'].str.contains('Jeugd'), 'category'] = 'Jeugd'

    # extract kilometers from 'category'
    df_results['km'] = df_results['category'].str.extract('(\d+)', expand=False)

    # if 'km' is not a number, set it to 0
    df_results.loc[df_results['km'].isna(), 'km'] = 0

    # convert 'km' to integer
    df_results['km'] = df_results['km'].astype(int)

    # if km > 9 but < 14, set category to 'Recreatief Middellang'
    df_results.loc[(df_results['km'] >= 15), 'category'] = 'Recreatief LSR'
    df_results.loc[(df_results['km'] >= 10) & (df_results['km'] <= 14), 'category'] = 'Recreatief MSR'
    df_results.loc[(df_results['km'] >= 7) & (df_results['km'] <= 9), 'category'] = 'Recreatief KSR'
    # df_results.loc[(df_results['km'] <= 6) & (df_results['km'] > 0), 'category'] = 'Recreatief Kort'
    df_results.loc[df_results['category'].str.contains('Kort', case=False), 'category'] = 'Recreatief KSR'
    df_results.loc[df_results['category'].str.contains('Lang'), 'category'] = 'Recreatief LSR'
    df_results.loc[df_results['category'].str.contains('Hele run', case=False), 'category'] = 'Recreatief LSR'
    df_results.loc[df_results['category'].str.contains('Middel'), 'category'] = 'Recreatief MSR'
    df_results.loc[df_results['category'].str.contains('Halve run', case=False), 'category'] = 'Recreatief MSR'
    df_results.loc[df_results['category'].str.contains('Recreanten middellange'), 'category'] = 'Recreatief MSR'

    # drop rows where category is not 'KSR', 'MSR' or 'LSR'
    df_results = df_results[
        df_results['category'].isin(['KSR', 'MSR', 'LSR', 'Recreatief KSR', 'Recreatief MSR', 'Recreatief LSR'])]

    return df_results

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)
    main()
