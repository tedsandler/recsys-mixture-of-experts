import os
import re
import datetime as dt
import numpy as np
import pandas as pd


movielens_dir = './ml-1m'
output_dir = f"{movielens_dir}/processed"

encoding = "ISO-8859-1"

age_groups = {
	1:  "01-18",
	18:  "18-24",
	25:  "25-34",
	35:  "35-44",
	45:  "45-49",
	50:  "50-55",
    56:  "56+",
}

occupations = {
    0:  "unknown",
    1:  "academic/educator",
    2:  "artist",
    3:  "clerical/admin",
    4:  "college/grad student",
    5:  "customer service",
    6:  "doctor/health care",
    7:  "executive/managerial",
    8:  "farmer",
    9:  "homemaker",
    10:  "k-12 student",
    11:  "lawyer",
    12:  "programmer",
    13:  "retired",
    14:  "sales/marketing",
    15:  "scientist",
    16:  "self-employed",
    17:  "technician/engineer",
    18:  "tradesman/craftsman",
    19:  "unemployed",
    20:  "writer",
}

def load_users(users_file):
    users = []
    with open(users_file, encoding=encoding) as fin:
        for line in fin:
            fields = line.strip().split('::')
            assert len(fields) == 5
            user_id = int(fields[0]) - 1
            gender = fields[1]
            age_group_id = int(fields[2])
            occupation_id = int(fields[3])
            # parse zip5 zipcode
            zip_code = re.search('^\\d{5}', fields[4]).group(0)
            users.append({
                'user_id' : user_id,
                'gender' : gender,
                'age_group' : age_groups[age_group_id],
                'occupation' : occupations[occupation_id],
                'zip3' : zip_code[:3],
                'zip5' : zip_code,
            })
    return pd.DataFrame(users)


def load_movies(movies_file):
    movies = []
    with open(movies_file, encoding=encoding) as fin:
        for line in fin:
            fields = line.strip().split('::')
            assert len(fields) == 3
            movie_id = int(fields[0]) - 1
            title = fields[1]
            genres = fields[2].split('|')
            movies.append({
                'movie_id' : movie_id,
                'title' : title,
                'genres' : genres,
            })
    return pd.DataFrame(movies)


def load_ratings(ratings_file):
    ratings = []
    with open(ratings_file, encoding=encoding) as fin:
        for line in fin:
            fields = line.strip().split('::')
            user_id = int(fields[0]) - 1
            movie_id = int(fields[1]) - 1
            rating = int(fields[2])
            timestamp = dt.datetime.fromtimestamp(int(fields[3]))
            ratings.append({
                'user_id' : user_id,
                'movie_id' : movie_id,
                'rating' : rating,
                'timestamp' : timestamp,
            })
        return pd.DataFrame(ratings)


def make_dummies(features):
    feature_set = set()

    for feature_row in features:
        if type(feature_row) != list:
            feature_row = [feature_row]
        for feature_name in feature_row:
            feature_set.add(feature_name)

    dummy_index = {
        feature_name : ix
        for ix, feature_name
        in enumerate(sorted(feature_set))
    }

    A = np.zeros((len(features), len(dummy_index)))

    for i, feature_row in enumerate(features):
        if type(feature_row) != list:
            feature_row = [feature_row]
        for feature_name in feature_row:
            j = dummy_index[feature_name]
            A[i,j] = 1

    return A, dummy_index


def make_ratings_matrices(df_ratings):
    ii = df_ratings.user_id
    jj = df_ratings.movie_id
    num_users = ii.max() + 1
    num_movies = jj.max() + 1
    R = np.zeros((num_users, num_movies))
    R[ii, jj] = df_ratings.rating
    Y = np.zeros((num_users, num_movies))
    Y[ii, jj] = 1
    return R, Y


if __name__ == "__main__":
    df_users = load_users(f"{movielens_dir}/users.dat")
    df_movies = load_movies(f"{movielens_dir}/movies.dat")
    df_ratings = load_ratings(f"{movielens_dir}/ratings.dat")

    X_gender, ix_gender = make_dummies(df_users.gender)
    X_age_group, ix_age_group = make_dummies(df_users.age_group)
    X_occupation, ix_occupation = make_dummies(df_users.occupation)
    X_zip3, ix_zip3 = make_dummies(df_users.zip3)

    X = np.concatenate([
        X_gender, X_age_group, X_occupation, X_zip3,
    ], axis=1)

    X_cols = np.concatenate([
        [f"gender::{s}" for s in ix_gender.keys()],
        [f"age::{s}" for s in ix_age_group.keys()],
        [f"occupation::{s}" for s in ix_occupation.keys()],
        [f"zip3::{s}" for s in ix_zip3.keys()],
    ])
    
    Z_genres, ix_genres = make_dummies(df_movies.genres)
    title_word_features = df_movies.title.apply(lambda s: re.split("\\W+", s.lower()))
    Z_title_words, ix_title_words = make_dummies(title_word_features)

    Z = np.concatenate([Z_genres, Z_title_words], axis=1)
    Z_cols = np.concatenate([
        [f"genre::{s}" for s in ix_genres.keys()],
        [f"title_word::{s}" for s in ix_title_words.keys()],
    ])

    os.makedirs(output_dir, exist_ok=True)
    df_users.to_csv(f"{output_dir}/users.csv", sep="\t", index=False)
    df_movies.to_csv(f"{output_dir}/movies.csv", sep="\t", index=False)

    np.save(f"{output_dir}/X.npy", X)
    with open(f"{output_dir}/X.cols", "w") as fout:
        for i, colname in enumerate(X_cols):
            fout.write(f"{i}\t{colname}\n")

    np.save(f"{output_dir}/Z.npy", Z)
    with open(f"{output_dir}/Z.cols", "w") as fout:
        for i, colname in enumerate(Z_cols):
            fout.write(f"{i}\t{colname}\n")

    R, Y = make_ratings_matrices(df_ratings)
    np.save(f"{output_dir}/R.npy", R)
    np.save(f"{output_dir}/Y.npy", Y)
