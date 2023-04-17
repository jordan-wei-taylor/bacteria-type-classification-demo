# simple linear model https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from   sklearn.linear_model    import LogisticRegression

# train test split helper function https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from   sklearn.model_selection import train_test_split        

# data reading package https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
import pandas as pd

# garbage collector to remove files once deleted to free RAM
import gc

# ignore non-fatal warning messages
import warnings

warnings.filterwarnings('ignore')

def main(name, X, y):
    """
    Parameters
    ==========
        name : str
               name of output csv

        X    : array[N, M]
               input observation array

        y    : array[N]
               target array with values {0, 1}

    Outputs
    ==========
        `name`.csv : csv file containing the Logistic Regression weight coefficients (see .ipynb for more info)
    """
    # split the data randomly (set random_state to control randomness for reproducible results)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, train_size = 0.75)

    # train model (ignore warning)
    model          = LogisticRegression(random_state = 0).fit(X_train, y_train)

    # print accuracy
    print(f'{name:6s} train accuracy :', model.score(X_train, y_train))
    print(f'{name:6s} test  accuracy :', model.score(X_test , y_test ))

    # re-train model on all data
    model          = LogisticRegression(random_state = 0).fit(X, y)

    # feature importance for a Logistic Regression model is based on the weight coefficients
    weights        = model.coef_.flatten()

    # create csv with importances learnt
    csv            = pd.DataFrame(index = df.columns[:-1], data = weights, columns = ['weight'])

    # order it by increasing order
    csv            = csv.iloc[weights.argsort()]

    # save to csv file
    csv.to_csv(f'{name}.csv', header = True, index = True)

# load the sflex data (noticed it repeated the first row so ignore the first row with skiprows = 1)
df           = pd.read_csv('data/Sflex_countsMAF0.05_dedup2381_uniqpatout.txt.gz', sep = '\t', skiprows = 1).set_index('pattern_id').T # .T swaps the rows and columns
df['target'] = pd.read_csv('data/Sflex_fullmar_1000B_y1_2381dedup_pheno.txt', sep = '\t').set_index('sample')

main('sflex', df.values[:,:-1], df.values[:,-1])

del df; gc.collect()

# load the sonnei data
df           = pd.read_csv('data/Sonneicounts_MAF0.05_3745_uniqpatout.txt.gz', sep = '\t').set_index('pattern_id').T
df['target'] = pd.read_csv('data/Sonneifullmar_1000B_y2_MSMcladeisolate_pheno.txt', sep = '\t').set_index('sample')

main('sonnei', df.values[:,:-1], df.values[:,-1])

# execution time : 4m 3s
# max RAM usage  : 21.66 GB
