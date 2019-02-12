import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt

###  ----- Questions Section
# ----------------------------------------------------------------------------------
# given alcohol, diet, and drugs can I predict body type? (classifier)
# given sex and education can I guess essay length? (regression)

###  ----- Data Exploration Section
# ----------------------------------------------------------------------------------

df = pd.read_csv("profiles.csv")

print(df.drinks.value_counts())
print(df.drugs.value_counts())
print(df.diet.value_counts().sort_index())
print(df.body_type.value_counts().sort_index())
print(df.sex.value_counts())
print(df.education.value_counts())
#print(df.job.head())
#print(df.body_type.head())
#print(df.diet.head())



###  ----- Create New Data Section
# ----------------------------------------------------------------------------------

#Combine essays into all one string, then split them
# into a word list to find the average word length
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]


# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
essay_total = all_essays
#adding total essay length to the data
df["essay_len"] = all_essays.apply(lambda x: len(x))

#print(df.essay_len.head())

#creating a list of words and looping through to find average
#word_list = essay_total.apply(lambda x: ' '.split(x), axis=1)
#word_average = 0
#for i in range(word_list):
#    word_average += len(word_list[i])
#word_average = word_average / range(word_list)
#adding the word length average to the data set
#df["avg_word_length"] = word_average

# replace all NaNs with empty strings in the drinks, drugs, diet,
# body_type, sex, and education data so that I don't have to deal
# with them when creating the number valued mappings of them for
# the classifier

#print(df.isnull().sum())

df.drinks.replace(np.nan, '', inplace = True, regex=True)
df.drugs.replace(np.nan, '', inplace = True, regex=True)
df.diet.replace(np.nan, '', inplace = True, regex=True)
df.body_type.replace(np.nan, '', inplace = True, regex=True)
df.sex.replace(np.nan, '', inplace = True, regex=True)
df.education.replace(np.nan, '', inplace = True, regex=True)

#print(df.isnull().sum())

#re-map drinks to numbers
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5, "": 6}
df["drinks_code"] = df.drinks.map(drink_mapping)

#re-map drugs to numbers
drug_mapping = {"never": 0, "sometimes": 1, "often": 2, "": 3}
df["drugs_code"] = df.drugs.map(drug_mapping)

#re-map diet to numbers
diet_mapping = {"anything": 0, "halal": 1, "kosher": 2, "mostly anything": 3, "mostly halal": 4,
                "mostly kosher": 5, "mostly other": 6, "mostly vegan": 7, "mostly vegetarian": 8,
                "other": 9, "strictly anything": 10, "strictly halal": 11, "strictly kosher": 12,
                "strictly other": 13, "strictly vegan": 14, "strictly vegetarian": 15,
                "vegan": 16, "vegetarian": 17, "": 18}
df["diet_code"] = df.diet.map(diet_mapping)

#re-map body_type to numbers
body_mapping = {"a little extra": 0, "athletic": 1, "average": 2, "curvy": 3, "fit": 4, "full figured": 5,
                "jacked": 6, "overweight": 7, "rather not say": 8, "skinny": 9, "thin": 10, "used up": 11, "": 12}
df["body_type_code"] = df.body_type.map(body_mapping)

#re-map sex to numbers
sex_mapping = {"m": 0, "f": 1, "": 2}
df["sex_code"] = df.sex.map(sex_mapping)

#re-map drugs to numbers
education_mapping = {"graduated from college/university": 0, "graduated from masters program": 1, "working on college/university": 2,
                "working on masters program": 3, "graduated from two-year college": 4, "graduated from high school": 5,
                "graduated from ph.d program": 6, "graduated from law school": 7, "working on two-year college": 8,
                "dropped out of college/university": 9, "working on ph.d program": 10, "college/university": 11,
                "graduated from space camp": 12, "dropped out of space camp": 13, "graduated from med school": 14,
                "working on space camp": 15, "working on law school": 16, "two-year college": 17, "working on med school": 18,
                "dropped out of two-year college": 19, "dropped out of masters program": 20, "masters program": 21,
                "dropped out of ph.d program": 22, "dropped out of high school": 23, "": 24, "high school": 25,
                "working on high school": 26, "space camp": 27, "ph.d program": 28, "law school": 29,
                "dropped out of law school": 30, "dropped out of med school": 31, "med school": 32, "": 33}
df["education_code"] = df.education.map(education_mapping)

###  ----- Data Plot Section
# ----------------------------------------------------------------------------------

# plot essay length histogram
#plt.hist(df.essay_len, bins=200)
#plt.xlabel("Essay Length")
#plt.ylabel("Frequency")
#plt.xlim(0, 10000)
#plt.show()


# --- plot
#plt.hist(df.height, bins=200)
#plt.xlabel("Height (inches)")
#plt.ylabel("Frequency")
#plt.xlim(55, 90)
#plt.show()


###  ----- Normalization of Classifier Data


feature_data = df[['diet_code', 'drugs_code', 'drinks_code']]


classifier_X = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(classifier_X)

classifier_Y = df[['body_type_code']]

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

#classifier_Y.ravel(order='C')

###  ----- Classifiers Section
# ----------------------------------------------------------------------------------

# given alcohol, diet, and drugs can I predict body type? (classifier)


c_x_train, c_x_test, c_y_train, c_y_test = train_test_split(classifier_X, classifier_Y, train_size = 0.8, test_size = 0.2, random_state=6)

#print(classifier_X.__format__)
#print(classifier_Y.shape[0])

k_means_classifier = KNeighborsClassifier(n_neighbors = 10)
k_means_classifier.fit(c_x_train, c_y_train)

svm_classifier = SVC(kernel = 'linear')
svm_classifier.fit(c_x_train, c_y_train)


k_means_guesses = k_means_classifier.predict(c_x_test)
svm_guesses = svm_classifier.predict(c_x_test)

print("K-Means Accuracy, Recall, Precision, and F1 score are as follows:")
print(accuracy_score(c_y_test, k_means_guesses))
print(recall_score(c_y_test, k_means_guesses, average='macro'))
print(precision_score(c_y_test, k_means_guesses, average='macro'))
print(f1_score(c_y_test, k_means_guesses, average='macro'))


print("Support Vector Machine Accuracy, Recall, Precision, and F1 score are as follows:")
print(accuracy_score(c_y_test, svm_guesses))
print(recall_score(c_y_test, svm_guesses, average='macro'))
print(precision_score(c_y_test, svm_guesses, average='macro'))
print(f1_score(c_y_test, svm_guesses, average='macro'))

#print(guesses)

###  ----- Normalization of Regression Data
# ----------------------------------------------------------------------------------


feature_data1 = df[['sex_code', 'education_code']]


regression_X = feature_data1.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(regression_X)

regression_Y = df[['essay_len']]

feature_data1 = pd.DataFrame(x_scaled, columns=feature_data1.columns)


###  ----- Regression Section
# ----------------------------------------------------------------------------------

# given sex and education can I guess essay length? (regression)

r_x_train, r_x_test, r_y_train, r_y_test = train_test_split(regression_X, regression_Y, train_size = 0.8, test_size = 0.2, random_state=6)

mlr_regressor = LinearRegression()
mlr_regressor.fit(r_x_train, r_y_train)

k_regressor = KNeighborsRegressor(n_neighbors = 10, weights = "distance")
k_regressor.fit(r_x_train, r_y_train)

mlr_guesses = mlr_regressor.predict(r_x_test)
k_r_guesses = k_regressor.predict(r_x_test)

print("MLR Accuracy score for training and testing set are as follows:")
print(mlr_regressor.score(r_x_train, r_y_train))
print(mlr_regressor.score(r_x_test, r_y_test))

print("K-Regressor Accuracy, Recall, Precision, and F1 score are as follows:")
print(k_regressor.score(r_x_train, r_y_train))
print(k_regressor.score(r_x_test, r_y_test))

print("End of Code Test Statement")

