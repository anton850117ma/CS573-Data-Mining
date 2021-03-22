import pandas as pd
quotes_changed = 0
lowered_case = 0
label_encoding = {}

def perform_label_encoding(column):
    column = column.astype('category')
    codes_for_column = {}
    for i, category in enumerate(column.cat.categories):
        codes_for_column[category] = i
    label_encoding[column.name] = codes_for_column
    return column.cat.codes

def remove_quotes(x):
    global quotes_changed
    if "'" in x:
        quotes_changed = quotes_changed + 1
        return x.replace("'", "")
    else:
        return x

def to_lower(x):
    global lowered_case
    if x.islower():
        return x
    else:
        lowered_case = lowered_case + 1
        return x.lower()

#Read the dataset
data = pd.read_csv('dating-full.csv')
decision_col = data['decision']

#Remove quotes
data['race'] = data['race'].apply(remove_quotes)
data['race_o'] = data['race_o'].apply(remove_quotes)
data['field'] = data['field'].apply(remove_quotes)

#Convert to lowercase
data['field'] = data['field'].apply(to_lower)

# print('Quotes removed from '+str(quotes_changed) + ' cells.')
# print('Standardized '+str(lowered_case) + ' cells to lower case.')


#Label encode
data[['race','race_o','gender','field']] = data[['race','race_o','gender','field']].apply(perform_label_encoding)

# print('Value assigned for male in column gender:', label_encoding['gender']['male'])
# print('Value assigned for European/Caucasian-American in column race:',
#       label_encoding['race']['European/Caucasian-American'])
# print('Value assigned for Latino/Hispanic American in column race o:',
#       label_encoding['race_o']['Latino/Hispanic American'])
# print('Value assigned for law in column field:', label_encoding['field']['law'])

#Normalize preference scores of the participant
columns1  = ['attractive_important', 'sincere_important', 'intelligence_important','funny_important', 'ambition_important', 'shared_interests_important']
columns2  = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',
             'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']
data[columns1] = data[columns1].div(data[columns1].sum(axis=1), axis=0)
data[columns2] = data[columns2].div(data[columns2].sum(axis=1), axis=0)

# for column in columns1:
#     print('Mean of '+column+': ' + str(round(data[column].mean(),2)))

# for column in columns2:
#     print('Mean of '+column+': ' + str(round(data[column].mean(),2)))

#Move the target class to the end
data = data.drop(['decision'], axis = 1)
data['decision'] = decision_col

#Save the csv file
data.to_csv('dating2.csv', index = False)
