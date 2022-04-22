from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics  import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor



#command to execute the code
# streamlit run C:\Users\Kushagra\Desktop\jpmc\webapp.py


st.header("""CROP YIELD PREDICTION""");

#get the data
df_yield = pd.read_csv('yield.csv') 
df_yield = df_yield.drop(['Year Code','Element Code','Element','Year Code','Area Code','Domain Code','Domain','Unit','Item Code'], axis=1)
df_yield = df_yield.rename(index=str, columns={"Value": "hg/ha_yield"})
# df_yield = df_yield.drop(['Year Code', 'Element Code', 'Element', 'Year Code', 'Area Code', 'Domain Code', 'Domain', 'Unit', 'Item Code'], axis=1)

#set a header
st.subheader("Yield Information:")

#show the data
st.dataframe(df_yield)

#showing the dataset of the data
st.write(df_yield.describe())

st.write(df_yield.info())



###rainfall data

#get the data
df_rain = pd.read_csv('rainfall.csv')
df_rain = df_rain.rename(index=str, columns={" Area": 'Area'})
df_rain['average_rain_fall_mm_per_year'] = pd.to_numeric(df_rain['average_rain_fall_mm_per_year'],errors = 'coerce')
df_rain = df_rain.dropna()

#set a header
st.subheader("Rainfall Information:")

#show the data
st.dataframe(df_rain)

#showing the dataset of the data
st.write(df_rain.describe())

st.write(df_rain.info())


#merging the two dataframes
yield_df = pd.merge(df_yield, df_rain, on=['Year','Area'])


###pesticides data

#get the data

df_pes = pd.read_csv('pesticides.csv')
df_pes = df_pes.rename(index=str, columns={"Value": "pesticides_tonnes"})
df_pes = df_pes.drop(['Element','Domain','Unit','Item'], axis=1)


#set a header
st.subheader("Pesticide Information:")

#show the data
st.dataframe(df_pes)

#showing the dataset of the data
st.write(df_pes.describe())

st.write(df_pes.info())

#merging the two dataframes
yield_df = pd.merge(yield_df, df_pes, on=['Year','Area'])


###avg temp dataset
avg_temp=  pd.read_csv('temp.csv')

avg_temp = avg_temp.rename(index=str, columns={"year": "Year", "country":'Area'})

#set a header
st.subheader("Avg. Rainfall Information:")

#show the data
st.dataframe(avg_temp)

#showing the dataset of the data
st.write(avg_temp.describe())

st.write(avg_temp.info())

##merging the two datasets
yield_df = pd.merge(yield_df,avg_temp, on=['Area','Year'])


####The final dataframe
yield_df.groupby('Item').count()
yield_df['Area'].nunique()
yield_df.groupby(['Area'],sort=True)['hg/ha_yield'].sum().nlargest(10)
yield_df.groupby(['Item','Area'],sort=True)['hg/ha_yield'].sum().nlargest(10)

st.header("The final Dataframe")

st.dataframe(yield_df)

st.subheader("Information regarfing the dataframe:")

#showing the dataset of the data
st.write(yield_df.describe())

st.write(yield_df.info())



###Time for data exploration, finding whether there's any correlation between the variables



correlation_data=yield_df.select_dtypes(include=[np.number]).corr()

mask = np.zeros_like(correlation_data, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.palette="vlag"

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_data, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5});
st.write(f)
st.write("It can be seen from the above correlation map that there is no correlation between any of the colmuns in the dataframe.")


#Data Preprocessing
st.header("Data Preprocessing")
st.write("Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.")

st.subheader("Encoding Categorical Variables:")
st.write('''There are two categorical columns in the dataframe, categorical data are variables that contain label values rather than numeric values. The number of possible values is often limited to a fixed set, like in this case, items and countries values. Many machine learning algorithms cannot operate on label data directly. They require all input variables and output variables to be numeric.

This means that categorical data must be converted to a numerical form. One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction. For that purpose, One-Hot Encoding will be used to convert these two columns to one-hot numeric array.

The categorical value represents the numerical value of the entry in the dataset. This encoding will create a binary column for each category and returns a matrix with the results.''')


yield_df_onehot = pd.get_dummies( yield_df, columns=['Area', "Item"], prefix=['Country', "Item"])
features = yield_df_onehot.loc[:, yield_df_onehot.columns != 'hg/ha_yield']
label = yield_df['hg/ha_yield']
features = features.drop(['Year'], axis=1)

st.subheader("One-Hot Encoded data:")
st.dataframe(features.head())
st.write(features.info())


#scaling the data
st.subheader("Scaling Features:")
st.write('''Taking a look at the dataset above, it contains features highly varying in magnitudes, units and range. The features with high magnitudes will weigh in a lot more in the distance calculations than features with low magnitudes.

To supress this effect, we need to bring all features to the same level of magnitudes. This can be acheived by scaling.''')

scaler = MinMaxScaler()
features = scaler.fit_transform(features) 

st.write(features)


#train test split dataset

st.header("Training Data")

st.write("""For training of the model, we use majority of the data to training the model and the rest of the data is used for testing the model. Here, we are using 80% of the data for training and 20% of the data for testing.

The training dataset is the intial dataset used to train ML algorithm to learn and produce right predictions. (80% of dataset is training dataset)

The test dataset, however, is used to assess how well ML algorithm is trained with the training dataset.
""")


train_data, test_data, train_labels, test_labels = train_test_split(features, label, test_size=0.2, random_state=42)


#Model comparing, training and selection
st.header("Model Comparison and Selection")

st.write("""Here we test 4 different regressor algorithms namely: 
1. Gradient Boosting Regressor 
2. Random Forest Regressor
3. Decision Tree Regressor
4. Support Vector Regressor
""")



# not printing all of this as it takes too much time to run, rather just pasting the output as obtained on the kaggle notebook

# models = [
#     GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0),
#     RandomForestRegressor(n_estimators=200, max_depth=3, random_state=0),
#     svm.SVR(),
#     DecisionTreeRegressor()
# ]
# model_train=list(map(compare_models,models)) 

# print(*model_train, sep="\n") 

st.subheader("Testing of the model on test dataset on kaggle notebook")


st.write("""The output below shows the accuracy of the model, the accuracy is the percentage of the correct predictions.

['GradientBoostingRegressor', 0.8959545600619471]

['RandomForestRegressor', 0.6807690552605921]

['SVR', -0.19466686625412555]

['DecisionTreeRegressor', 0.9605155680634376]


Therefore, from the above output, we can see that the Decision Tree Regressor is the best model to use for this dataset.""")



#minor fitting in the dataframe

yield_df_onehot = yield_df_onehot.drop(['Year'], axis=1)


test_df=pd.DataFrame(test_data,columns=yield_df_onehot.loc[:, yield_df_onehot.columns != 'hg/ha_yield'].columns) 

cntry=test_df[[col for col in test_df.columns if 'Country' in col]].stack()[test_df[[col for col in test_df.columns if 'Country' in col]].stack()>0]
cntrylist=list(pd.DataFrame(cntry).index.get_level_values(1))
countries=[i.split("_")[1] for i in cntrylist]
itm=test_df[[col for col in test_df.columns if 'Item' in col]].stack()[test_df[[col for col in test_df.columns if 'Item' in col]].stack()>0]
itmlist=list(pd.DataFrame(itm).index.get_level_values(1))
items=[i.split("_")[1] for i in itmlist]

st.write(test_df.head())

test_df.drop([col for col in test_df.columns if 'Item' in col],axis=1,inplace=True)
test_df.drop([col for col in test_df.columns if 'Country' in col],axis=1,inplace=True)
st.write(test_df.head())

test_df['Country'] = countries
test_df['Item'] = items
st.write(test_df.head())


#prediction

st.header("Prediction")
st.write("""As seen above, the best regressor is Decision Tree Regressor. So, we will use this regressor to predict the yield of the crops.

Using the same, the plot is made here using MatPlotLib showing the relation between the predicted and the actual score.""")
clf = DecisionTreeRegressor()
model = clf.fit(train_data, train_labels)

test_df["yield_predicted"] = model.predict(test_data)
test_df["yield_actual"] = pd.DataFrame(test_labels)["hg/ha_yield"].tolist()
test_group = test_df.groupby("Item")

fig, ax = plt.subplots()

ax.scatter(test_df["yield_actual"], test_df["yield_predicted"], edgecolors=(0, 0, 0))

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
plt.show()
st.write(fig)


