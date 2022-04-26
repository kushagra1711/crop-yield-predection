from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
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
# streamlit run webapp.py


st.header("""CROP YIELD PREDICTION""");

st.markdown("[Jump To Results](#prediction-on-custom-inputs)",unsafe_allow_html=True)

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

# st.write(df_yield.info())


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

# st.write(df_rain.info())


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

# st.write(df_pes.info())

#merging the two dataframes
yield_df = pd.merge(yield_df, df_pes, on=['Year','Area'])


###avg temp dataset
avg_temp=  pd.read_csv('temp.csv')

avg_temp = avg_temp.rename(index=str, columns={"year": "Year", "country":'Area'})

#set a header
st.subheader("Temperature Information:")

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
yield_df.to_csv()

st.header("The final Dataframe")

st.dataframe(yield_df)

st.subheader("Information regarfing the dataframe:")

#showing the dataset of the data
st.write(yield_df.describe())

# st.write(yield_df.info())



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

features = yield_df.loc[:, yield_df.columns != 'hg/ha_yield']
features = features.drop(['Year'], axis=1)
label = yield_df['hg/ha_yield']
ct1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
features = np.array(ct1.fit_transform(features))
le = LabelEncoder()
features[:,10] = le.fit_transform(features[:,10])
yield_df_onehot = pd.DataFrame(features)
yield_df_onehot["hg/ha_yield"] = label

st.subheader("One-Hot Encoded data:")
st.dataframe(yield_df_onehot.head())
st.write(yield_df_onehot.info())


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

st.write("""Here we test 3 different regressor algorithms namely: 
1. Gradient Boosting Regressor 
2. Random Forest Regressor
3. Decision Tree Regressor
""")



# not printing all of this as it takes too much time to run, rather just pasting the output as obtained on the kaggle notebook

# models = [
#     GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0),
#     RandomForestRegressor(n_estimators=200, max_depth=3, random_state=0),
#     svm.SVR(),
#     DecisionTreeRegressor()
# ]
# model_train=list(map(compare_models,models)) 

# print(*model_train, sep="\n") The evaluation metric is set based on R^2 (coefficient of determination) regression score function, that will represents the proportion of the variance for items (crops) in the regression model. R^2 score shows how well terms (data points) fit a curve or line.

st.write("""R ^ 2 is a statistical measure between 0 and 1 which calculates how similar a regression line is to the data it's fitted to. If it's a 1, the model 100 % predicts the data variance if it's a 0, the model predicts none of the variance.""")

st.write("")

st.subheader("Testing of the model on test dataset on kaggle notebook")


st.write("""The output below shows the accuracy of the model, the accuracy is the percentage of the correct predictions.

['GradientBoostingRegressor', 0.8959545600619471]

['RandomForestRegressor', 0.6807690552605921]

['DecisionTreeRegressor', 0.9605155680634376]


Therefore, from the above output, we can see that the Decision Tree Regressor is the best model to use for this dataset.""")



test_df=pd.DataFrame(test_data,columns=yield_df_onehot.loc[:, yield_df_onehot.columns != 'hg/ha_yield'].columns) 


st.write("The final dataset is ready to be used for the model. The final dataset contains the following columns:")
st.write(test_df.head())


#prediction

st.header("Prediction")
st.write("""As seen above, the best regressor is Decision Tree Regressor. So, we will use this regressor to predict the yield of the crops.

Using the same, the plot is made here using MatPlotLib showing the relation between the predicted and the actual yield.""")
st.success("The model predicts the results with an accuracy of  96.051%")
clf = DecisionTreeRegressor()
model = clf.fit(train_data, train_labels)

test_df["yield_predicted"] = model.predict(test_data)
test_df["yield_actual"] = pd.DataFrame(test_labels)["hg/ha_yield"].tolist()

fig, ax = plt.subplots()

ax.scatter(test_df["yield_actual"], test_df["yield_predicted"], edgecolors=(0, 0, 0))

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
plt.show()
st.write(fig)


st.header("Prediction on Custom Inputs")

data = pd.read_csv("yield_df.csv")
data.columns = ["","Area","Item","Year","hg/ha_yield","average_rain_fall_mm_per_year","pesticides_tonnes","avg_temp"]
rawlist = list(data.Area)
rawlist2 = list(data.Item)
#get unique elements
area_list = []
item_list = []
for x in rawlist2:
    if x not in item_list:
        item_list.append(x)
for x in rawlist:
    if x not in area_list:
        area_list.append(x)


def getUserInput():
    
    #Area,Item,Year,hg/ha_yield,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp
    area_ip=st.sidebar.selectbox("Enter The name of the Country",area_list)
    item_ip = st.sidebar.selectbox("Enter The name of the Crop", item_list)    
    rainfall_ip=st.sidebar.slider('Rainfall',55,3200,1100)
    pesticides_ip=st.sidebar.slider('Pesticides',0,1807000,20000)
    avg_temp_ip=st.sidebar.slider('Temprature',-14,45,16)

    userData=np.array([[area_ip,item_ip,rainfall_ip,pesticides_ip,avg_temp_ip]])
    return userData

inputs=getUserInput()
inputs=np.array(ct1.transform(inputs))
inputs[:,10]=le.transform(inputs[:,10])
inputs=scaler.transform(inputs)
prediction = model.predict(inputs)

predic_y = "Predicted hg/ha_yield is: " + str(prediction[0])
st.success(predic_y)
