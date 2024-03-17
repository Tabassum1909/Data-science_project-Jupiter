#!/usr/bin/env python
# coding: utf-8

# # CAPSTONE PROJECT-3 :- PRCP-1027- SKIN DISORDER CLASSIFICATION
# 
# # PROJECT TEAM ID :- PTID-CDS-FEB-24-1805
# 

# ## Dataset Information:
# 
# * This database contains 34 attributes, 33 of which are linear valued and one of them is nominal.
# 
# 
# * The differential diagnosis of erythemato-squamous diseases is a real problem in dermatology. They all share the clinical     features of erythema and scaling, with very little differences.
# 
# 
# * The diseases in this group are **psoriasis**, **seboreic dermatitis**, **lichen planus**, **pityriasis rosea**, **cronic dermatitis**, and **pityriasis rubra pilaris**.
# 
# 
# * Usually a biopsy is necessary for the diagnosis but unfortunately these diseases share many histopathological features as well. (Histopathology:The study of diseased cells and tissues using a microscope.)
# 
# 
# * Another difficulty for the differential diagnosis is that a disease may show the features of another disease at the beginning stage and may have the characteristic features at the following stages.**Patients were first evaluated clinically with 12 features. Afterwards, skin samples were taken for the evaluation of 22 histopathological features.**
# 
# 
# * The values of the histopathological features are determined by an analysis of the samples under a microscope.
# 
# 
# * In the dataset constructed for this domain, the family history feature has the value 1 if any of these diseases has been observed in the family, and 0 otherwise.
# 
# 
# * The age feature simply represents the age of the patient.
# 
# 
# * Every other feature (clinical and histopathological) was given a degree in the range of 0 to 3. Here, 0 indicates that the feature was not present, 3 indicates the largest amount possible, and 1, 2 indicate the relative intermediate values.
# 
# 
# * The names and id numbers of the patients were recently removed from the database .
# 

# ## Dataset Information:
# 
# * This database contains 34 attributes, 33 of which are linear valued and one of them is nominal.
# 
# 
# * The differential diagnosis of erythemato-squamous diseases is a real problem in dermatology. They all share the clinical     features of erythema and scaling, with very little differences.
# 
# 
# * The diseases in this group are **psoriasis**, **seboreic dermatitis**, **lichen planus**, **pityriasis rosea**, **cronic dermatitis**, and **pityriasis rubra pilaris**.
# 
# 
# * Usually a biopsy is necessary for the diagnosis but unfortunately these diseases share many histopathological features as well. (Histopathology:The study of diseased cells and tissues using a microscope.)
# 
# 
# * Another difficulty for the differential diagnosis is that a disease may show the features of another disease at the beginning stage and may have the characteristic features at the following stages.**Patients were first evaluated clinically with 12 features. Afterwards, skin samples were taken for the evaluation of 22 histopathological features.**
# 
# 
# * The values of the histopathological features are determined by an analysis of the samples under a microscope.
# 
# 
# * In the dataset constructed for this domain, the family history feature has the value 1 if any of these diseases has been observed in the family, and 0 otherwise.
# 
# 
# * The age feature simply represents the age of the patient.
# 
# 
# * Every other feature (clinical and histopathological) was given a degree in the range of 0 to 3. Here, 0 indicates that the feature was not present, 3 indicates the largest amount possible, and 1, 2 indicate the relative intermediate values.
# 
# 
# * The names and id numbers of the patients were recently removed from the database .
# 

# ## Import Basic Libraries

# In[12]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot  as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import plotly.express as px
from matplotlib import rcParams


# ## Load Data

# In[13]:


Data=pd.read_csv(r"C:\Users\farhe\Downloads\PRCP-1028-Skin-Disorder-Prediction-20220512T101734Z-001\PRCP-1028-Skin-Disorder-Prediction\dataset_35_dermatology (1).csv")


# In[14]:


# To see all the rows
pd.set_option("display.max_rows",None)
Data


# ## Basic Checks

# In[15]:


Data.shape


# In[16]:


Data.head()


# In[17]:


Data.info()


# In[18]:


Data.dtypes


# ## Attribute Information :
# 
# ### Clinical Attributes:
# 
# 1. **Erythema:** The presence or absence of redness or inflammation of the skin.
# 
# 2. **Scaling:** Whether there are visible scales or flakes on the skin's surface.
# 
# 3. **Definite Borders:** Describes whether the skin lesion has well-defined or clear borders.
# 
# 4. **Itching:** Indicates whether the patient experiences itching or pruritus associated with the skin condition.
# 
# 5. **Koebner Phenomenon:** Refers to whether new skin lesions appear at sites of trauma or injury to the skin.
# 
# 6. **Polygonal Papules:** Presence or absence of polygonal-shaped raised skin lesions.
# 
# 7. **Follicular Papules:** Whether there are raised lesions involving hair follicles.
# 
# 8. **Oral Mucosal Involvement:** Indicates if the condition affects the mucous membranes inside the mouth.
# 
# 9. **Knee and Elbow Involvement:** Describes whether the skin condition is specifically located on the knees and elbows.
# 
# 10. **Scalp Involvement:** Whether the condition affects the scalp.
# 
# 11. **Family History(0 or 1):** Indicates whether there is a family history of similar skin conditions or relevant genetic factors.
# 
# 12. **Age:** The age of the patient at the time of data collection.
# 
# _These clinical attributes are important for dermatologists to assess and diagnose various skin conditions accurately. They are used in clinical examinations and research to characterize the presentation and progression of dermatological diseases. Researchers and healthcare providers can analyze these attributes to understand patterns and correlations in different skin conditions and to develop treatment plans tailored to individual patients._
# 
# 
# ### Histopathological Attributes:
# 
# 
# 1. **Melanin Incontinence:** Melanin incontinence refers to the presence of melanin pigment within the dermis. It can occur when melanocytes are damaged or when melanin leaks from epidermal cells into the deeper skin layers. Melanin incontinence is often associated with conditions like melanoma and lichen planus.
# 
# 2. **Eosinophils in the Infiltrate:** Eosinophils are a type of white blood cell involved in allergic and inflammatory responses. The presence of eosinophils in the inflammatory infiltrate can be indicative of certain allergic or eosinophilic skin conditions.
# 
# 3. **PNL Infiltrate:** PNL (polymorphonuclear leukocyte) infiltrate refers to the presence of polymorphonuclear white blood cells in the skin tissue. PNLs are involved in the early stages of inflammation and can be seen in various inflammatory skin disorders.
# 
# 4. **Fibrosis of the Papillary Dermis:** Fibrosis in the papillary dermis indicates the accumulation of excessive fibrous tissue in the uppermost layer of the dermis. It can result from chronic inflammation and is often associated with scarring.
# 
# 5. **Exocytosis:** Exocytosis refers to the migration of inflammatory cells (usually lymphocytes) from the bloodstream into the epidermis. This phenomenon is commonly observed in conditions like psoriasis.
# 
# 6. **Acanthosis:** Acanthosis is characterized by the thickening of the epidermis, particularly the stratum spinosum layer. It is a common histological finding in conditions like acanthosis nigricans and psoriasis.
# 
# 7. **Hyperkeratosis:** Hyperkeratosis is the excessive thickening of the stratum corneum, the outermost layer of the epidermis. It leads to the formation of a thickened, keratinized surface and can be seen in various skin disorders.
# 
# 8. **Parakeratosis:** Parakeratosis is a histological feature where nuclei are retained in the stratum corneum. It is often seen in psoriasis and other hyperproliferative skin conditions.
# 
# 9. **Clubbing of the Rete Ridges:** Clubbing of the rete ridges refers to the bulbous enlargement of the rete ridges, which are the finger-like projections of the epidermis into the dermis. This can be seen in conditions like lichen planus.
# 
# 10. **Elongation of the Rete Ridges:** Elongation of the rete ridges indicates the lengthening of these epidermal projections into the dermis. It is a common finding in many skin diseases.
# 
# 11. **Thinning of the Suprapapillary Epidermis:** This feature involves a reduction in the thickness of the suprapapillary epidermis, which is the epidermal layer above the dermal papillae. It can be observed in some inflammatory skin conditions.
# 
# 12. **Spongiform Pustule:** A spongiform pustule is a blister-like structure filled with neutrophils and located within the epidermis. It is characteristic of pustular psoriasis.
# 
# 13. **Munro Microabscess:** Munro microabscesses are small collections of neutrophils in the stratum corneum. They are commonly found in the epidermis of individuals with psoriasis.
# 
# 14. **Focal Hypergranulosis:** Focal hypergranulosis refers to localized thickening of the granular layer of the epidermis. This can be seen in various skin disorders.
# 
# 15. **Disappearance of the Granular Layer:** In some skin conditions, the granular layer of the epidermis may be absent or significantly reduced.
# 
# 16. **Vacuolization and Damage of Basal Layer:** Vacuolization refers to the formation of empty spaces (vacuoles) within the basal layer of the epidermis. It is often associated with autoimmune blistering disorders.
# 
# 17. **Spongiosis:** Spongiosis is the presence of intercellular edema (fluid accumulation between epidermal cells) in the epidermis. It is a common feature of eczematous conditions.
# 
# 18. **Saw-tooth Appearance of Rete Ridges:** The saw-tooth appearance is characterized by irregular, jagged projections of the epidermal rete ridges and is often seen in lichen planus.
# 
# 19. **Follicular Horn Plug:** A follicular horn plug is a collection of keratinous material within a hair follicle. It can be seen in various conditions, including acne.
# 
# 20. **Perifollicular Parakeratosis:** Perifollicular parakeratosis is the presence of parakeratosis around hair follicles. It can be observed in certain inflammatory skin conditions.
# 
# 22. **Inflammatory Mononuclear Infiltrate:** This refers to the presence of mononuclear white blood cells (such as lymphocytes) in the dermal or epidermal infiltrate, indicating chronic inflammation.
# 
# 23. **Band-like Infiltrate:** A band-like infiltrate is characterized by a dense, linear accumulation of inflammatory cells within the skin tissue. It can be seen in conditions like lichen planus.
# 
# 
# _These histopathological attributes provide crucial information for dermatopathologists and researchers to diagnose and classify various skin disorders accurately. They aid in understanding the underlying histological changes associated with different dermatological conditions._
# 
# 
# ### Class of Diseases
# The diseases in this group are
# 
# 1. **Psoriasis(1)**,
# 2. **Seboreic Dermatitis(2)**,
# 3. **Lichen Planus(3)**,
# 4. **Pityriasis Rosea(4)**http://localhost:8889/notebooks/reference%20projects/skin%20disorder.ipynb#Data-Manipulation-and-Cleaning,
# 5. **Cronic Dermatitis(5)**,
# 6. **Pityriasis Rubra Pilaris(6)**.

# ## Data Manipulation and Cleaning

# In[19]:


num_col=Data.select_dtypes(include=["int64","float64"]).columns
num_col


# In[20]:


cat_col=Data.select_dtypes(include=["object"])
cat_col.columns


# In[21]:


# find unique value in all numerical columns
for i in num_col:
    print(i,Data[i].unique())
    print(Data[i].value_counts())


# In[22]:


# find unique value in all categorical columns
for i in cat_col:
    print(i,Data[i].unique())
    print(Data[i].value_counts())
    print("**********************")


# In[23]:


Data['class'].value_counts()


# In[24]:


Data.isnull().sum()


# In[25]:


Data.describe()


# In[26]:


Data.describe(include='O')


# In[27]:


for column in Data.columns:
    Data[column]=pd.to_numeric(Data[column], errors='coerce')


# In[28]:


Data


# In[29]:


Data['Age'] = Data['Age'].astype(pd.Int64Dtype())


# In[30]:


Data.dtypes


# In[31]:


Data['class'].replace([1,2,3,4,5,6],['Psoriasis', 'Seboreic_Dermatitis', 'Lichen_Planus', 'Pityriasis_Rosea','Cronic_Dermatitis','Pityriasis_rubra_pilaris'], inplace=True)


# In[32]:


# Replacing NaN and 0 age value using median.
median = Data['Age'].median()
Data['Age'].fillna(median, inplace=True)
Data['Age'] = Data['Age'].replace(0,Data['Age'].median())


# In[33]:


Data


# In[34]:


new_data = Data.copy()


# In[35]:


new_data.describe()


# ## Exploratory Data Analysis :

# As per the information provided, Patients were first evaluated clinically with 12 features. Afterwards, skin samples were taken for the evaluation of 22 histopathological features.
# 
# - **Clinical Features:** Erythema, Scaling, Definite_borders, Itching, Koebner_phenomenon, Polygonal_papules, Follicular_papules, Oral_mucosal_involvement, Knee_and_elbow_involvement, Scalp_involvement, Family_history, and Age.
# 
# - **Histopathological features:** Melanin_incontinence, Eosinophils_in_the_infiltrate, PNL_infiltrate, Fibrosis_of_the_papillary_dermis, Exocytosis, Acanthosis, Hyperkeratosis, Parakeratosis, Clubbing_of_the_rete_ridges, Elongation_of_the_rete_ridges, Thinning_of_the_suprapapillary_epidermis, Spongiform_pustule, Munro_microabcess, Focal_hypergranulosis, Disappearance_of_the_granular_layer, Vacuolisation_and_damage_of_basal_layer, Spongiosis, Saw-tooth_appearance_of_retes, Follicular_horn_plug, Perifollicular_parakeratosis, Inflammatory_monoluclear_inflitrate, Band-like_infiltrate.

# ## Data Visualization
# 
# ### Insights:
# In this section, we utilize data visualization techniques to visually explore and communicate insights from the Dermatology Dataset. By creating meaningful and informative visual representations such as plots, charts, and graphs, we aim to uncover patterns, relationships, and trends within the data. Data visualization enhances our understanding of the dataset, making it easier to convey findings and key messages to stakeholders and decision-makers. Through effective visualizations, we can highlight important patterns and correlations, enabling better-informed decision-making and actionable insights.

# In[36]:


clinical_features=['erythema', 'scaling', 'definite_borders', 'itching',
       'koebner_phenomenon', 'polygonal_papules', 'follicular_papules',
       'oral_mucosal_involvement', 'knee_and_elbow_involvement',
       'scalp_involvement', 'family_history']


# In[37]:


# histplot
plt.figure(figsize=(12,16), facecolor='white')
plotnumber = 1

for clinical_features in Data:
    if plotnumber<=11 :
        ax = plt.subplot(4,3,plotnumber)
        sns.histplot(x=new_data[clinical_features],kde=True)
        plt.xlabel(clinical_features,fontsize=15)
        plt.ylabel('count',fontsize=15)
    plotnumber+=1
plt.tight_layout()


# Insights:
# The histogram allows us to visualize the distribution of a numerical variable, such as the 'age' column in your DataFrame. By plotting a histogram, we can understand the frequency and range of ages present in our dataset. This can help identify patterns or anomalies in the age distribution, such as whether it is skewed, normally distributed, or has any significant peaks or gaps.

# ## Count and Distribution of Histopathological Features

# In[38]:


histopath_features = ['melanin_incontinence',
       'eosinophils_in_the_infiltrate', 'PNL_infiltrate',
       'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis',
       'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges',
       'elongation_of_the_rete_ridges',
       'thinning_of_the_suprapapillary_epidermis', 'spongiform_pustule',
       'munro_microabcess', 'focal_hypergranulosis',
       'disappearance_of_the_granular_layer',
       'vacuolisation_and_damage_of_basal_layer', 'spongiosis',
       'saw-tooth_appearance_of_retes', 'follicular_horn_plug',
       'perifollicular_parakeratosis', 'inflammatory_monoluclear_inflitrate',
       'band-like_infiltrate']


# In[39]:


# histplot
plt.figure(figsize=(12,28), facecolor='white')
plotnumber = 1

for histopath_features in Data:
    if plotnumber<=22 :
        ax = plt.subplot(8,3,plotnumber)
        sns.histplot(x=new_data[histopath_features],kde=True)
        plt.xlabel(histopath_features,fontsize=15)
        plt.ylabel('count',fontsize=15)
    plotnumber+=1
plt.tight_layout()
plt.show()


# In[ ]:





# In[40]:


plt.figure(figsize=(12,6))
ax=sns.countplot(x=Data['class'])
for label in ax.containers:
    ax.bar_label(label)
plt.tight_layout()


# ### Insights:
# The bar plot is useful for visualizing the frequency or count of different categories in a categorical variable, such as the 'class' column in our DataFrame. By plotting a count plot, we can compare the number of instances for each class and gain insights into the class distribution. This visualization can reveal class imbalances, identify dominant or minority classes, or provide an overview of the distribution of the target variable.

# In[41]:


plt.figure(figsize=(16,6))
new_data['class'].value_counts().plot(kind='pie', autopct='%0.02f%%', radius=1.0, pctdistance=0.6, colors= ['#89519c','#FC9532','#964B00','#3a69b5', '#32a82c','#c4182c'] ,explode = [0.025, 0.025, 0.025, 0.025, 0.025, 0.025])


# ## Age wise Distribution of classes of Skin diseases.

# In[42]:


plt.figure(figsize=(15,7))
sns.swarmplot(y="Age", x="class", data=new_data, hue='family_history')
plt.title('Relationship between Age and Class')
plt.show()


# 
# df=AV.AutoViz(filename='skin disorder.csv',ver

# In[43]:


get_ipython().run_line_magic('pip', 'install autoviz')
from autoviz.AutoViz_Class import AutoViz_Class
AV=AutoViz_Class()

df=AV.AutoViz(filename='skin disorder.csv',verbose=2,chart_format='html')


# In[44]:


## Univariate Analysis
get_ipython().system('pip install sweetviz')


# In[45]:


import sweetviz as sv#importing sweetviz library
my_report = sv.analyze(Data)#syntax to use sweetviz
my_report.show_html()#Default arguments will generate to "SWEETVIZ_REPORT.html"


# # Data Preprocessing

# ## Checking for Missing Values

# In[46]:


new_data.isnull().sum()


# ## Checking for duplicates

# In[47]:


new_data.duplicated().sum()


# In[48]:


new_data


# In[49]:


new_data['class']=new_data['class'].map({'Psoriasis':0,'Seboreic_Dermatitis':1,'Lichen_Planus':2,'Pityriasis_Rosea':3,'Cronic_Dermatitis':4,'Pityriasis_rubra_pilaris':5})


# In[50]:


new_data


# ## Understanding Data using t-SNE
# 
# **t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique used to visualize high-dimensional data in a lower-dimensional space. In this section, we apply t-SNE to gain a better understanding of the data's structure and identify potential clusters or patterns that may exist in the dataset.**

# In[51]:


x = new_data.iloc[:, :-1].values
y = new_data.iloc[:, -1].values


# In[52]:


Data['class'].value_counts()


# In[53]:


from sklearn.manifold import TSNE

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=110, verbose=2)
x_tsne = tsne.fit_transform(x)


# Create a scatterplot of the t-SNE graph
plt.figure(figsize=(8, 8))
sns.scatterplot(
    x=x_tsne[:,0], y=x_tsne[:,1],
    hue=y, palette=sns.color_palette("hls", len(np.unique(y))),
    alpha=0.8, edgecolor='none'
)


plt.title('t-SNE Plot for Dermatology Dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Visualization of Data with perplexity = 30')
plt.legend(loc='upper right')
plt.show()


# In[54]:


# Apply t-SNE with higher perplexity
tsne = TSNE(n_components=2, random_state=110, verbose=2, perplexity = 100)
x_tsne = tsne.fit_transform(x)


# Create a scatterplot of the t-SNE graph
plt.figure(figsize=(8, 8))
sns.scatterplot(
    x=x_tsne[:,0], y=x_tsne[:,1],
    hue=y, palette=sns.color_palette("hls", len(np.unique(y))),
    alpha=1, edgecolor='none'
)


plt.title('t-SNE Plot for Dermatology Dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Visualization of Data with perplexity = 100')
plt.legend(loc='upper right')
plt.show()


# In[55]:


# Apply t-SNE with more high perplexity
tsne = TSNE(n_components=2, random_state=110, verbose=2, perplexity = 200)
x_tsne = tsne.fit_transform(x)


# Create a scatterplot of the t-SNE graph
plt.figure(figsize=(8, 8))
sns.scatterplot(
    x=x_tsne[:,0], y=x_tsne[:,1],
    hue=y, palette=sns.color_palette("hls", len(np.unique(y))),
    alpha=1, edgecolor='none'
)


# plt.title('t-SNE Plot for Dermatology Dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Visualization of Data with perplexity = 200')
plt.legend(loc='upper right')
plt.show()


# * Upon analyzing the dataset, we can observe that the class features **'0', '2', and '5'** appear to be well separated and distinguishable. However, the class features **'1', '3', and '4'** exhibit overlapping patterns, making it challenging to separate them effectively. This overlapping nature may pose difficulties for our model, as it may struggle to accurately classify instances belonging to these classes. We can further examine this observation by reviewing the confusion matrix and classification report during the model evaluation process.

# In[56]:


# Calculate the correlation matrix
corr_matrix = new_data.corr()

corr_matrix


# # Plot the correlation heatmap
# plt.figure(figsize=(40, 40))
# sns.heatmap(corr_matrix, annot=True,annot_kws={"size":2})
# plt.title('Correlation Heatmap')
# plt.show()

# ## split data into x and y

# In[57]:


x=new_data.drop(columns=["class"])
y=new_data['class']


# In[58]:


new_data


# ## split data for training and testing

# In[59]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.3, random_state=42)


# In[60]:


y_train.value_counts()


# In[61]:


x.shape, y.shape


# In[62]:


print("The shape of x_train is:", x_train.shape)
print("The shape of x_test is:", x_test.shape)
print("The shape of Y_train is:", y_train.shape)
print("The shape of Y_test is:", y_test.shape)


# # Model Building

# In[63]:


# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# ## 1. Logistic Regression

# In[64]:


Log_Reg = LogisticRegression()
Log_Reg.fit(x_train, y_train)


# In[65]:


# MODEL EVALUATION
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
y_train_pred_Log_Reg = Log_Reg.predict(x_train) # training model
y_test_pred_Log_Reg = Log_Reg.predict(x_test) # testing model


# In[66]:


#Evaluate Logistic Regression model:
print("Logistic Regression training set score:" , accuracy_score(y_train, y_train_pred_Log_Reg))
print("Logistic Regression test set score:" , accuracy_score(y_test, y_test_pred_Log_Reg))


# In[67]:


# Classification Report of Tuned Logistic Regression Model
print(classification_report(y_test, y_test_pred_Log_Reg, target_names=
                            ['psoriasis','seboreic_dermatitis','lichen_planus','pityriasis_rosea','cronic_dermatitis',
                             'pityriasis_rubra_pilaris']))


# In[68]:


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_Log_Reg, xticks_rotation='vertical')


# ## 2. K- Nearest Neighbors Classifier

# In[69]:


from sklearn.neighbors import KNeighborsRegressor # KNN


# In[70]:


KNN_CLF = KNeighborsClassifier()
KNN_CLF.fit(x_train,y_train)


# In[71]:


KNN_CLF.get_params()


# In[72]:


# Defining Parameters' ranges for Tuning
params_KNN_CLF= {'n_neighbors':[2,3,4,5,6,7,8,9,10],
         'weights':["uniform", "distance"],
         'metric':['minkowski', 'chebyshev','euclidean','manhattan'],
         'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
         'leaf_size': [20,30,40,50,60,70,80,90,100],
         'p': [1,2,3,4]}


# In[73]:


# %%time
from sklearn.model_selection import  GridSearchCV
grid_search_KNN_CLF= KNeighborsClassifier()
grid_search_KNN_CLF = GridSearchCV(grid_search_KNN_CLF, params_KNN_CLF,cv=5,scoring='accuracy',n_jobs=-1)

grid_result_KNN_CLF= grid_search_KNN_CLF.fit(x_train, y_train)
print('Best Params: ', grid_result_KNN_CLF.best_params_)


# In[74]:


KNN_CLF_Tuned = KNeighborsClassifier(algorithm='ball_tree', leaf_size= 20, metric= 'minkowski', n_neighbors= 10, p=1, weights= 'distance')
KNN_CLF_Tuned.fit(x_train, y_train)


# In[75]:


y_train_pred_KNN_CLF_Tuned= KNN_CLF_Tuned.predict(x_train) # training model
y_test_pred_KNN_CLF_Tuned = KNN_CLF_Tuned.predict(x_test) # testing model


# In[76]:


#Evaluate Tuned KNN Classifier model:
print("KNN Tuned training set score:", accuracy_score(y_train, y_train_pred_KNN_CLF_Tuned))
print("KNN Tuned test set score:", accuracy_score(y_test, y_test_pred_KNN_CLF_Tuned))


# In[77]:


# Classification Report of Tuned KNN Classifier Model
print(classification_report(y_test, y_test_pred_KNN_CLF_Tuned, target_names=
                            ['psoriasis','seboreic_dermatitis','lichen_planus','pityriasis_rosea','cronic_dermatitis',
                             'pityriasis_rubra_pilaris']))


# In[78]:


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_KNN_CLF_Tuned, xticks_rotation='vertical')


# 
# ## 3. Support Vector Classifier

# In[79]:


SVC = SVC(class_weight='balanced')
SVC.fit(x_train, y_train)


# In[80]:


y_train_pred_SVC=SVC.predict(x_train)
y_test_pred_SVC=SVC.predict(x_test)


# In[81]:


#Evaluate SVC model:
print("SVC training set score:", accuracy_score(y_train, y_train_pred_SVC))
print("SVC test set score:", accuracy_score(y_test, y_test_pred_SVC))


# In[82]:


#Classification Report of Support Vector Classifier Model
print(classification_report(y_test, y_test_pred_SVC, target_names=
                           ['psoriasis','seboreic_dermatitis','lichen_planus','pityriasis_rosea','cronic_dermatitis',
                            'pityriasis_rubra_pilaris']))


# In[83]:


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_SVC, xticks_rotation='vertical')


# In[84]:


SVC.get_params()


# In[85]:


# Defining Parameters' ranges for Tuning
params_SVM_CLF= {'C':[0.1,5,10,50,60,70],'gamma':[1,0.1,0.01,0.001,0.001],
            'random_state':list(range(1,20)),'kernel':['rbf','poly','sigmoid','linear']}


# In[86]:


# %%time
from sklearn.svm import SVC
from sklearn.model_selection import  GridSearchCV
grid_search_SVM_CLF= SVC()
grid_search_SVM_CLF = GridSearchCV(grid_search_SVM_CLF, params_SVM_CLF,cv=5,scoring='accuracy',n_jobs=-1)

grid_result_SVM_CLF= grid_search_SVM_CLF.fit(x_train, y_train)
print('Best Params: ', grid_result_SVM_CLF.best_params_)


# In[87]:


SVM_CLF_Tuned = SVC(C=5, gamma=1, kernel= 'linear', random_state=1)
SVM_CLF_Tuned.fit(x_train, y_train)


# In[88]:


y_train_pred_SVM_CLF_Tuned= SVM_CLF_Tuned.predict(x_train) # training model
y_test_pred_SVM_CLF_Tuned = SVM_CLF_Tuned.predict(x_test) # testing model


# In[89]:


#Evaluate Tuned KNN Classifier model:
print("SVC Tuned training set score:", accuracy_score(y_train, y_train_pred_SVM_CLF_Tuned))
print("SVC Tuned test set score:", accuracy_score(y_test, y_test_pred_SVM_CLF_Tuned))


# In[90]:


# Classification Report of Tuned KNN Classifier Model
print(classification_report(y_test, y_test_pred_SVM_CLF_Tuned, target_names=
                            ['psoriasis','seboreic_dermatitis','lichen_planus','pityriasis_rosea','cronic_dermatitis',
                             'pityriasis_rubra_pilaris']))


# In[91]:


y_test_pred_SVM_CLF_Tuned = SVM_CLF_Tuned.predict(x_test)


# In[92]:


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_SVM_CLF_Tuned, xticks_rotation='vertical')


# # 4.Decision Tree Classifier 

# In[93]:


DTC = DecisionTreeClassifier()
DTC.fit(x_train, y_train)


# In[94]:


y_train_pred_DTC = DTC.predict(x_train)
y_test_pred_DTC = DTC.predict(x_test)


# In[95]:


#Evaluate DTC model:
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
print("DTC training set score:", accuracy_score(y_train, y_train_pred_DTC))
print("DTC test set score:",  accuracy_score(y_test, y_test_pred_DTC))


# In[96]:


# Classification Report of DTC Classfier Model
print(classification_report(y_test, y_test_pred_DTC, target_names=
                            ['psoriasis','seboreic_dermatitis','lichen_planus','pityriasis_rosea','cronic_dermatitis',
                             'pityriasis_rubra_pilaris']))


# In[97]:


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_DTC, xticks_rotation='vertical')


# # 5. Random Forest Classifier

# In[98]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
RFC = RandomForestClassifier(random_state=42)
RFC.fit(x_train, y_train)


# In[99]:


y_train_pred_RFC=RFC.predict(x_train)
y_test_pred_RFC=RFC.predict(x_test)


# In[100]:


#Evaluate RF Classifier model:

print("RFC Classifier training set score:", accuracy_score(y_train,y_train_pred_RFC))
print("RFC Classifier test set score:", accuracy_score(y_test, y_test_pred_RFC))


# In[101]:


# Classification Report of RF Classifier Model
print(classification_report(y_test, y_test_pred_RFC, target_names=
                            ['psoriasis','seboreic_dermatitis','lichen_planus','pityriasis_rosea','cronic_dermatitis',
                             'pityriasis_rubra_pilaris']))


# In[102]:


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_RFC, xticks_rotation='vertical')


# ## 6. Multinomial Naive Bayes Classifier

# In[103]:


MNBC = MultinomialNB()
MNBC.fit(x_train, y_train)


# In[104]:


y_train_pred_MNBC = MNBC.predict(x_train) # training model
y_test_pred_MNBC = MNBC.predict(x_test)


# In[105]:


#Evaluate Multinomial Naive Bayes model:
print("Multinomial Naive Bayes training set score:", accuracy_score(y_train, y_train_pred_MNBC))
print("Multinomial Naive Bayes test set score:", accuracy_score(y_test, y_test_pred_MNBC))


# In[106]:


# Classification Report of Multinomial Naive Bayes Model
print(classification_report(y_test, y_test_pred_MNBC, target_names=
                            ['psoriasis','seboreic_dermatitis','lichen_planus','pityriasis_rosea','cronic_dermatitis',
                             'pityriasis_rubra_pilaris']))


# In[107]:


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_MNBC, xticks_rotation='vertical')


# # 7. Gradient Boosting Classifier
# 

# In[108]:


GBC = GradientBoostingClassifier(random_state=42)
GBC.fit(x_train, y_train)


# In[109]:


y_train_pred_GBC = GBC.predict(x_train)
y_test_pred_GBC= GBC.predict(x_test)


# In[110]:


#Evaluate GBC model:
print("GBC training set score:", accuracy_score(y_train,y_train_pred_GBC))
print("GBC test set score:", accuracy_score(y_test, y_test_pred_GBC))


# In[111]:


# Classification Report of GBC Model
print(classification_report(y_test, y_test_pred_GBC, target_names=
                            ['psoriasis','seboreic_dermatitis','lichen_planus','pityriasis_rosea','cronic_dermatitis',
                             'pityriasis_rubra_pilaris']))


# In[112]:


ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_GBC, xticks_rotation='vertical')


# # 8 XG Boost Classifier

# In[113]:


XGB_CLF = XGBClassifier()
XGB_CLF.fit(x_train, y_train)


# In[114]:


y_train_pred_XGB_CLF =XGB_CLF.predict(x_train)
y_test_pred_XGB_CLF=XGB_CLF.predict(x_test)


# In[115]:


#Evaluate XGB Classifier model:
print("XGB_CLF training set score:", accuracy_score(y_train,y_train_pred_XGB_CLF))
print("XGB_CLF test set score:", accuracy_score(y_test, y_test_pred_XGB_CLF))


# In[116]:


# Classification Report of XGB Classifier Model
print(classification_report(y_test, y_test_pred_XGB_CLF, target_names=
                            ['psoriasis','seboreic_dermatitis','lichen_planus','pityriasis_rosea','cronic_dermatitis',
                             'pityriasis_rubra_pilaris']))


# In[117]:


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_XGB_CLF, xticks_rotation='vertical')


# # 9.Artificial Neural Networks Classifier

# In[118]:


# model creation
from sklearn.neural_network import MLPClassifier # MLP stands for Multi Layer Perceptron
ANN_CLF = MLPClassifier()
ANN_CLF.fit(x_train,y_train)


# In[119]:


y_train_pred_ANN_CLF= ANN_CLF.predict(x_train)
y_test_pred_ANN_CLF= ANN_CLF.predict(x_test)


# In[120]:


# Evaluate ANN Classifier model:

print("ANN training set score:", accuracy_score(y_train, y_train_pred_ANN_CLF))
print("ANN test set score:", accuracy_score(y_test, y_test_pred_ANN_CLF))


# In[121]:


# Classification Report of ANN Classifier Model
print(classification_report(y_test, y_test_pred_ANN_CLF, target_names=
                            ['psoriasis','seboreic_dermatitis','lichen_planus','pityriasis_rosea','cronic_dermatitis',
                             'pityriasis_rubra_pilaris']))


# In[122]:


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_ANN_CLF, xticks_rotation='vertical')


# In[123]:


Model_Comparison= pd.DataFrame({'Model':['Logistic Regressor ()',
                                         'KNN Classifier (Tuned)',
                                         'Support Vector Classifier(Tuned) ',
                                         'Decision Tree Classifier()',
                                         'Random Forest Classifier',
                                         'Multinomial Naive Bayes Classifier()',
                                         'Gradient Boosting Classifier()',
                                         'XG Boosting Classifier()',
                                         'ANN Classifier()'],

                                'Train Score':[accuracy_score(y_train, y_train_pred_Log_Reg),
                                               accuracy_score(y_train, y_train_pred_KNN_CLF_Tuned),
                                               accuracy_score(y_train, y_train_pred_SVM_CLF_Tuned),
                                               accuracy_score(y_train, y_train_pred_DTC),
                                               accuracy_score(y_train, y_train_pred_RFC),
                                               accuracy_score(y_train, y_train_pred_MNBC),
                                               accuracy_score(y_train,y_train_pred_GBC),
                                               accuracy_score(y_train,y_train_pred_XGB_CLF),
                                               accuracy_score(y_train, y_train_pred_ANN_CLF)],
                                'Test Score':[accuracy_score(y_test, y_test_pred_Log_Reg),
                                              accuracy_score(y_test, y_test_pred_KNN_CLF_Tuned),
                                              accuracy_score(y_test, y_test_pred_SVM_CLF_Tuned),
                                              accuracy_score(y_test, y_test_pred_DTC),
                                              accuracy_score(y_test, y_test_pred_RFC),
                                              accuracy_score(y_test, y_test_pred_MNBC),
                                              accuracy_score(y_test, y_test_pred_GBC),
                                              accuracy_score(y_test, y_test_pred_XGB_CLF),
                                              accuracy_score(y_test, y_test_pred_ANN_CLF)]})
Model_Comparison.index+=1
Model_Comparison


# In[124]:


import matplotlib.pyplot as plt

# Get the data from the dataframe
model_names = Model_Comparison['Model'].tolist()
train_scores = Model_Comparison['Train Score'].tolist()
test_scores = Model_Comparison['Test Score'].tolist()
plt.figure(figsize=(12,6))
# Plot the data
plt.plot(model_names, train_scores, label='Train Score', color='#0590DA')
plt.plot(model_names, test_scores, label='Test Score', color='#FF6600')
plt.legend()
plt.xticks(rotation=90)
plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim(0.900, 1.05)
plt.title('Comparison between Train and Test Scores of each Classification Model')
plt.show()


# ## SUMMARY
# 
# * **The differential diagnosis of erythemato-squamous diseases is a real problem in dermatology. They all share the clinical features of erythema and scaling, with very little differences. The diseases in this group are psoriasis, seboreic dermatitis, lichen planus, pityriasis rosea, cronic dermatitis, and pityriasis rubra pilaris. Usually a biopsy is necessary for the diagnosis but unfortunately these diseases share many histopathological features as well. Another difficulty for the differential diagnosis is that a disease may show the features of another disease at the beginning stage and may have the characteristic features at the following stages. Patients were first evaluated clinically with 12 features. Afterwards, skin samples were taken for the evaluation of 22 histopathological features. The values of the histopathological features are determined by an analysis of the samples under a microscope.**
# 
# 
# * **The Primary objective is to build a machine learning techniques, which can effectively differentiate skin disease classification.**

# ## METHODOLOGY FOLLOWED
# 
# 
# 
# * **The dataset contains  12 clinical features and 22 histopathological features, the feature family history has the value 1 if any of these diseases has been observed in the family, and 0 otherwise. The age feature simply represents the age of the patient. Every other feature (clinical and histopathological) was given a degree in the range of 0 to 3. Here, 0 indicates that the feature was not present, 3 indicates the largest amount possible, and 1, 2 indicate the relative intermediate values.**
# 
# 
# * **Exploratory data analysis has been carried out on each features and their relationship with other features.**
# 
# 
# * **The age feature had NAN values and zero values which was handled using the median of the age feature.**
# 
# 
# * **Correlation of the features has been captured using heatmap.**

# ## INSIGHTS FROM EDA:
# 
# * **The distribution of 12 clinical features and 22 histopathological features in the dataset with 366 instances is depicted using histogram and KDE plot.**
# 
# 
# * **Dieseases like Lichen_Planus, Cronic_Dermatitis,and Pityriasis_Rosea does not show any relation of tranfer within family members. Also, theses diseases can occurs in early ages till the age of 70 years is also as seen from the distribution.**
# 
# 
# * **Diseases like Psoriasis and Pityriasis_rubra_pilaris shows strong relation of transfer from family and disease Seboreic_Dermatitis too shows a some relation of transfer within family members.**
# 
# 
# * **Disease like Pityriasis_rubra_pilaris shows its presence in early ages and till the age of 15 to 20 years only.**
# 
# 
# * **Diseases like Psoriasis and Seboreic_Dermatitis shows its presence in early ages and till the age of 70 years.**
# 
# 
# * **The Disease / target class count plot showed imbalanced values between diffrent classes which was balanced using combination of SMOTE and edited nearest neighbor technique.**
# 
# 
# * **Heatmap is used to find the correlation between the features too.**

# ## INSIGHTS FROM CLASSIFICATION MODELS:
# 
# ### Model-1: Logistic Classifier
# * **For the tuned Logistic classifer model, the  accuracy score for train and test data were 1 and 0.972 respectively.**
# 
# ### Model-2: KNN Classifier
# * **For the tuned KNN classifer model, the  accuracy score for train and test data were 1 and 0.936 respectively.**
# 
# ### Model-3: Support Vector Classifier
# * **For Support Vector classifer model, the  accuracy score for train and test data were 1 and 0.963 respectively.**
# 
# ### Model-4: Desicion Tree Classifier
# * **For Desicion Tree classifer model, the  accuracy score for train and test data were 1 and 0.972 respectively.**
# 
# ### Model-5: Random Forest Classifier
# * **For Random Forest classifer model, the  accuracy score for train and test data were 1 and 0.963 respectively.**
# 
# ### Model-6: Multinomial Naive Bayes Classifier
# * **For Random Forest classifer model, the  accuracy score for train and test data were 0.984 and 0.981 respectively.**
# 
# ### Model-7: Gradient Boost Classifier
# * **For Gradient Boost classifer model, the  accuracy score for train and test data were 1 and 0.963 respectively.**
# 
# ### Model-8: XGB Classifier
# * **For XGB classifer model, the accuracy score for train and test data were 1 and 0.963 respectively.**
# 
# ### Model-9: KNN Classifier
# * **For KNN classifier model, the accuracy score for train and test data were 1 and 0.963 respectively.**

# ## CONCLUSION:
# 
# * **Patients were first evaluated clinically with 12 features. Afterwards, skin samples were taken for the evaluation of 22 histopathological features.**
# 
# 
# * **From EDA, skin diseases Psoriasis and Pityriasis_rubra_pilaris shows strong relation of transfer from family members and disease Seboreic_Dermatitis too shows a some relation of transfer within family members. Skin disease like Pityriasis_rubra_pilaris shows its presence in early ages till the age of 15 to 20 years only. Skin diseases Psoriasis and Seboreic_Dermatitis shows its presence in early ages till the age of 70 years.**
# 
#  
# * **As observed during the t-SNE analysis, the class features '1', '3', and '4' exhibited overlapping patterns. This observation aligns with the findings from the confusion matrix and classification report, where it becomes evident that the model is encountering challenges in accurately classifying instances belonging to these classes. The model's performance in these specific classes is reflected in the confusion matrix and reflected in the metrics reported in the classification report.**
# 
# 
# * **The 9 machine learning models have been studied and tuned and evaluated for acheiving better performance of the model using various metrices and accuracy scores.**
# 
# 
# * **The Multinominal Naive Bayes classifier Model has given a better accuracy Score and F1 Score compared to other classifier models and hence recommend to use Multinominal Naive Bayes classifier Model in order for the doctors to identify the skin diseases of the patient at the earliest.**

# ## CHALLENGES FACED:
# 
# * **The age feature had NAN and zero values which were replaced using the median of the age feature.**
# 
# * **Various classifier models were studied and evaluated using accuracy score, classification report and confusion matrix.**
# 
# * T**he model with highest accuracy score and F1 Score was obtained in Multinominal Naive Bayes classifier and hence will be the deciding factor.**

# In[ ]:




