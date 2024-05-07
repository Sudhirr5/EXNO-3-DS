# EX:3 Feature Encoding and Transformation

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Developed by : SUDHIR KUMAR .R
Register number : 212223230221
```
```python
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```

![322713287-6690fef9-9ef6-49fb-a97b-2f13b06585d2](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/6293daec-3c6f-4d92-a8c4-a7e67b96391c)

## ORDINAL ENCODER
```python
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![322713314-60cf861a-4491-4ba7-88de-6ea2e2c65414](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/3993ec73-14ef-49be-9a89-5ce91104b367)

```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![322713375-88ab6c7c-2856-47d4-a664-b9d55db9493e](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/f5e7180b-1aa7-4740-a56e-928465f705ea)

## LABEL ENCODER
```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![322713413-ad5cb5da-7fff-4e73-9e61-d389f1aceca6](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/2fe9fdf9-6f91-45a7-b63e-9d03c4c28dc3)

## ONEHOT ENCODER
```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```
```python
df2=pd.concat([df2,enc],axis=1)
df2
```

![322713458-28e7e838-ee21-4bce-a468-f907a96228f1](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/ba49595a-a2d5-4d00-9a29-6ce7d4530c40)

```python
pd.get_dummies(df2,columns=["nom_0"])
```

![322713497-2fc9da87-853d-4ecc-b8f0-a92464d0187f](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/d7d8848a-ab6c-44e3-8a94-d977651c0438)

## BINARY ENCODER
```python
pip install --upgrade category_encoders
```
![322713551-a7857f36-1ec0-4c3d-9cb2-0f18a80fa899](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/80439712-4cb4-46cc-bd1d-b02231bad4fe)

```python
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![322713577-0c48f712-ff33-42bd-b05e-59140aab98ad](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/26450ef6-b2aa-48eb-a3e9-bd62f11a0fce)

```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```

![322713823-1a4c7e12-cbf6-4348-8644-19f0275cbc25](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/0f6c6131-32a8-4f37-817a-b1822b662f64)

## TARGET ENCODER
```python
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![322713856-e10f4494-fe8b-4d73-b334-604d5f87f034](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/e92088d6-5b74-421a-a157-3a8daa8c538c)

## DATA TRANSFORMATION
```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/content/Data_to_Transform.csv')
df
```

![322713892-572af179-b671-46cb-809b-70f7aed9f578](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/756bc1c7-93db-4bbb-ba63-42969c200f50)

```python
df.skew()
```

![322713936-309e3381-95a8-411b-9273-03aedb302f05](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/d89109a6-b076-42ce-92ac-92c8d2df865e)

```python
np.log(df["Highly Positive Skew"])
```

![322713974-d4a06498-bc6c-4be1-9ca9-af278cf86a76](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/142cdb7b-cfa2-477c-89a6-1c2334151a8a)

```python
np.reciprocal(df["Moderate Positive Skew"])
```

![322713990-0363ca0d-ffb2-470b-8940-adf073c91c93](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/a6a8cf72-da2a-49a2-a84a-cbb1b4f46dcf)

```python
np.sqrt(df["Highly Positive Skew"])
```

![322714022-0281a301-1396-43f6-8cf1-3ceab8d42a60](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/5c45f29d-3f2a-40fe-ac92-9b58f487b1e8)

```python
np.square(df["Highly Positive Skew"])
```

![322714052-dc4d3202-425d-444c-a05c-bb0f8bf987fe](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/e8f28c48-8e05-48d7-bed6-26c7e79df7c2)

```python
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![322714126-c6d89ba9-2aef-478b-ae3f-e5bd6c6402b1](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/c8a11db9-a553-4186-bd67-c9f71ab59cd9)

```python
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```

![322714190-3388c7d2-bc9c-4b2a-b7b5-94779280f895](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/78205689-0d94-4214-91bd-e3c01e1f9dc9)

```python
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![322714227-2e30cfe9-19c0-47d6-991c-01d1081df635](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/1a77388f-f42e-42b5-bc7e-7f10b93e5aa6)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![322714330-d9c03df0-957e-44e1-99cd-abcf4dca1af0](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/940d5470-2027-4a7a-bbb8-00742e988161)

```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![322714384-093aceda-03cc-465b-af35-27118e987b23](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/1b3cf48e-6bfa-4f1d-8b80-71f9a918f9bf)

```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![322714454-5761c4d2-81df-43af-95eb-16d8fbb84113](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/27a04677-d0e5-4f1a-b3cc-b7acd3131006)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![322714506-35b9aad4-e65d-48cd-9965-01965232bfdf](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/b99a0581-b46d-4afb-aea2-c6525cf71760)

```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![322714594-0d08e666-5b25-4e54-8bb2-680d2b097e4a](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/3f09a847-96d6-47a9-86e8-23f53d28c5b4)

```python
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```

![322715090-1b2895a0-fb43-40c3-9af3-efd8917d9e6d](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/2074efdd-1b0d-4a5b-bbb0-3ec71b9f6808)

```python
dt=pd.read_csv("/content/titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```

![322715022-3fec8566-0fbc-4f28-93b7-7e731dd97df5](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/48ce5181-b027-4d79-a5b7-8f392117fad7)

```python
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```

![322714954-4fb5c326-f125-4dd4-ac83-6bac0abb7360](https://github.com/Sudhirr5/EXNO-3-DS/assets/139332214/7564536a-3a2f-45e2-8058-9fbbd8646bfe)

# RESULT:
Thus perform Feature Encoding and Transformation process is executed successfully.

       
