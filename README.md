# dataanalyticsAssignment1
**Data Analytics Assignment 1** 

45 

46 47 

50 

51 

52 

57 Several preprocessing steps were applied to ensure the data is in a 58 suitable format for modeling: 59 

60 

69 

70 

80 

# **Comprehensive Analysis and Classification of Suspects in Response to a High-Profile Incident: A Data Analytics Approach** 

## **Nagalla Devisri Prasad, Darapu Adhithya Shiva Kumar Reddy, Eeshwar Aditya** 

## **Abstract** 

This report details the development and evaluation of several classification models for predicting criminal behavior using a dataset of potential suspects. Initially, Exploratory Data Analysis (EDA) was performed to understand the dataset’s structure, identify patterns, and handle missing values. This analysis informed the subsequent model implementation. A custom Naive Bayes classifier was created and evaluated against standard models, including a scikit-learn Naive Bayes, Support Vector Machine (SVM), Decision Tree, and K-Nearest Neighbors (KNN). Performance was assessed using metrics such as accuracy, precision, recall, and F1 score. Additionally, an ensemble model was constructed to combine the strengths of individual classifiers and improve overall prediction accuracy. The results offer insights into the effectiveness of different models and highlight the potential of ensemble methods in enhancing predictive performance. 

## **1. Introduction** 

- 1 

2 In the field of data science and machine learning, classification mod3 els are crucial for predicting categorical outcomes based on input 4 features. This report focuses on the development and evaluation 5 of various classification models to predict criminal behavior from a 6 dataset of potential suspects. The dataset comprises various features, 7 including demographic and occupational information, with the target 8 variable indicating whether an individual is a criminal or not. 9 To ensure a robust model development process, Exploratory Data 10 Analysis (EDA) was conducted prior to implementing the classifica11 tion models. EDA involved analyzing the dataset to understand its 12 characteristics, uncover patterns, and identify potential issues such as 13 missing values or feature imbalances. This preliminary analysis pro14 vided valuable insights that guided the selection and implementation 15 of the classification algorithms. 

- 16 The report includes the following key components: 

- **educationno** : The numeric representation of the education 43 level. 44 

- **maritalstatus** : The marital status (e.g., Married, Single). 

- **occupation** : The type of occupation (e.g., Tech-support, Craftrepair). 

- **relationship** : Relationship status (e.g., Wife, Husband, Not-in48 family). 49 

- **race** : The race of the individual (e.g., White, Black). 

- **sex** : The gender of the individual (Male, Female). 

- **hoursperweek** : The number of hours worked per week. 

- **native** : Country of origin of the individual. 

- 53 

- • **Possibility** : Target variable indicating whether the individual 54 is likely to be involved in criminal activity (1 for criminal, 0 for 55 not criminal). 56 

## **2.2. Preprocessing Steps** 

- 17 • Implementation of a custom Naive Bayes classifier to handle 18 categorical and numerical features. 

- 19 • Evaluation of standard classifiers using scikit-learn, including 20 Naive Bayes, Support Vector Machine (SVM), Decision Tree, and 21 K-Nearest Neighbors (KNN). 

- 22 • Construction and assessment of an ensemble model designed to 23 improve predictive performance by combining multiple classi24 fiers. 

- 25 • Comparison of model performance based on metrics such as 26 accuracy, precision, recall, and F1 score. 

- 27 The goal of this assignment is to explore the effectiveness of differ28 ent classification approaches and demonstrate the potential benefits 29 of ensemble methods in improving prediction accuracy for criminal 30 behavior classification. The findings from this analysis aim to provide 31 a comprehensive understanding of how various models perform and 32 how they can be leveraged to enhance decision-making processes. 

## **2. Data Overview** 

- 33 

- 34 **2.1. Dataset Features** 

35 The dataset used in this analysis contains information about indi36 viduals, including demographic and work-related attributes. It is 37 structured as a table with the following features: 

- 38 • **age** : The age of the individual. 

- 39 • **workclass** : The type of employment (e.g., Private, Self40 employed, Government). 

- 41 • **education** : The education level of the individual (e.g., Bache42 lors, Masters). 

## • **Handling Missing Values** : 

- Records with significant missing data were removed to 61 prevent biases caused by incomplete information. 62 

- For categorical variables, missing values were imputed 63 with the most frequent category. 64 

- For continuous variables, missing values were imputed 65 using the mean based on the distribution of the data. 66 

- Outliers contributing to missing or erroneous values were 67 investigated to determine whether they should be imputed 68 or removed. 69 

## • **Encoding Categorical Variables** : 

- Categorical variables with nominal data, such as _workclass_ , 71 _occupation_ , and _relationship_ , were one-hot encoded to cre72 ate binary features for each category. 73 

- Ordinal variables, such as _educationno_ , were label-encoded 74 to preserve the inherent order in the data. 75 

- The target variable _Possibility_ was also encoded such that 76 probabilities greater than 0.5 were classified as 1 (criminal), 77 and those less than or equal to 0.5 were classified as 0 (non78 criminal). 79 

## • **Handling Duplicate Records** : 

- Duplicate rows were checked and removed to ensure the 81 integrity of the dataset and avoid skewing the model results. 82 

- **–** Comparisons of similar records were made to ensure sub83 tle duplicates were detected, especially when values were 84 missing or mislabeled. 85 

Creative Commons CC BY 4.0 

College name **May 21, 2024** L<sup>A</sup> TEX Template **1** –6 



<!-- Start of picture text -->
Pie. chart of sex<br>Female<br>Male<br><!-- End of picture text -->



<!-- Start of picture text -->
Pie chart of Possibility<br>>0.5<br><=0.5<br><!-- End of picture text -->



<!-- Start of picture text -->
Distribution of maritalstatus<br>14000<br>12000<br>10000<br>#58 8000<br>6000<br>4000<br>2000<br>0<br><é& soSseS& & YFegSs SsRa asxse&$<br>¥ RS v<br>w<br>maritalstatus<br><!-- End of picture text -->



<!-- Start of picture text -->
Distribution of relationship<br>12000<br>10000<br>8000<br>€58 6000<br>4000<br>2000<br>0)<br>~ cSSs4 ~- § & ssESo RS<<& sé&Se<br>relationship<br><!-- End of picture text -->



<!-- Start of picture text -->
Distribution of race<br>25000<br>20000<br>2 15000<br>35<br>10000<br>5000<br>0) & eS é os<br>SS s &xs ss§es &<br>s aS<br>° eS<br>race<br><!-- End of picture text -->



<!-- Start of picture text -->
Distribution of workclass<br>20000<br>15000<br>€58<br>10000<br>5000<br>0)<br>Ss ce Ss Ss & cy)<br>ea & & & & €<br>SS cd Rad ef<br>aee & gS s<br>workclass<br><!-- End of picture text -->



<!-- Start of picture text -->
Density plot of age<br>0.025<br>0.020<br>&8>2 0.015<br>0.010<br>0.005<br>0.000<br>20 40 60 80<br>age<br><!-- End of picture text -->



<!-- Start of picture text -->
Density plot of educationno<br>oo<br>0.35<br>0.30<br>0.25<br>Fy£ 0.20<br>0.15<br>0.10<br>0s<br>0.00<br>0.0 2.5 5.0 75 10.0 12.5 15.0 175<br>educationno<br>Density plot of capitalgain<br>0.00040<br>0.00035<br>0.00030<br>0.00025<br>2<br>2 0.00020<br>&<br>0.00015<br>0.00010<br>0.00005<br>0.00000 0 =20000 40000 60000 80000 100000<br>capitalgain<br><!-- End of picture text -->



<!-- Start of picture text -->
Correlation Matrix Heatmap 100<br>age RB 208 0.00 008 1028 0.01 1025 002 008 010 -0.00 0.24<br>workclass - 0.08 | .00 FR 0.04 -0.03 0.02 -0.07 0.04 0.07 0.05 0.01 0.02 0.75<br>education --0.00 0.02| +00 ES 0.04 -0.04 -0.01 0.01 -0.03 0.06 0.08 0.08 0.50<br>educationno- 0.04 0.04 0.35[00 0.09 -0.09 0.03 0.01 0.15 0.09 0.34<br>relationship --0.25 -0.07 -0.01 -0.09 0.18 -o0s Pon 0.25 -0.01 -0.25 - 0.00<br>race -0.02 0.04 0.01 0.03 -0.07 0.00 0.21 BR] 009 005 013 007 --0.25<br>sex-0.08 007 003 001 0.12 006 (PEt) 009 FR 023 000 022 - -0.50<br>hoursperweek -0.10 0.05 0.06 0.15 -0.19 0.02 -0.25 0.05 o23 Joo: 023<br>native --0.00 0.01 008 099 0.03 0.00 0.01 013 0.00 001 BRM 002 -0.75<br>Possibility- 0.24Re0.02 0.08 0.34ee-0.19 0.05 -0.25 0.07eee0.22 0.23 0.02 1.00<br>®es4 25¢ 8 e4 25¢ 2 yo@ ¥§ weBeew DB<br>£8 2 8 § 5 8 3<br>238 8 g & e &<br>B 8 2 23<br><!-- End of picture text -->



<!-- Start of picture text -->
Density plot of capitalloss<br>0.007<br>0.006<br>0.005<br>2a= 0.004<br>0.003<br>0.002<br>0.001<br>0.000 0) 1000 —2000 3000 4000<br>capitalloss<br><!-- End of picture text -->



<!-- Start of picture text -->
Density plot of hoursperweek<br>0.12<br>0.10<br>0.08<br>2&Fd4 0.06<br>0.04<br>0.02<br>0.00 0 20 40 60 80— 100<br>hoursperweek<br><!-- End of picture text -->

Comprehensive Analysis and Classification of Suspects in Response to a High-Profile Incident: A Data Analytics Approach 

Author last name et al. 

- 143 • The class probabilities were estimated based on the frequency 144 of each class in the dataset. 

144 145 • Likelihoods were calculated separately for categorical and nu146 merical features. 

- 147 • For numerical features, Gaussian assumptions were used to 148 estimate the likelihoods. 149 • Predictions were made by computing the posterior probabilities 150 for each class and selecting the class with the highest probability. 

- 151 This implementation ensures that all components of the Naive 152 Bayes model are handled manually, providing a clear understanding 153 of the underlying mechanisms of the classifier. 

## **4. Model Implementations and Evaluations** 

154 

155 In this section, we present the implementations and evaluations of 156 several classification models using both custom and library-based ap157 proaches. The models include a Naive Bayes classifier implemented 158 from scratch, and Naive Bayes, Support Vector Machine (SVM), Deci159 sion Tree, and K-Nearest Neighbors (KNN) classifiers implemented 160 using libraries. 

## **4.1. Naive Bayes Implementation using Libraries** 

161 

162 The Naive Bayes classifier was implemented using the `GaussianNB` 163 class from the `sklearn.naive_bayes` library. The process involved 164 the following steps: 

- 165 • The dataset was split into training and test sets using 166 `train_test_split` . 

- 167 • The `GaussianNB` model was initialized and trained on the train168 ing set. 

   - Predictions were made on the test set. 

169 

- 170 • Model performance was evaluated using accuracy, precision, 171 recall, and F1 score metrics. 

The results of the evaluation were: 

172 

- Accuracy: `0.76` 

173 

- Precision: `0.53` 

174 

- Recall: `0.70` 

175  176 • F1 Score: `0.60` 

## **4.2. SVM Implementation** 

177 

178 The Support Vector Machine (SVM) classifier was implemented using 179 the `SVC` class from the `sklearn.svm` library. The implementation 180 included: 

   - Splitting the dataset into training and test sets. 

- 181 

- Initializing and training the `SVC` model. 

182 

- Making predictions on the test set. 

183 

- 184 • Evaluating the model using accuracy, precision, recall, and F1 185 score. 

The evaluation results were: 

186 

- Accuracy: `0.80` 

187 

- Precision: `0.72` 

188 

- Recall: `0.36` 

189 

- F1 Score: `0.48` 

190 

## **4.3. Decision Tree Implementation** 

191 

192 The Decision Tree classifier was implemented using the 

193 `DecisionTreeClassifier` class from the `sklearn.tree` library. 194 The steps included: 

- Splitting the dataset into training and test sets. 

195 

- Initializing and training the `DecisionTreeClassifier` model. 

196 

- Making predictions on the test set. 

197 

- 198 • Evaluating the performance of the model using accuracy, preci199 sion, recall, and F1 score. 

The results of the evaluation were: 

200 • Accuracy: `0.78` 201 • Precision: `0.56` 202 • Recall: `0.55` 203 • F1 Score: `0.55` 204 **4.4. KNN Implementation** 205 The K-Nearest Neighbors (KNN) classifier was implemented using the 206 `KNeighborsClassifier` class from the `sklearn.neighbors` library. 207 The implementation followed these steps: 208 • The dataset was split into training and test sets. 209 • The `KNeighborsClassifier` model was initialized and trained. 210 • Predictions were made on the test set. 211 • The model’s performance was evaluated using accuracy, preci212 sion, recall, and F1 score metrics. 213 

The evaluation results were: 

214 

- Accuracy: `0.79` 215 

- • Precision: `0.61` 216 • Recall: `0.55` 217 • F1 Score: `0.58` 218 

**5. Comparison of Custom and Library-Based Naive Bayes** 219 **Implementations** 220 

In this section, we compare the performance of a custom Naive Bayes 221 classifier implemented from scratch with a Naive Bayes classifier 222 implemented using the `sklearn.naive_bayes.GaussianNB` library. 223 The comparison focuses on various performance metrics, including 224 accuracy, precision, recall, and F1 score. 225 

## **5.1. Custom Naive Bayes Classifier** 

226 

The custom Naive Bayes classifier was implemented with the follow227 ing features: 228 

- Calculation of prior probabilities for each class based on the 229 training data. 230 

- • Computation of likelihoods for categorical features using 231 frequency-based estimates. 232 

- • Handling of numerical features by calculating means and vari233 ances, and applying Gaussian probability density functions. 234 

- • Use of logarithms to prevent underflow and improve numerical 235 stability. 236 

The performance of the custom Naive Bayes classifier was evalu237 ated on the training data. The results were: 238 

- Accuracy: `0.81` 

 239 • Precision: `0.59` 240 • Recall: `0.77` 241 • F1 Score: `0.67` 242 

## **5.2. Library-Based Naive Bayes Classifier** 

243 

The Naive Bayes classifier was also implemented using the 244 `GaussianNB` class from the `sklearn.naive_bayes` library. This im245 plementation includes: 246 

- Automated computation of class priors and feature likelihoods. 

247 248 

- Built-in handling of numerical and categorical features. 

- Utilization of optimized algorithms for model training and pre249 diction. 250 

The performance of the library-based Naive Bayes classifier was 251 evaluated on the test set. The results were: 252 

252 

- Accuracy: `0.76` 

253 

- Precision: `0.53` 

254 

- Recall: `0.70` 

255 

- F1 Score: `0.60` 256 

4–6 



<!-- Start of picture text -->
10 Model Performance Comparison<br>08 0.80 0.76 0.80 078 or oer<br>06<br>a<br>& 04<br>02<br>00<br>s s ae &<br>eof<br>Models<br><!-- End of picture text -->

Comprehensive Analysis and Classification of Suspects in Response to a High-Profile Incident: A Data Analytics Approach 

Author last name et al. 

359 **7.2. Discussion** 

360 From the comparison chart and the performance metrics, we observe 361 the following insights: 

- 362 • **Custom Naive Bayes** achieved an accuracy of 0.80, which is 363 comparable to the accuracy of the SVM and slightly lower than 364 the Ensemble Model. It also shows strong performance in terms 365 of recall (0.75), which indicates good sensitivity in identifying 366 positive cases. 

- 367 • **Naive Bayes (sklearn)** has the lowest accuracy (0.76) among 368 the models. Despite a relatively high recall (0.70), it suffers 369 from lower precision (0.53), which suggests a higher rate of false 370 positives. 

- 371 • **SVM** performed well in terms of precision (0.72) but has a lower 372 recall (0.36), indicating it is less effective at capturing all positive 373 cases. The overall F1 score is also the lowest (0.48), reflecting a 374 trade-off between precision and recall. 

- 375 • **Decision Tree** provided a balanced performance with accuracy 376 (0.78) and an F1 score (0.56) that are intermediate compared to 377 other models. Its precision and recall are reasonably balanced. 

- 378 • **KNN** showed an accuracy of 0.79 with a decent balance between 379 precision (0.61) and recall (0.56). Its F1 score (0.58) indicates a 380 good trade-off between precision and recall. 

- 381 • **Ensemble Model** outperforms the individual classifiers with an 382 accuracy of 0.81 and an improved F1 score (0.62). The ensemble 383 approach benefits from combining multiple models, leading to 384 better overall performance, especially in precision. 

385 In conclusion, the Ensemble Model provides the best overall accu386 racy and balances precision and recall effectively. However, the choice 387 of the best model can vary depending on the specific requirements of 388 the task, such as whether higher precision or recall is prioritized. 

6–6 
