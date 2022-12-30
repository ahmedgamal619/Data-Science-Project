import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from apyori import apriori

datasetPath = str('')
numberofclusters=int(0)
minSupport=float(0)
minConfidence=float(0)

while( datasetPath != 'grc.csv'):
    datasetPath=str(input('Enter Data Set Name: '))
    if(datasetPath != 'grc.csv'):
     print('Invalid name')


while(not(0.001<=minSupport<=1)):
    minSupport=float(input('Enter the Minimum Support: '))
    if(not(0.001<=minSupport<=1)):
          print("Invalid Minimum Support")


while(not(0.001<=minConfidence<=1)):
    minConfidence=float(input('Enter the Minimum Confidence: '))
    if(not(0.001<=minConfidence<=1)):
     print('Invalid Minimum Confidence')

df=pd.read_csv(datasetPath)

d1={'Name': pd.Series(df.customer),'Age':pd.Series(df.age),'Total':pd.Series(df.total)}
d1 = pd.DataFrame(d1)
print(d1)


data=list(zip(df.age,df.total))
inertias=[]
for i in range(1,11):
    kmeans= KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)
plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

while(not(2<=numberofclusters<=4)):
    numberofclusters=int(input('Enter Number of Clusters: '))
    if(not(2<=numberofclusters<=4)):
     print('Invalid number')

km=KMeans(n_clusters=numberofclusters)
y_predicted=km.fit_predict(df[['age','total']])
pd.concat([df,pd.DataFrame(y_predicted)],axis=1)

plt.scatter(df.age[y_predicted==0],df.total[y_predicted==0])
plt.scatter(df.age[y_predicted==1],df.total[y_predicted==1])
plt.scatter(df.age[y_predicted==2],df.total[y_predicted==2])
plt.scatter(df.age[y_predicted==3],df.total[y_predicted==3])
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('age')
plt.ylabel('total')
plt.show()




#The Apriori library we are going to use requires our dataset to be in the form of a list of lists
records = []
for i in range(0, 9835):
    records.append(str(df.values[i][0]).split(','))

for i in range(0,9835):
    for j in range(0,len(records[i])):
        if records[i][j]=='nan':
         records.remove(records[i][j])


association_rules = apriori(records, min_support=minSupport, min_confidence=minConfidence, min_lift=3, min_length=2)
association_results = list(association_rules)
print(len(association_results))

for item in association_results:

    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")