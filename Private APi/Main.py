import requests
import pandas as pd
import json
import csv


df = []
url = "https://www.forbes.com/forbesapi/person/billionaires/2023/position/true.json"



r = requests.get(url)
data = r.json()

for item in data['personList']['personsLists']:
    
    try:
        rank    =  item['rank']
    except:
        rank  =""
    try:
        name  =item['personName']
    except:
        name =  ""
    try:
        age = item['age']
    except:
        age  =""
    try:
        city  = item['city']
    except:
        city =  ""
    try:
        gender  = item['gender']
    except:
        gender = ""
    try:
        country = item['country']
    except:
        country = ""
    
    try:
        source = item['source']
    except:
        source = ""
        
    try:
        organization = item['organization']
    except:
        organization = ""
        
    try:
        
        year = item['year']
    except:
        year = ""
        
    try:
        month = item['month']
    except:
        month = ""
    
    try:
        title = item['title']
    except:
        title = ""
        
    try:
        finalWorth = item['finalWorth']
    except:
        finalWorth = ""
        
    try:
        category    = item['category']
    except:
        category  =""
        
        
    dic = {
        "Rank":rank,
        "Name":name,
        "Age":age,
        "City":city,
        "Gender":gender,
        "Country":country,
        "Source":source,
        "Organization":organization,
        "Title":title,
        "Year":year,
        "Month":month,
        "Networth":finalWorth,
        "Category":category,
         }
    
    df.append(dic)

df_2 = pd.DataFrame(df)
df_2.to_csv("billionaires.csv",index = False)







# fields = ["rank", "category", "personName", "country", "source", "industries", "year", 'organization', "title", "gender", "month", "finalWorth", "age"]








