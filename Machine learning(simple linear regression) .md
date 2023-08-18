## simple linear regression


``````python
df
``````




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YearsExperience</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.1</td>
      <td>39343.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.3</td>
      <td>46205.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.5</td>
      <td>37731.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>43525.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.2</td>
      <td>39891.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.9</td>
      <td>56642.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.0</td>
      <td>60150.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.2</td>
      <td>54445.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3.2</td>
      <td>64445.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.7</td>
      <td>57189.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3.9</td>
      <td>63218.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4.0</td>
      <td>55794.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4.0</td>
      <td>56957.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4.1</td>
      <td>57081.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4.5</td>
      <td>61111.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4.9</td>
      <td>67938.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5.1</td>
      <td>66029.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5.3</td>
      <td>83088.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5.9</td>
      <td>81363.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>6.0</td>
      <td>93940.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>6.8</td>
      <td>91738.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7.1</td>
      <td>98273.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7.9</td>
      <td>101302.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>8.2</td>
      <td>113812.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8.7</td>
      <td>109431.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>9.0</td>
      <td>105582.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>9.5</td>
      <td>116969.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>9.6</td>
      <td>112635.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>10.3</td>
      <td>122391.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>10.5</td>
      <td>121872.0</td>
    </tr>
  </tbody>
</table>
</div>



### spliting data set into training and testing
20% of testing and 80% of training data\
important to create 2d arrays


```python
 x = df[['YearsExperience']]
 y =df[['Salary']]
```

x.head()

### import library



```python
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size = 0.2 , random_state = 0)
```

### fit linear regression model


```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model = model.fit(x_train , y_train)
```


```python
model
```




<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>



### plotting (training model)



```python
import matplotlib.pyplot as plt
plt.scatter(x_train  ,y_train)
plt.plot(x_train , model.predict(x_train), color = 'Green')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.title('train plot')
```




    Text(0.5, 1.0, 'train plot')




    
![png](output_12_1.png)
    


### plotting testing model


```python
import matplotlib.pyplot as plt
plt.scatter(x_test  ,y_test)
plt.plot(x_test , model.predict(x_test), color = 'Green')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.title('test plot')
```




    Text(0.5, 1.0, 'test plot')




    
![png](output_14_1.png)
    


### check the model fitness


```python
print('score for train data =' ,model.score(x_train , y_train))

```

    score for train data = 0.9411949620562126
    


```python
print('score for test data =' ,model.score(x_test , y_test))
```

    score for test data = 0.988169515729126
    

### prediction of unknown values (change of year of experience)


```python
model.predict([[3],[6],[7],[12]])
```

    




    array([[ 54717.82453082],
           [ 82655.549911  ],
           [ 91968.12503773],
           [138531.00067138]])




```python
year_of_experience = ([3], [5],[8],[12])
```


```python
model.predict(year_of_experience)
```





    array([[ 54717.82453082],
           [ 73342.97478427],
           [101280.70016446],
           [138531.00067138]])




```python

```
