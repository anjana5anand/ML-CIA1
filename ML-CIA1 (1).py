#!/usr/bin/env python
# coding: utf-8

# In[17]:


import random
import matplotlib.pyplot as plt
import numpy as np
points = []
for i in range(0,1000):
    x=np.random.randint(1,1001,1000)
    points.append(x)
for i in range (0,1000):
    y=np.random.randint(1,1001,1000)
    points.append(y)
plt.scatter(x,y)


# In[22]:


import random
import matplotlib.pyplot as plt
import numpy as np
points = []
for i in range(0,1000):
    x=np.random.randint(1,1001,1000)
    points.append(x)
for i in range (0,1000):
    y=np.random.randint(1,1001,1000)
    points.append(y)
plt.scatter(x,y)
x_mean = np.mean(x)
y_mean = np.mean(y)
n = len(x)
numerator = 0
denominator = 0
for i in range(n):
    numerator += (x[i] - x_mean) * (y[i] - y_mean)
    denominator += (x[i] - x_mean) ** 2
    
b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)
print(b1, b0)
x_max = np.max(x) + 100
x_min = np.min(x) - 100
x = np.linspace(x_min, x_max, 1000)
y = b0 + b1 * x
plt.plot(x, y, color='#00ff00', label='Linear Regression')
plt.legend()
plt.show()


# In[34]:


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.rand(1000, 1)
y = 2 + 3 * x + np.random.randn(1000, 1)

w = np.random.randn(1,1)
b = np.random.randn(1,1)
lr = 0.01

for i in range(1000):
    y_pred = x @ w + b
    dw = (2/1000)*x.T @ (y_pred - y)
    db = (2/1000)*np.sum(y_pred - y)
    w = w - lr*dw
    b = b - lr*db

plt.scatter(x, y)
plt.plot(x, x*w+b, 'r')
plt.show()


# In[32]:


import numpy as np
from sklearn.linear_model import LinearRegression
import random
import matplotlib.pyplot as plt
import numpy as np
points = []
for i in range(0,1000):
    x=np.random.randint(1,1001,1000)
    points.append(x)
for i in range (0,1000):
    y=np.random.randint(1,1001,1000)
    points.append(y)
plt.scatter(x,y)
model = LinearRegression().fit(x, y) 

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)

print('slope:', model.coef_) 

y_pred = model.predict(x)
print('Predicted response:', y_pred, sep='\n')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit);
plt.show()


# In[ ]:




