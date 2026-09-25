import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X=np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
y=np.array([35,40,50,55,60,68,75,82])

X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.25,random_state=42
)
model=LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print("slope (b1):",model.coef_[0])
print("intercept(b0):",model.intercept_)
print("\n--- evaluation metrices ---")
print(f"mae :{mae:.2f}")
print(f"mse:{mae:.2f}")
print(f"rmse: {mae:.2f}")
print(f"r2:{mae:.2f}")

plt.figure(figsize=(7,5))
plt.scatter(X,y,color="blue",label="actual data")
plt.plot(X,model.predict(X),color="red",linewidth=2,label="regression line")
plt.xlabel("study Hours")
plt.ylabel("Exam score")
plt.title("study hours vs score")
plt.legend()
plt.grid(True,linestyle="--",alpha=0.6)
plt.show()