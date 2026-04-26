import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn


class Fansen(nn.Module):
    def __init__(self):
        super().__init__()
        self.mo = nn.Sequential(
            nn.Linear(8, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),

            nn.Linear(32, 1),

        )

    def forward(self, x):
        return self.mo(x)

fansen=Fansen()
fansen.load_state_dict(torch.load("food_time_model.pth"))
fansen.eval()

df=pd.read_csv("F:\pycharm\Food Delivery Time\delivery_dataset.csv")

str_to_int=["Traffic_Level","weather_description","Type_of_order","Type_of_vehicle"]
for i in str_to_int:
    le=LabelEncoder()
    df[i]=le.fit_transform(df[i].astype(str))

text_value=df.drop("TARGET",axis=1).values
ture_value=df["TARGET"].values

scaler=StandardScaler()
text_value=scaler.fit_transform(text_value)

scaler_y = StandardScaler()
ture_value_scaled = scaler_y.fit_transform(ture_value.reshape(-1, 1))

text_value=torch.tensor(text_value,dtype=torch.float32)

with torch.no_grad():
    xun_value=fansen(text_value).squeeze().numpy()


xun_value = scaler_y.inverse_transform(xun_value.reshape(-1, 1)).flatten()


pingjun_error=mean_absolute_error(ture_value,xun_value)
junfanggen_error=np.sqrt(mean_squared_error(ture_value,xun_value))

print(f"平均误差：{pingjun_error:.2f}分钟")
print(f"均方根误差：{junfanggen_error:.2f}分钟")

results=pd.DataFrame({
    "真实时间":ture_value,
    "预测时间":np.round(xun_value,2),
    "误差":np.round(np.abs(ture_value-xun_value),2)
})
print("所有对比\n")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(results)