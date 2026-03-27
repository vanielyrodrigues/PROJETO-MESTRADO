import pandas as pd

df = pd.read_csv('DUSTAINEW.csv', sep=';')

df_hora = pd.to_datetime(df['HoraNew'], format='%H:%M:%S').dt.time  #
df['DataNew'] = pd.to_datetime(df['DataNew'],dayfirst=True)
tipo_dados_hora_new = df['HoraNew'].dtype
tipo_dados_date_new = df['DataNew'].dtype

print(df_hora)