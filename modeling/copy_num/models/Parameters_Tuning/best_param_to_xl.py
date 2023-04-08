import pandas as pd
import os.path
from openpyxl import load_workbook

def best_params_to_xl(d_params, model_name):
    df = pd.DataFrame(d_params.values())
    df['score'] = d_params.keys()
    path = 'Best_params.xlsx'
    sheet_exists = os.path.isfile(path) and model_name in load_workbook(path).sheetnames
    with pd.ExcelWriter(path, mode='a' if sheet_exists else 'w') as writer:
        if sheet_exists:
            writer.book = load_workbook(path)
        df.to_excel(writer, sheet_name=f'{model_name} best params')

best_params_to_xl({1.02:{'f':131,'d':4}}, 'yam')




