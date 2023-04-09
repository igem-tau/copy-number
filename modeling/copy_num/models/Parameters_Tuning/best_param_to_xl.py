
import pandas as pd
import openpyxl
import os
def write_to_xl(dic,model_name):
    df1=pd.DataFrame(dic.values())
    df1['scores']=dic.keys()
    xl_name=f'{model_name}_best_params.xlsx'
    target_file=os.path.join(os.getcwd(), xl_name)
    try:
        df2 = pd.read_excel(target_file)
    except:
        print(f'create xl {xl_name}')
        workbook = openpyxl.Workbook()
        workbook.save(target_file)
        df2=pd.read_excel(target_file)
    df= pd.concat([df2, df1], ignore_index=True)
    df.to_excel(target_file,index=False)


def get_best_params_set(model_name):
    xl_name = f'{model_name}_best_params.xlsx'
    df=pd.read_excel(xl_name)
    score=df['scores'].max()
    params=df.iloc[df['scores'].idxmax(),:].dropna().drop('scores')
    print(f'Best params for {model_name} model are:\n{params}\nAnd their predicted score is {score}')
    return(params)
## example##
# dic={1200:{'r':4,'t':300}}
# model_name='xgb'
# write_to_xl(dic,model_name)
# a=get_best_params_set(model_name)




