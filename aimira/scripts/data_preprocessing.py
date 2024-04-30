from glob import glob
import csv

dirname = "R:\AIMIRA\AIMIRA_Database\LUMC"
file_ls = glob(dirname + "\*")
file_ls = [{'patient_id': i[-8:-4], 'category':i[-3:]} for i in file_ls]
column_names = ['patient_id', 'category']

csv_file = "patient_id_with_category.csv"
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=column_names)
    
    # 写入CSV文件的标题行
    writer.writeheader()
    
    # 循环写入数据行
    for row in file_ls:
        writer.writerow(row)
        
print(file_ls)