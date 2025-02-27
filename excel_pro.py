import pandas as pd
import re

# ----------------------------------------------------------------
# 作用是将acc.txt文件中的准确率提取出来，保存到excel中，以方便origin画图
# ----------------------------------------------------------------

# Define the path to the text file and the output Excel file
text_file_path = 'D:/python object/federate learning/Semi_decentralized_FD/result_DS-FL/acc.txt'
excel_file_path = 'D:/python object/federate learning/Semi_decentralized_FD/result_DS-FL/accuracy_data.xlsx'

# Initialize a list to hold all extracted accuracy data
accuracy_data = []

# Open the text file and read lines
with open(text_file_path, 'r') as file:
    for line in file:
        # Use regular expression to find data within square brackets
        match = re.search(r'\[(.*?)\]', line)
        if match:
            # Extract the string inside the brackets, split by comma, and convert to float
            data = [float(x.strip()) for x in match.group(1).split(',')]
            accuracy_data.append(data)

# Create a DataFrame from the extracted data
df = pd.DataFrame(accuracy_data)

# Save the DataFrame to an Excel file
df.to_excel(excel_file_path, index=False, header=False)

print("Data has been extracted and saved to Excel.")
