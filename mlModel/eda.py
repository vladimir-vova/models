import pandas as pd

if __name__ == '__main__':
    DATASET = 'https://raw.githubusercontent.com/aiedu-courses/eda_and_dev_tools/refs/heads/main/datasets/abalone.csv'

    df = pd.read_csv(DATASET)

    diametr = df['Diameter'].median()
    df['Diameter'].fillna(diametr, inplace=True)

    whole = df['Whole weight'].median()
    df['Whole weight'].fillna(whole, inplace=True)

    shell = df['Shell weight'].median()
    df['Shell weight'].fillna(shell, inplace=True)

    df['Sex'] = df['Sex'].replace('f', 'F')

    df.to_csv('preprocessed_data.csv', index=False)
