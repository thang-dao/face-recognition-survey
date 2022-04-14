import pandas as pd
import shutil
import os


def evaluate(result_file, vis=False):
    print('Evaluate %s' % result_file)
    df = pd.read_csv(result_file)
    correct_count = 0
    for index, row in df.iterrows():
        correct_flag = False
        if row['label'] == 'NO' and not row['pred']:
            correct_flag = True
        if row['label'] == 'YES' and row['pred']:
            correct_flag = True
        if correct_flag:
            correct_count += 1
        if vis:
            save_path = result_file[:-4] + '/' + str(correct_flag)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            src = row['id']
            shutil.copytree(src, save_path + '/' + row['label'] + '/' + os.path.basename(src))
    print('accuracy %f' % (correct_count / df.shape[0]))


if __name__=='__main__':
    evaluate('Facenet.csv', vis=True)
    evaluate('DeepFace.csv', vis=True)
