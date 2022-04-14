from deepface import DeepFace
import pandas as pd
import os


def test_model(data_path, model, save_name):
    data = []
    for ty in ['YES', 'NO']:
        path_type = os.path.join(data_path, ty)
        for id in os.listdir(path_type):
            path_id = os.path.join(path_type, id)
            img_list = os.listdir(path_id)
            if not img_list:
                print('Empty folder %s' % path_id)
                continue
            if len(img_list) != 2:
                print('Size folder not enough %d' % len(img_list))
                continue
            img1 = os.path.join(path_id, img_list[0])
            img2 = os.path.join(path_id, img_list[1])
            print(ty, id, img1, img2)
            result = DeepFace.verify(img1_path=img1, img2_path=img2,
                                     model_name=model, enforce_detection=False)
            data.append([path_id, ty, result['verified']])
    df = pd.DataFrame(data, columns=['id', 'label', 'pred'])
    df.to_csv(save_name + '.csv', index=True)

if __name__=='__main__':
    data_path = '../dataset/crop_dataset'

    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
    test_model(data_path=data_path, model=models[1], save_name=models[1])
    test_model(data_path=data_path, model=models[4], save_name=models[4])

        