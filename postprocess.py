import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
import queue
from tqdm import tqdm
import multiprocessing


def postprocess(data_dir):
    files = os.listdir(data_dir)
    for f in files:
        seg = sitk.ReadImage(os.path.join(data_dir, f))
        seg = sitk.GetArrayFromImage(seg)
        post_seg = np.zeros_like(seg[0])
        # print(post_seg.shape)
        post_seg[seg[0] == 1] = 1
        post_seg[seg[1] == 1] = 2
        post_seg[seg[2] == 1] = 3
        post_seg[seg[3] == 1] = 4
        print(post_seg.max())
        post_seg = sitk.GetImageFromArray(post_seg)
        print(post_seg.GetSize())
        sitk.WriteImage(post_seg, f'./result/segment/{f}')


def BFS(data, isvisited, start):
    q = queue.Queue()
    cnt = 1
    i, j, k = start
    isvisited[i][j][k] = True
    mask = np.zeros_like(data)
    mask[i][j][k] = 1
    q.put(start)
    h, w, d = data.shape
    while not q.empty():
        i, j, k = q.get()
        for ti in [i - 1, i, i + 1]:
            if ti < 0 or ti >= h: continue
            for tj in [j - 1, j, j + 1]:
                if tj < 0 or tj >= w: continue
                for tk in [k - 1, k, k + 1]:
                    if tk < 0 or tk >= d: continue
                    if not isvisited[ti][tj][tk] and data[ti][tj][tk] == 1:
                        isvisited[ti][tj][tk] = True
                        q.put((ti, tj, tk))
                        cnt += 1
                        mask[ti][tj][tk] = 1

    return mask, cnt


def keep_max_connect(data):
    organ_num = data.shape[0]
    for oid in range(organ_num):
        isvisited = np.zeros_like(data[oid], dtype=bool)
        h, w, d = data[oid].shape
        max_area = 0
        max_mask = np.ones_like(data[oid])
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    if data[oid][i][j][k] == 1 and not isvisited[i][j][k]:
                        mask, area = BFS(data[oid], isvisited, (i, j, k))
                        if area > max_area:
                            max_area = area
                            max_mask = mask
        data[oid][max_mask == 0] = 0

    return data


def postprocess_v2(data_dir, files, pid):
    # files = os.listdir(data_dir)
    print('postprocessing: ', pid)
    for f in tqdm(files):
        seg = sitk.ReadImage(os.path.join(data_dir, f))
        seg = sitk.GetArrayFromImage(seg)
        # print(seg.shape)
        seg = keep_max_connect(seg)
        post_seg = np.zeros_like(seg[0])
        post_seg[seg[0] == 1] = 1
        post_seg[seg[1] == 1] = 2
        post_seg[seg[2] == 1] = 3
        post_seg[seg[3] == 1] = 4
        print('pid: ', pid, f, post_seg.max())
        post_seg = sitk.GetImageFromArray(post_seg)
        print('pid: ', pid, f, post_seg.GetSize())
        sitk.WriteImage(post_seg, f'./result/segment/{f}')


def multiprocess(data_dir):
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('nii.gz')]
    num_process = 8
    files_list, p_list = [], []
    num = len(files) // num_process
    start = 0
    for i in range(num_process):
        if i < num_process - 1:
            tmp = files[start: start + num]
        else:
            tmp = files[start:]
        start += num
        p = multiprocessing.Process(target=postprocess_v2, args=(data_dir, tmp, i))
        p_list.append(p)

    for p in p_list:
        p.start()

    for p in p_list:
        p.join()


def postprocess_image(files, pid):
    # files = os.listdir(data_dir)
    print('postprocessing: ', pid)
    for f in tqdm(files):
        # seg = sitk.ReadImage(os.path.join(data_dir, f))
        # seg = sitk.GetArrayFromImage(seg)

        seg = keep_max_connect(f)
        post_seg = np.zeros_like(seg[0])
        post_seg[seg[0] == 1] = 1
        post_seg[seg[1] == 1] = 2
        post_seg[seg[2] == 1] = 3
        post_seg[seg[3] == 1] = 4
        print('pid: ', pid, f, post_seg.max())
        post_seg = sitk.GetImageFromArray(post_seg)
        print('pid: ', pid, f, post_seg.GetSize())
        sitk.WriteImage(post_seg, f'./result/segment/{f}')


def multiprocess_image(seg_images):
    # files = os.listdir(data_dir)
    # files = [f for f in files if f.endswith('nii.gz')]
    num_process = 8
    files_list, p_list = [], []
    num = len(seg_images) // num_process
    start = 0
    for i in range(num_process):
        if i < num_process - 1:
            tmp = seg_images[start: start + num]
        else:
            tmp = seg_images[start:]
        start += num
        p = multiprocessing.Process(target=postprocess_image, args=(tmp, i))
        p_list.append(p)

    for p in p_list:
        p.start()

    for p in p_list:
        p.join()


if __name__ == '__main__':
    # postprocess('./model/tmp_data')
    # postprocess_v2('./segment')
    os.makedirs('./result/segment', exist_ok=True)
    multiprocess('./model/tmp_data')
    # data_dir = '/public/pazhou/pazhou_data/preliminary_test/'
    # files = os.listdir(data_dir)
    # for f in files:
    #    image = sitk.ReadImage(os.path.join(data_dir, f))
    #    sitk.WriteImage(image, f'./test_images/{f}')

    # df = pd.read_csv('./result_prob.csv')
    # new_df = pd.DataFrame()
    # new_df['ID'] = df['ID']
    # new_df['liver'] = np.zeros(len(df))
    # new_df['spleen'] = np.zeros(len(df))
    # new_df['left kidney'] = np.zeros(len(df))
    # new_df['right kidney'] = np.zeros(len(df))

    # new_df['liver'][df['liver'] >= 0.55] = 1
    # new_df['spleen'][df['spleen'] >= 0.55] = 1
    # new_df['left kidney'][df['left kidney'] >= 0.45] = 1
    # new_df['right kidney'][df['right kidney'] >= 0.45] = 1

    # new_df.to_csv('./output/submit/result.csv', index=False)

