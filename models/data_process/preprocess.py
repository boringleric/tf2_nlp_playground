import pandas as pd


def load_cls_csv(file_config_list, ischinese=False):
    '''
    file_config_list：文件配置列表，包含(文件路径, 文件编码, 文件分隔符)
    '''
    all_data, all_labels = [], []
    cnt = 0
    label_dict = {}

    for config in file_config_list:
        path, encoding, sep = config
        df_ext = pd.DataFrame(pd.read_csv(path, encoding=encoding, sep=sep))

        if ischinese:
            ext_list = df_ext["相关问"].tolist()
            ext_std_list = df_ext["意图"].tolist()
        else:
            ext_list = df_ext["ext"].tolist()          
            ext_std_list = df_ext["std"].tolist()

        for index, content in enumerate(ext_list):   
            if "|" not in ext_std_list[index]:
                all_data.append(content)
                all_labels.append(ext_std_list[index])

    label_set = sorted(set(all_labels))
    for label in label_set:
        label_dict[label] = cnt
        cnt += 1

    return all_data, all_labels, label_dict


def load_sem_data(file_config_list):
    '''
    加载非cls类数据，直接分类
    '''
    all_data, all_labels = [], []
    cnt = 0
    label_dict = {}

    for config in file_config_list:
        path, encoding, sep = config
        df_ext = pd.DataFrame(pd.read_csv(path, encoding=encoding, sep=sep))

        ext_list = df_ext["text"].tolist()
        ext_std_list = df_ext["label"].tolist()

        for index, content in enumerate(ext_list):   
            all_data.append(content)
            all_labels.append(ext_std_list[index])

    label_set = sorted(set(all_labels))
    for label in label_set:
        label_dict[label] = cnt
        cnt += 1

    return all_data, all_labels, label_dict


def load_ner_data(filename):
    """
    加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    categories = set()

    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            splitlist = l.split('\n')
            for i, c in enumerate(splitlist):
                if c == "":
                    continue
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'I':
                    d[-1][1] = i
                elif flag[0] == 'E':
                    d[-1][1] = i
                elif flag[0] == 'S':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
            D.append(d)

    categories = list(sorted(categories))
    
    return D, categories